"""Reference-validated forward sampler for the explicit hybrid teapot renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any,Literal

import torch

from reimpl.hybrid_teapot.environment_query import EnvironmentNerfQuery
from reimpl.renderers.mirror_sphere_renderer import CameraRays
from reimpl.sampling.cosine_hemisphere import sample_cosine_hemisphere

SamplingMode=Literal["cosine","bsdf","vmf","mis"]


@dataclass(frozen=True)
class DemoTrace:
    rgb:torch.Tensor
    hit:torch.Tensor
    surface:torch.Tensor
    sample_variance:float
    diagnostics:dict


@dataclass(frozen=True)
class PrimarySamples:
    origins:torch.Tensor
    directions:torch.Tensor
    x:torch.Tensor
    y:torch.Tensor
    resolution:int
    samples_per_source_pixel:int


def camera_rays_from_positions(emitter:Any,camera:Any,resolution:int,
                               x:torch.Tensor,y:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    source_h,source_w=int(camera.height.item()),int(camera.width.item()); scale=resolution/source_w
    fx,fy=float(camera.fx.item())*scale,float(camera.fy.item())*scale
    cx,cy=float(camera.cx.item())*scale,float(camera.cy.item())*scale
    local=torch.stack(((x-cx)/fx,-(y-cy)/fy,-torch.ones_like(x)),-1)
    c2w=camera.camera_to_worlds.float()
    d_ns=torch.sum(local[...,None,:]*c2w[:3,:3],dim=-1)
    d_ns=torch.nn.functional.normalize(d_ns,dim=-1).to(emitter.device)
    o_ns=c2w[:3,3].expand_as(d_ns).contiguous().to(emitter.device)
    return emitter.nerfstudio_to_external(o_ns.reshape(-1,3),d_ns.reshape(-1,3))


def center_camera_rays(emitter:Any,camera:Any,resolution:int) -> CameraRays:
    y,x=torch.meshgrid(torch.arange(resolution,dtype=torch.float32)+.5,
                       torch.arange(resolution,dtype=torch.float32)+.5,indexing="ij")
    o,d=camera_rays_from_positions(emitter,camera,resolution,x,y)
    return CameraRays(torch.empty(0),torch.empty(0),o.reshape(resolution,resolution,3),
                      d.reshape(resolution,resolution,3),resolution,resolution)


def jittered_primary_rays(emitter:Any,camera:Any,resolution:int,spp:int,seed:int,
                          border_size:int=2) -> PrimarySamples:
    """Independent uniform pixel jitter over the reference Gaussian film border."""
    axis=torch.arange(-border_size,resolution+border_size,dtype=torch.float32)
    py,px=torch.meshgrid(axis,axis,indexing="ij")
    px=px.reshape(-1,1).expand(-1,spp); py=py.reshape(-1,1).expand(-1,spp)
    generator=torch.Generator().manual_seed(seed)
    jitter=torch.rand((len(px),spp,2),generator=generator)
    x=(px+jitter[...,0]).reshape(-1); y=(py+jitter[...,1]).reshape(-1)
    origins,directions=camera_rays_from_positions(emitter,camera,resolution,x,y)
    return PrimarySamples(origins,directions,x.to(emitter.device),y.to(emitter.device),resolution,spp)


def gaussian_splat(samples:PrimarySamples,radiance:torch.Tensor,radius:int=2) -> torch.Tensor:
    """Mitsuba-style separable Gaussian reconstruction with per-pixel normalization."""
    res=samples.resolution; device=radiance.device
    accum=torch.zeros((res*res,3),device=device); weights=torch.zeros(res*res,device=device)
    base_x=torch.floor(samples.x-.5).long(); base_y=torch.floor(samples.y-.5).long()
    cutoff=torch.exp(torch.tensor(-2.0*radius*radius,device=device))
    for oy in range(-radius+1,radius+1):
        ty=base_y+oy; dy=(ty.float()+.5)-samples.y
        wy=(torch.exp(-2*dy.square())-cutoff).clamp_min(0)
        for ox in range(-radius+1,radius+1):
            tx=base_x+ox; dx=(tx.float()+.5)-samples.x
            wx=(torch.exp(-2*dx.square())-cutoff).clamp_min(0); w=wx*wy
            valid=(tx>=0)&(tx<res)&(ty>=0)&(ty<res)&(w>0)
            index=(ty[valid]*res+tx[valid]); accum.index_add_(0,index,radiance[valid]*w[valid,None])
            weights.index_add_(0,index,w[valid])
    return (accum/weights.clamp_min(1e-12)[:,None]).reshape(res,res,3)


@torch.inference_mode()
def trace_primary(emitter:Any,geometry:Any,bsdf:Any,vmf:Any,origins:torch.Tensor,
                  directions:torch.Tensor,uv_override:torch.Tensor,spp:int,mode:SamplingMode,
                  camera_idx:int=0,seed:int=1,epsilon:float=1e-4,chunk_size:int=16384) -> DemoTrace:
    query=EnvironmentNerfQuery(emitter); hit=geometry.intersect(origins,directions)
    indices=torch.nonzero(hit.hit,as_tuple=False).squeeze(-1)
    rgb=query.query_environment_radiance(origins,directions,camera_idx,chunk_size=chunk_size)
    if len(indices)==0:
        return DemoTrace(rgb,hit.hit,torch.empty((0,3),device=emitter.device),0.0,{"hit_pixels":0})
    p=hit.point[indices]; n=hit.normal[indices]; wo=torch.nn.functional.normalize(-directions[indices],dim=-1)
    uv=uv_override[indices]; count=len(indices)
    generator=torch.Generator(device=emitter.device).manual_seed(seed)
    u=torch.rand((count,spp,4),generator=generator,device=emitter.device)
    nn=n[:,None,:].expand(-1,spp,3); ww=wo[:,None,:].expand(-1,spp,3); uu=uv[:,None,:].expand(-1,spp,2)
    flat_n=nn.reshape(-1,3); flat_wo=ww.reshape(-1,3); flat_uv=uu.reshape(-1,2)
    points=p[:,None,:].expand(-1,spp,3).reshape(-1,3)
    sampled_pdf_error=0.0
    if mode=="cosine":
        sampled=sample_cosine_hemisphere(nn,u[...,:2]); wi=sampled.wi.reshape(-1,3)
        proposal_pdf=sampled.pdf.reshape(-1); delta=torch.zeros_like(proposal_pdf,dtype=torch.bool)
    elif mode=="bsdf":
        wi,proposal_pdf,delta=bsdf.sample(flat_n,flat_wo,flat_uv,u[...,1].reshape(-1),u[...,2:4].reshape(-1,2))
    elif mode=="vmf":
        sample=vmf.sample(points,u[...,1:3].reshape(-1,2)); wi=sample.direction
        proposal_pdf=sample.pdf; delta=torch.zeros_like(proposal_pdf,dtype=torch.bool)
    elif mode=="mis":
        emitter_sample=vmf.sample(points,u[...,1:3].reshape(-1,2))
        bsdf_wi,bsdf_sample_pdf,bsdf_delta=bsdf.sample(flat_n,flat_wo,flat_uv,
                                                       u[...,1].reshape(-1),u[...,2:4].reshape(-1,2))
        choose_emitter=(u[...,0].reshape(-1)<.5)
        wi=torch.where(choose_emitter[:,None],emitter_sample.direction,bsdf_wi)
        proposal_pdf=torch.where(choose_emitter,emitter_sample.pdf,bsdf_sample_pdf)
        delta=torch.where(choose_emitter,torch.zeros_like(bsdf_delta),bsdf_delta)
    else: raise ValueError(mode)

    secondary_o=(points+epsilon*flat_n).contiguous()
    occluded=geometry.occluded(secondary_o,wi,epsilon); valid=(~occluded)&(proposal_pdf>0)
    Li=torch.zeros((len(wi),3),device=emitter.device)
    if valid.any(): Li[valid]=query.query_environment_radiance(secondary_o[valid],wi[valid],camera_idx,chunk_size=chunk_size)
    ev=bsdf.evaluate(flat_n,wi,flat_wo,flat_uv)
    if mode=="mis":
        emitter_pdf=vmf.pdf(points,wi); mixture_pdf=.5*(emitter_pdf+ev.pdf)
        mixture_pdf=torch.where(delta,.5*ev.pdf,mixture_pdf); denominator=mixture_pdf
        sampled_pdf_error=float((proposal_pdf[valid]-torch.where(choose_emitter,emitter_pdf,ev.pdf)[valid]).abs().max()) if valid.any() else 0.0
    elif mode=="vmf": denominator=proposal_pdf
    elif mode=="bsdf":
        denominator=proposal_pdf
        sampled_pdf_error=float((proposal_pdf[valid]-ev.pdf[valid]).abs().max()) if valid.any() else 0.0
    else: denominator=proposal_pdf
    contribution=Li*ev.weighted/denominator.clamp_min(1e-12)[:,None]
    contribution=torch.where(valid[:,None],contribution,torch.zeros_like(contribution))
    samples=contribution.reshape(count,spp,3); surface=samples.mean(1); rgb[indices]=surface
    luminance=(samples*torch.tensor([.299,.587,.114],device=emitter.device)).sum(-1)
    variance=float(luminance.var(1,unbiased=False).mean()/spp)
    diagnostics={"mode":mode,"spp":spp,"primary_rays":len(origins),"hit_rays":count,
                 "secondary_rays":len(wi),"visible_secondary":int(valid.sum()),
                 "sample_mean_variance_of_estimator":variance,"sampled_pdf_max_error":sampled_pdf_error,
                 "surface_max":float(surface.max()),"all_finite":bool(torch.isfinite(rgb).all())}
    return DemoTrace(rgb,hit.hit,surface,variance,diagnostics)


@torch.inference_mode()
def render_center(emitter:Any,geometry:Any,bsdf:Any,vmf:Any,rays:CameraRays,
                  uv_override:torch.Tensor,spp:int,mode:SamplingMode,camera_idx:int=0,seed:int=1) -> DemoTrace:
    trace=trace_primary(emitter,geometry,bsdf,vmf,rays.origins_external.reshape(-1,3),
                        rays.directions_external.reshape(-1,3),uv_override,spp,mode,camera_idx,seed)
    return DemoTrace(trace.rgb.reshape(rays.height,rays.width,3),trace.hit.reshape(rays.height,rays.width),
                     trace.surface,trace.sample_variance,trace.diagnostics)
