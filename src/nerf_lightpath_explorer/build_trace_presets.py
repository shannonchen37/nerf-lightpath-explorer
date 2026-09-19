"""Generate exact pixel traces, including Gaussian reconstruction samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from nerf_lightpath_explorer.pbir.demo_renderer import camera_rays_from_positions,jittered_primary_rays
from nerf_lightpath_explorer.pbir.aabb_control import set_aabb_disabled
from nerf_lightpath_explorer.pbir.environment_query import EnvironmentNerfQuery
from nerf_lightpath_explorer.pbir.mitsuba_principled import MitsubaPrincipledTextureBSDF
from nerf_lightpath_explorer.pbir.spatial_vmf_guiding import SpatialVmfMixture
from nerf_lightpath_explorer.pbir.teapot_geometry import ExplicitTeapotGeometry,_bilinear
from nerf_lightpath_explorer.validate_image_reconstruction import load_exr
from nerf_lightpath_explorer.runtime_paths import ASSET_ROOT as ASSETS, CHECKPOINT_PATH, MODEL_ROOT as MODEL_RUNTIME, RESULTS_ROOT, load_environment as load_emitter

DEMO_RESULTS=RESULTS_ROOT/"demo"; OUT=RESULTS_ROOT/"pixel_traces"; TRACES=OUT/"traces"
RES=256; PRIMARY_SPP=16; SECONDARY_SPP=16; PRIMARY_SEED=12001; PATH_SEED=12002
LUMA=torch.tensor([.299,.587,.114])


def barycentric(geometry,pid:int,point:torch.Tensor) -> list[float]:
    face=geometry.faces[pid]; a,b,c=[torch.tensor(geometry.vertices[i],device=point.device) for i in face]
    v0=b-a; v1=c-a; v2=point-a; d00=v0@v0; d01=v0@v1; d11=v1@v1; d20=v2@v0; d21=v2@v1
    den=d00*d11-d01*d01; v=(d11*d20-d01*d21)/den; w=(d00*d21-d01*d20)/den
    return [float(1-v-w),float(v),float(w)]


def dfg(n,wi,wo,roughness):
    no_l=(n*wi).sum(-1).clamp_min(1e-8); no_v=(n*wo).sum(-1).clamp_min(1e-8)
    h=torch.nn.functional.normalize(wi+wo,dim=-1); no_h=(n*h).sum(-1).clamp(0,1); vo_h=(wo*h).sum(-1).clamp(0,1)
    alpha=roughness.square().clamp_min(.001); a2=alpha.square()
    D=a2/(torch.pi*(no_h.square()*(a2-1)+1).square()).clamp_min(1e-12)
    def g1(x): return 2*x/(x+torch.sqrt(a2+(1-a2)*x.square())).clamp_min(1e-12)
    G=g1(no_l)*g1(no_v); F=.08+.92*(1-vo_h).pow(5)
    return no_l,no_v,D,F,G


@torch.inference_mode()
def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--pixels",nargs="*",default=[]); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); TRACES.mkdir(exist_ok=True)
    emitter=load_emitter(); device=emitter.device; query=EnvironmentNerfQuery(emitter)
    previous=Path.cwd(); os.chdir(MODEL_RUNTIME)
    try: test=emitter.config.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split="test")
    finally: os.chdir(previous)
    camera=test.cameras.flatten()[0]; geometry=ExplicitTeapotGeometry(ASSETS/"mesh.obj",ASSETS/"reflectance.png",ASSETS/"roughness.png")
    bsdf=MitsubaPrincipledTextureBSDF(ASSETS/"reflectance.png",ASSETS/"roughness.png",specular=1.0)
    vmf=SpatialVmfMixture.from_checkpoint(CHECKPOINT_PATH,device)
    primary=jittered_primary_rays(emitter,camera,RES,PRIMARY_SPP,PRIMARY_SEED,2)
    oracle=np.load(DEMO_RESULTS/"jittered_uv_output.npz"); uv_all=torch.from_numpy(oracle["uv"]).to(device).float()
    hit=geometry.intersect(primary.origins,primary.directions); hit_ids=torch.nonzero(hit.hit).squeeze(-1)
    rank=torch.full((len(hit.hit),),-1,dtype=torch.long,device=device); rank[hit_ids]=torch.arange(len(hit_ids),device=device)
    g=torch.Generator(device=device).manual_seed(PATH_SEED); uniforms=torch.rand((len(hit_ids),SECONDARY_SPP,4),generator=g,device=device)
    recorded=load_exr(DEMO_RESULTS/"ours_final.exr",device); expected_mask=torch.load(DEMO_RESULTS/"expected_spp_256_mask.pt")[...,0]>.5

    if args.pixels:
        pixels=[tuple(map(int,p.split(","))) for p in args.pixels]
    else:
        mask=expected_mask.cpu().numpy().astype(np.uint8); eroded=torch.from_numpy(__import__("cv2").erode(mask,np.ones((7,7),np.uint8))>0)
        ref=load_exr(DEMO_RESULTS/"expected_reference.exr",device); lum=(ref*LUMA.to(device)).sum(-1).cpu()
        hy,hx=np.unravel_index(int(torch.where(eroded,lum,torch.tensor(-1.)).argmax()),(RES,RES))
        coords=torch.nonzero(eroded); median=torch.median(lum[eroded]); di=int((lum[eroded]-median).abs().argmin()); dy,dx=map(int,coords[di])
        boundary=torch.from_numpy((mask-__import__("cv2").erode(mask,np.ones((3,3),np.uint8)))>0); bc=torch.nonzero(boundary); target=torch.tensor([RES//2,0]); si=int(((bc-target).float().square().sum(-1)).argmin()); sy,sx=map(int,bc[si])
        pixels=[(16,16),(dx,dy),(int(hx),int(hy)),(sx,sy)]
        labels=["background","diffuse","highlight","silhouette"]
        (OUT/"acceptance_pixels.json").write_text(json.dumps({labels[i]:list(pixels[i]) for i in range(4)},indent=2))

    cutoff=torch.exp(torch.tensor(-8.0,device=device)); outputs=[]
    for x,y in pixels:
        dx=(x+.5)-primary.x; dy=(y+.5)-primary.y
        weights=(torch.exp(-2*dx.square())-cutoff).clamp_min(0)*(torch.exp(-2*dy.square())-cutoff).clamp_min(0)
        selected=torch.nonzero(weights>0).squeeze(-1); selected_weights=weights[selected]
        radiance=torch.zeros((len(selected),3),device=device); primary_records=[]; surface_records=[]; all_sample_records=[]
        misses=~hit.hit[selected]
        if misses.any(): radiance[misses]=query.query_environment_radiance(primary.origins[selected[misses]],primary.directions[selected[misses]],0)
        selected_hit=selected[~misses]; representative=None
        if len(selected_hit):
            representative=int(selected_hit[selected_weights[~misses].argmax()]); ranks=rank[selected_hit]
            p=hit.point[selected_hit]; n=hit.normal[selected_hit]; wo=torch.nn.functional.normalize(-primary.directions[selected_hit],dim=-1); uv=uv_all[selected_hit]
            u=uniforms[ranks]; nn=n[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3); ww=wo[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3)
            uuv=uv[:,None,:].expand(-1,SECONDARY_SPP,2).reshape(-1,2); points=p[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3)
            es=vmf.sample(points,u[...,1:3].reshape(-1,2)); bw,bspdf,bdelta=bsdf.sample(nn,ww,uuv,u[...,1].reshape(-1),u[...,2:4].reshape(-1,2))
            choose=u[...,0].reshape(-1)<.5; wi=torch.where(choose[:,None],es.direction,bw); source=torch.where(choose,torch.zeros_like(choose,dtype=torch.long),torch.ones_like(choose,dtype=torch.long))
            base=torch.from_numpy(_bilinear(geometry.reflectance,uv.cpu().numpy())).to(device); rough=torch.from_numpy(_bilinear(geometry.roughness_texture,uv.cpu().numpy())).to(device)
            base_flat=base[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3); rough_flat=rough[:,None].expand(-1,SECONDARY_SPP).reshape(-1)
            secondary_o=points+1e-4*nn; occluded=geometry.occluded(secondary_o,wi,1e-4); ev=bsdf.evaluate(nn,wi,ww,uuv); lobes=bsdf.evaluate_lobes(nn,wi,ww,uuv,base_flat,rough_flat)
            epdf=vmf.pdf(points,wi); bpdf=ev.pdf; denom=.5*(epdf+bpdf); valid=(~occluded)&(denom>0)
            Li=torch.zeros((len(wi),3),device=device); Li[valid]=query.query_environment_radiance(secondary_o[valid],wi[valid],0)
            contribution=torch.where(valid[:,None],Li*ev.weighted/denom.clamp_min(1e-12)[:,None],torch.zeros_like(Li))
            diffuse_contribution=torch.where(valid[:,None],Li*lobes["diffuse_weighted"]/denom.clamp_min(1e-12)[:,None],torch.zeros_like(Li))
            specular_contribution=torch.where(valid[:,None],Li*lobes["specular_weighted"]/denom.clamp_min(1e-12)[:,None],torch.zeros_like(Li))
            surface=contribution.reshape(len(selected_hit),SECONDARY_SPP,3).mean(1); radiance[~misses]=surface
            no_l,no_v,D,F,G=dfg(nn,wi,ww,rough[:,None].expand(-1,SECONDARY_SPP).reshape(-1))
            proposal=torch.where(choose,epdf,bspdf); mis_weight=proposal/(epdf+bpdf).clamp_min(1e-12)
            for j,ray_id in enumerate(selected_hit.tolist()):
                surface_records.append({"ray_id":ray_id,"position":p[j].tolist(),"shading_normal":n[j].tolist(),
                    "wo":wo[j].tolist(),"uv":uv[j].tolist(),"base_color":base[j].tolist(),"roughness":float(rough[j])})
            for j,ray_id in enumerate(selected_hit.tolist()):
                for s in range(SECONDARY_SPP):
                    k=j*SECONDARY_SPP+s
                    all_sample_records.append({"primary_ray_id":ray_id,"sample_id":s,"source":"emitter_vmf" if bool(choose[k]) else "principled_bsdf",
                        "wi":wi[k].tolist(),"visible":bool(valid[k]),"occluded":bool(occluded[k]),"Li":Li[k].tolist(),
                        "bsdf_f_cos":ev.weighted[k].tolist(),"cos_theta":float(no_l[k]),"NoV":float(no_v[k]),
                        "D":float(D[k]),"F":[float(F[k])]*3,"G":float(G[k]),"bsdf_pdf":float(bpdf[k]),"emitter_pdf":float(epdf[k]),
                        "diffuse_f_cos":lobes["diffuse_weighted"][k].tolist(),"specular_f_cos":lobes["specular_weighted"][k].tolist(),
                        "proposal_pdf":float(proposal[k]),"selection_probability":.5,"mis_weight":float(mis_weight[k]),
                        "component":int(es.component[k]),"vmf_mu":es.mu[k].tolist(),"vmf_kappa":float(es.kappa[k]),
                        "diffuse_contribution":diffuse_contribution[k].tolist(),"specular_contribution":specular_contribution[k].tolist(),
                        "contribution":contribution[k].tolist()})
        reconstructed=(radiance*selected_weights[:,None]).sum(0)/selected_weights.sum()
        for j,ray_id in enumerate(selected.tolist()):
            primary_records.append({"ray_id":ray_id,"sample_position":[float(primary.x[ray_id]),float(primary.y[ray_id])],
                                    "filter_weight":float(selected_weights[j]),"origin":primary.origins[ray_id].tolist(),
                                    "direction":primary.directions[ray_id].tolist(),"hit":bool(hit.hit[ray_id]),"radiance":radiance[j].tolist()})
        nominal_o,nominal_d=camera_rays_from_positions(emitter,camera,RES,torch.tensor([x+.5]),torch.tensor([y+.5]))
        rep={"hit":False}
        if representative is not None:
            pid=int(hit.primitive_id[representative]); face=geometry.faces[pid]; verts=geometry.vertices[face]
            gn=np.cross(verts[1]-verts[0],verts[2]-verts[0]); gn=(gn/np.linalg.norm(gn)).tolist(); ruv=uv_all[representative]
            base=_bilinear(geometry.reflectance,ruv[None].cpu().numpy())[0].tolist(); rough=float(_bilinear(geometry.roughness_texture,ruv[None].cpu().numpy())[0])
            rep={"hit":True,"t_hit":float(hit.t[representative]),"world_position":hit.point[representative].tolist(),
                 "emitter_local_position":hit.point[representative].tolist(),"triangle_id":pid,"barycentric":barycentric(geometry,pid,hit.point[representative]),
                 "uv":ruv.tolist(),"geometric_normal":gn,"shading_normal":hit.normal[representative].tolist(),
                 "material":{"base_color":base,"roughness":rough,"specular":1.0,"metallic":0.0,"F0":.08,"ior":1.7887885053796064,"type":"Mitsuba principled"}}
        # Trace the single most important visible secondary ray through NeRF.
        best=None
        if all_sample_records:
            best=max((r for r in all_sample_records if r["visible"]),key=lambda r:sum(r["contribution"]),default=None)
            if best:
                rid=best["primary_ray_id"]
                with set_aabb_disabled(emitter.model,True):
                    nerf=emitter.debug_query_environment_ray(hit.point[rid]+1e-4*hit.normal[rid],torch.tensor(best["wi"],device=device),0)
            else:
                with set_aabb_disabled(emitter.model,True): nerf=emitter.debug_query_environment_ray(nominal_o[0],nominal_d[0],0)
        else:
            with set_aabb_disabled(emitter.model,True): nerf=emitter.debug_query_environment_ray(nominal_o[0],nominal_d[0],0)
        trace={"status":"PASS","pixel":[x,y],"mode":"EXACT_TRACE","final_renderer":{"primary_seed":PRIMARY_SEED,"path_seed":PATH_SEED,
               "primary_spp":PRIMARY_SPP,"secondary_spp":SECONDARY_SPP,"filter":"Gaussian radius2 stddev0.5"},
               "primary_ray":{"origin":nominal_o[0].tolist(),"direction":nominal_d[0].tolist(),**rep},
               "material":rep.get("material"),"primary_reconstruction_samples":primary_records,"primary_surface_samples":surface_records,
               "secondary_samples":all_sample_records,
               "nerf_selected_sample":best,"nerf_selected_ray":nerf,"recorded_final_rgb":recorded[y,x].tolist(),"reconstructed_rgb":reconstructed.tolist(),
               "reconstruction_max_abs":float((recorded[y,x]-reconstructed).abs().max()),"vMF":{"component_count":64}}
        path=TRACES/f"pixel_{x}_{y}.json"; path.write_text(json.dumps(trace,indent=2)); outputs.append(str(path))
    print(json.dumps({"status":"PASS","outputs":outputs},indent=2))


if __name__=="__main__": main()
