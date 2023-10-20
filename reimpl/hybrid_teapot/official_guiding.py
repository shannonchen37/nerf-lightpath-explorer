"""Torch reproduction of the official 64-component spatial-GMM/vMF emitter proposal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def _frame(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sign=torch.where(n[...,2]>=0,torch.ones_like(n[...,2]),-torch.ones_like(n[...,2]))
    a=-1.0/(sign+n[...,2]); b=n[...,0]*n[...,1]*a
    t=torch.stack((1+sign*n[...,0]*n[...,0]*a,sign*b,-sign*n[...,0]),-1)
    bt=torch.stack((b,sign+n[...,1]*n[...,1]*a,-n[...,1]),-1)
    return t,bt


@dataclass(frozen=True)
class VmfSamples:
    direction: torch.Tensor
    pdf: torch.Tensor
    component: torch.Tensor
    mu: torch.Tensor
    kappa: torch.Tensor


class OfficialVmfMixture:
    """Exact parameterization used by ``emitters/vMF.py`` in primal mode."""

    def __init__(self,position:torch.Tensor,std:torch.Tensor,weight:torch.Tensor) -> None:
        if position.shape!=(64,3) or std.shape!=(64,) or weight.shape!=(64,):
            raise ValueError("official teapot proposal must contain 64 isotropic components")
        self.position=position.float(); self.std=std.float(); self.weight=(weight/weight.sum()).float()
        self.inv_var=1.0/self.std.square(); self.cdf=torch.cumsum(self.weight,0); self.cdf[-1]=1.0

    @classmethod
    def from_checkpoint(cls,path:Path,device:torch.device) -> "OfficialVmfMixture":
        checkpoint=torch.load(path,map_location="cpu")
        state=checkpoint["pipeline"]
        return cls(state["vmf.position"].reshape(64,3).to(device),
                   state["vmf.std"].reshape(64).to(device),
                   state["vmf.weight"].reshape(64).to(device))

    def sample(self,points:torch.Tensor,uniforms:torch.Tensor) -> VmfSamples:
        if uniforms.shape!=(len(points),2): raise ValueError("uniforms must be [N,2]")
        index=torch.searchsorted(self.cdf,uniforms[:,0].contiguous(),right=False).clamp_max(63)
        center=self.position[index]; kappa=torch.linalg.vector_norm(center-points,dim=-1)*self.inv_var[index]
        mu=torch.nn.functional.normalize(center-points,dim=-1)
        cdf_before=torch.where(index>0,self.cdf[(index-1).clamp_min(0)],torch.zeros_like(uniforms[:,0]))
        # Mitsuba DiscreteDistribution.sample_reuse(): component selection and
        # the vMF polar sample share sample.x after interval remapping.
        u=((uniforms[:,0]-cdf_before)/self.weight[index].clamp_min(1e-12)).clamp(1e-7,1-1e-7)
        exp_term=torch.exp(-2*kappa)
        cos_theta=1+torch.log(u+(1-u)*exp_term)/kappa.clamp_min(1e-7)
        cos_theta=torch.where(kappa<1e-4,2*u-1,cos_theta).clamp(-1,1)
        sin_theta=torch.sqrt(torch.clamp_min(1-cos_theta.square(),0)); phi=2*torch.pi*uniforms[:,1]
        local=torch.stack((sin_theta*torch.cos(phi),sin_theta*torch.sin(phi),cos_theta),-1)
        t,b=_frame(mu); direction=torch.nn.functional.normalize(local[:,0:1]*t+local[:,1:2]*b+local[:,2:3]*mu,dim=-1)
        return VmfSamples(direction,self.pdf(points,direction),index,mu,kappa)

    def pdf(self,points:torch.Tensor,directions:torch.Tensor,chunk_size:int=32768) -> torch.Tensor:
        outputs=[]
        for start in range(0,len(points),chunk_size):
            p=points[start:start+chunk_size,None,:]; d=directions[start:start+chunk_size,None,:]
            lobe=self.position[None,:,:]-p; distance=torch.linalg.vector_norm(lobe,dim=-1)
            mu=lobe/distance.clamp_min(1e-8)[...,None]; kappa=distance*self.inv_var[None,:]
            cosine=(mu*d).sum(-1).clamp(-1,1)
            norm=kappa/(2*torch.pi*(-torch.expm1(-2*kappa)).clamp_min(1e-12))
            component_pdf=norm*torch.exp(kappa*(cosine-1))
            component_pdf=torch.where(kappa<1e-4,torch.full_like(component_pdf,1/(4*torch.pi)),component_pdf)
            outputs.append((component_pdf*self.weight[None,:]).sum(-1))
        return torch.cat(outputs)
