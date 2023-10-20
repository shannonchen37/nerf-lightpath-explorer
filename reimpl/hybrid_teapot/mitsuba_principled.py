"""Official-compatible Mitsuba principled BSDF evaluation for explicit surfaces.

Mitsuba's ``BSDF.eval`` returns ``f_r * abs(cos(theta_i))``.  This wrapper
keeps H4's torch/Open3D geometry and NeRF estimator, while delegating the exact
Disney-principled lobe semantics to the same Mitsuba plugin used by the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _mitsuba():
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb" if torch.cuda.is_available() else "llvm_ad_rgb")
    return mi


@dataclass(frozen=True)
class PrincipledEvaluation:
    weighted: torch.Tensor
    pdf: torch.Tensor
    brdf: torch.Tensor
    no_l: torch.Tensor


def _local_frame(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sign = torch.where(n[..., 2] >= 0, torch.ones_like(n[..., 2]), -torch.ones_like(n[..., 2]))
    a = -1.0 / (sign + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    tangent = torch.stack((1.0 + sign * n[..., 0] * n[..., 0] * a, sign * b, -sign * n[..., 0]), -1)
    bitangent = torch.stack((b, sign + n[..., 1] * n[..., 1] * a, -n[..., 1]), -1)
    return tangent, bitangent


def world_to_local(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    t, b = _local_frame(n)
    return torch.stack(((v * t).sum(-1), (v * b).sum(-1), (v * n).sum(-1)), -1)


def local_to_world(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    t,b=_local_frame(n)
    return v[...,0:1]*t+v[...,1:2]*b+v[...,2:3]*n


class MitsubaPrincipledTextureBSDF:
    """Exact official texture-backed principled BSDF, evaluated in batches."""

    def __init__(self, reflectance_path: Path, roughness_path: Path, specular: float = 1.0) -> None:
        mi = _mitsuba()
        self.mi = mi
        self.bsdf = mi.load_dict({
            "type": "principled",
            "base_color": {"type": "bitmap", "filename": str(reflectance_path), "raw": True},
            "roughness": {"type": "bitmap", "filename": str(roughness_path), "raw": True},
            "specular": float(specular),
        })

    def evaluate(self, normal: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, uv: torch.Tensor) -> PrincipledEvaluation:
        """Evaluate with project notation: wi=surface->light, wo=surface->camera."""
        import drjit as dr
        mi = self.mi
        shape = normal.shape
        if normal.shape != wi.shape or normal.shape != wo.shape or normal.shape[-1] != 3:
            raise ValueError("normal/wi/wo must share [...,3] shape")
        if uv.shape != shape[:-1] + (2,):
            raise ValueError("uv must share the leading shape and have two channels")
        n = normal.reshape(-1, 3)
        light_local = world_to_local(wi.reshape(-1, 3), n).detach().cpu().numpy()
        camera_local = world_to_local(wo.reshape(-1, 3), n).detach().cpu().numpy()
        uv_np = uv.reshape(-1, 2).detach().cpu().numpy()
        count = len(light_local)
        si = dr.zeros(mi.SurfaceInteraction3f, count)
        si.wi = mi.Vector3f(camera_local)
        si.uv = mi.Point2f(uv_np)
        active = mi.Mask((light_local[:, 2] > 0) & (camera_local[:, 2] > 0))
        light = mi.Vector3f(light_local)
        ctx = mi.BSDFContext()
        weighted = self.bsdf.eval(ctx, si, light, active)
        pdf = self.bsdf.pdf(ctx, si, light, active)
        weighted_np = np.asarray(weighted).astype(np.float32, copy=True)
        pdf_np = np.asarray(pdf).astype(np.float32, copy=True)
        weighted_t = torch.from_numpy(weighted_np).to(normal.device).reshape(shape)
        pdf_t = torch.from_numpy(pdf_np).to(normal.device).reshape(shape[:-1])
        no_l = torch.clamp_min((normal * wi).sum(-1), 0.0)
        brdf = weighted_t / no_l.clamp_min(1e-8)[..., None]
        brdf = torch.where(no_l[..., None] > 0, brdf, torch.zeros_like(brdf))
        return PrincipledEvaluation(weighted_t, pdf_t, brdf, no_l)

    def evaluate_lobes(self, normal: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor,
                       uv: torch.Tensor, base_color: torch.Tensor,
                       roughness: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exact lobe attribution for the enabled Mitsuba 3.4.1 teapot setup.

        The frozen setup has metallic/spec-trans/clearcoat/sheen/flatness all
        disabled. We reproduce the plugin's diffuse+retro term verbatim and
        define specular as the exact plugin total minus that term. This avoids
        pretending that ``BSDFContext.component`` filters ``eval()`` (it does
        not for this plugin) and guarantees lobe closure.
        """
        total = self.evaluate(normal, wi, wo, uv).weighted
        no_l = (normal * wi).sum(-1).clamp(0.0, 1.0)
        no_v = (normal * wo).sum(-1).clamp(0.0, 1.0)
        wh = torch.nn.functional.normalize(wi + wo, dim=-1)
        cos_d = (wh * wi).sum(-1)
        fi = (1.0 - no_v).pow(5)
        fo = (1.0 - no_l).pow(5)
        f_diff = (1.0 - 0.5 * fi) * (1.0 - 0.5 * fo)
        rr = 2.0 * roughness * cos_d.square()
        f_retro = rr * (fo + fi + fo * fi * (rr - 1.0))
        diffuse = no_l[..., None] * base_color * ((f_diff + f_retro) / torch.pi)[..., None]
        active = (no_l > 0) & (no_v > 0)
        diffuse = torch.where(active[..., None], diffuse, torch.zeros_like(diffuse))
        specular = total - diffuse
        return {"diffuse_weighted": diffuse, "specular_weighted": specular,
                "total_weighted": total,
                "provenance": "Mitsuba 3.4.1 principled.cpp diffuse+retro; exact total-minus-diffuse"}

    def sample(self,normal:torch.Tensor,wo:torch.Tensor,uv:torch.Tensor,
               sample1:torch.Tensor,sample2:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """Sample the installed official principled plugin; return world wi/pdf/delta."""
        import drjit as dr
        mi=self.mi; shape=normal.shape
        if wo.shape!=shape or uv.shape!=shape[:-1]+(2,): raise ValueError("incompatible BSDF sample shapes")
        n=normal.reshape(-1,3); camera_local=world_to_local(wo.reshape(-1,3),n).detach().cpu().numpy()
        uv_np=uv.reshape(-1,2).detach().cpu().numpy(); count=len(n)
        si=dr.zeros(mi.SurfaceInteraction3f,count); si.wi=mi.Vector3f(camera_local); si.uv=mi.Point2f(uv_np)
        active=mi.Mask(camera_local[:,2]>0); ctx=mi.BSDFContext()
        bs,_=self.bsdf.sample(ctx,si,mi.Float(sample1.reshape(-1).detach().cpu().numpy()),
                              mi.Point2f(sample2.reshape(-1,2).detach().cpu().numpy()),active)
        local=torch.from_numpy(np.asarray(bs.wo).astype(np.float32,copy=True)).to(normal.device)
        pdf=torch.from_numpy(np.asarray(bs.pdf).astype(np.float32,copy=True)).to(normal.device)
        sampled_type=torch.from_numpy(np.asarray(bs.sampled_type).astype(np.int64,copy=True)).to(normal.device)
        wi=torch.nn.functional.normalize(local_to_world(local,n),dim=-1)
        delta=(sampled_type & int(mi.BSDFFlags.Delta))!=0
        return wi.reshape(shape),pdf.reshape(shape[:-1]),delta.reshape(shape[:-1])


class MitsubaPrincipledConstantBSDF:
    """Official principled mode for H6 constant opaque material presets."""

    def __init__(self, base_color: tuple[float, float, float], roughness: float,
                 metallic: float = 0.0, specular: float = 1.0) -> None:
        mi = _mitsuba()
        self.mi = mi
        self.base_color = base_color
        self.roughness = roughness
        self.metallic = metallic
        self.specular = specular
        self.bsdf = mi.load_dict({"type": "principled", "base_color": {"type": "rgb", "value": list(base_color)},
                                  "roughness": roughness, "metallic": metallic, "specular": specular})

    evaluate = MitsubaPrincipledTextureBSDF.evaluate
    evaluate_lobes = MitsubaPrincipledTextureBSDF.evaluate_lobes
    sample = MitsubaPrincipledTextureBSDF.sample


class MitsubaPrincipledInterventionBSDF(MitsubaPrincipledTextureBSDF):
    """Official principled plugin with optional causal material overrides."""

    def __init__(self, reflectance_path: Path, roughness_path: Path, *,
                 base_color: tuple[float, float, float] | None = None,
                 roughness: float | None = None, specular: float = 1.0,
                 metallic: float = 0.0) -> None:
        mi = _mitsuba()
        self.mi = mi
        base = ({"type": "bitmap", "filename": str(reflectance_path), "raw": True}
                if base_color is None else {"type": "rgb", "value": list(base_color)})
        rough = ({"type": "bitmap", "filename": str(roughness_path), "raw": True}
                 if roughness is None else float(roughness))
        self.bsdf = mi.load_dict({"type": "principled", "base_color": base,
                                  "roughness": rough, "specular": float(specular),
                                  "metallic": float(metallic)})
