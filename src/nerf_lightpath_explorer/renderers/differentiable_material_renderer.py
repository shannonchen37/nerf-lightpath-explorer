"""Pure cache-based differentiable material renderer (zero NeRF calls)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nerf_lightpath_explorer.cache.material_light_cache import MaterialLightCache
from nerf_lightpath_explorer.materials.differentiable_material import DifferentiableGGXMaterial
from nerf_lightpath_explorer.materials.ggx import DENOMINATOR_EPSILON, GGXEvaluation, evaluate_ggx


@dataclass(frozen=True)
class DifferentiableRender:
    rgb: torch.Tensor
    object_rgb: torch.Tensor
    contribution: torch.Tensor
    evaluation: GGXEvaluation


class DifferentiableMaterialRenderer(torch.nn.Module):
    """Owns no NeRF/emitter and therefore cannot issue a radiance query."""

    def __init__(self, cache: MaterialLightCache) -> None:
        super().__init__()
        cache.assert_frozen()
        self.cache = cache
        self.nerf_query_calls = 0

    def forward(self, material: DifferentiableGGXMaterial) -> DifferentiableRender:
        count, spp = len(self.cache.hit_indices), self.cache.spp
        normals = self.cache.normals[:, None, :].expand(count, spp, 3)
        evaluation = evaluate_ggx(normals, self.cache.wi, self.cache.wo, material.as_ggx())
        # Keep the auditable estimator explicit even though NoL/pdf=pi for cosine sampling.
        factor = self.cache.NoL / self.cache.pdf.clamp_min(DENOMINATOR_EPSILON)
        contribution = self.cache.Li * evaluation.brdf * factor[..., None]
        if not torch.isfinite(contribution).all():
            raise RuntimeError("non-finite differentiable material contribution")
        object_rgb = contribution.mean(dim=1)
        object_canvas = torch.zeros_like(self.cache.background).reshape(-1, 3)
        object_canvas[self.cache.hit_indices] = object_rgb
        object_canvas = object_canvas.reshape(self.cache.height, self.cache.width, 3)
        rgb = torch.where(self.cache.hit_mask.reshape(self.cache.height, self.cache.width, 1), object_canvas, self.cache.background.detach())
        expects_grad = any(parameter.requires_grad for parameter in material.parameters()) and torch.is_grad_enabled()
        if expects_grad and not rgb.requires_grad:
            raise RuntimeError("rendered image lost the trainable material graph")
        if not torch.isfinite(rgb).all():
            raise RuntimeError("rendered image became non-finite")
        return DifferentiableRender(rgb, object_rgb, contribution, evaluation)
