"""Finite environment-NeRF camera segment color and transmittance queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from reimpl.hybrid_teapot.aabb_control import set_aabb_disabled


@dataclass(frozen=True)
class SegmentResult:
    color: torch.Tensor
    transmittance: torch.Tensor
    accumulation: torch.Tensor


@torch.inference_mode()
def query_environment_segment(
    emitter: Any,
    origins: torch.Tensor,
    directions: torch.Tensor,
    far_external: torch.Tensor,
    camera_idx: int,
    chunk_size: int = 8192,
) -> SegmentResult:
    """Integrate environment NeRF from ray origin to explicit surface distance.

    The public ray parameter is emitter-local. Position conversion multiplies
    distances by ``2*scene_scale``, whereas direction conversion is an axis
    permutation, so the internal far bound must carry that scale factor.
    Unlike the normal whole-ray renderer, the residual transmittance is not
    filled with a NeRF background color; it is returned for hybrid composition.
    """
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from nerfstudio.model_components.losses import scale_gradients_by_distance_squared

    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[-1] != 3:
        raise ValueError("origins/directions must have identical [N,3] shape")
    far_external = far_external.reshape(-1, 1).to(device=origins.device, dtype=torch.float32)
    if len(far_external) != len(origins) or torch.any(far_external <= 0) or not torch.isfinite(far_external).all():
        raise ValueError("far_external must be finite positive [N]")
    model = emitter.model
    colors, transmittances = [], []
    with set_aabb_disabled(model, True):
        for start in range(0, len(origins), chunk_size):
            end = min(start + chunk_size, len(origins))
            ray_bundle = emitter.make_ray_bundle(origins[start:end], directions[start:end], camera_idx)
            ray_bundle.nears = torch.zeros_like(far_external[start:end])
            ray_bundle.fars = far_external[start:end] * (2.0 * emitter.scene_scale)
            ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
            if model.rotater is not None:
                ray_samples.camera_indices = model.rotater.map_rotation_ids(ray_samples.camera_indices)
            field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
            if model.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            density = field_outputs[FieldHeadNames.DENSITY]
            weights = ray_samples.get_weights(density)
            color = torch.sum(weights * field_outputs[FieldHeadNames.RGB], dim=-2)
            optical_depth = torch.sum(ray_samples.deltas * density, dim=-2)
            transmittance = torch.exp(-optical_depth)
            colors.append(color.float())
            transmittances.append(transmittance.float())
    color = torch.cat(colors, dim=0)
    transmittance = torch.cat(transmittances, dim=0)
    accumulation = 1.0 - transmittance
    if not torch.isfinite(color).all() or not torch.isfinite(transmittance).all():
        raise RuntimeError("non-finite finite-segment query")
    return SegmentResult(color, transmittance, accumulation)

