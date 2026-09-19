"""Vectorized single-bounce GGX sphere renderer driven by NeRF illumination."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from nerf_lightpath_explorer.geometry.sphere import SphereHit, intersect_sphere
from nerf_lightpath_explorer.materials.ggx import DENOMINATOR_EPSILON, GGXEvaluation, GGXMaterial, evaluate_ggx
from nerf_lightpath_explorer.renderers.mirror_sphere_renderer import CameraRays
from nerf_lightpath_explorer.sampling.ggx_sampling import GGXSamples, sample_ggx_half_vector


@dataclass(frozen=True)
class GGXRender:
    rgb: torch.Tensor
    hit: SphereHit
    hit_indices: torch.Tensor
    wo: torch.Tensor
    secondary_origins: torch.Tensor
    Li: torch.Tensor
    samples: GGXSamples
    evaluation: GGXEvaluation
    contribution: torch.Tensor
    timings_seconds: dict[str, float]
    valid_query_count: int


def deterministic_uniforms(hit_count: int, spp: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand((hit_count, spp, 2), generator=generator, dtype=torch.float32).to(device)


@torch.inference_mode()
def render_ggx_sphere(
    emitter: Any,
    camera_rays: CameraRays,
    background: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    material: GGXMaterial,
    spp: int,
    camera_idx: int = 0,
    epsilon: float = 1.0e-4,
    seed: int = 42,
    chunk_size: int = 16384,
    uniforms: torch.Tensor | None = None,
) -> GGXRender:
    if spp < 1:
        raise ValueError("spp must be positive")
    material.validate()
    start_total = perf_counter()
    h, w = camera_rays.height, camera_rays.width
    origins = camera_rays.origins_external.reshape(-1, 3)
    directions = camera_rays.directions_external.reshape(-1, 3)
    hit = intersect_sphere(origins, directions, center, radius)
    hit_indices = torch.nonzero(hit.hit, as_tuple=False).squeeze(-1)
    points, normals = hit.point[hit_indices], hit.normal[hit_indices]
    wo_pixel = torch.nn.functional.normalize(-directions[hit_indices], dim=-1)
    count = len(hit_indices)
    normal = normals[:, None, :].expand(count, spp, 3)
    wo = wo_pixel[:, None, :].expand(count, spp, 3)
    if uniforms is None:
        uniforms = deterministic_uniforms(count, spp, seed, emitter.device)
    elif uniforms.shape != (count, spp, 2):
        raise ValueError(f"uniforms must be [{count},{spp},2]")

    start_sampling = perf_counter()
    samples = sample_ggx_half_vector(normal, wo, material.roughness, uniforms)
    evaluation = evaluate_ggx(normal, samples.wi, wo, material, samples.half_vector)
    secondary_origins = (points + epsilon * normals)[:, None, :].expand(count, spp, 3).contiguous()
    torch.cuda.synchronize(emitter.device)
    sampling_seconds = perf_counter() - start_sampling

    Li = torch.zeros((count, spp, 3), dtype=torch.float32, device=emitter.device)
    valid_flat = samples.valid.reshape(-1)
    valid_count = int(valid_flat.sum())
    start_query = perf_counter()
    if valid_count:
        queried = emitter.query_radiance(
            secondary_origins.reshape(-1, 3)[valid_flat],
            samples.wi.reshape(-1, 3)[valid_flat],
            camera_idx,
            near=0.0,
            chunk_size=chunk_size,
        )
        Li.reshape(-1, 3)[valid_flat] = queried
    torch.cuda.synchronize(emitter.device)
    query_seconds = perf_counter() - start_query
    if not torch.isfinite(Li).all():
        raise RuntimeError("non-finite NeRF Li")

    factor = evaluation.NoL / samples.pdf.clamp_min(DENOMINATOR_EPSILON)
    contribution = Li * evaluation.brdf * factor[..., None]
    contribution = torch.where(samples.valid[..., None], contribution, torch.zeros_like(contribution))
    if not torch.isfinite(contribution).all():
        raise RuntimeError("non-finite Li * BRDF * NoL / pdf contribution")
    hit_rgb = contribution.mean(dim=1)
    rgb_flat = background.reshape(-1, 3).clone()
    rgb_flat[hit_indices] = hit_rgb
    rgb = rgb_flat.reshape(h, w, 3)
    if not torch.isfinite(rgb).all():
        raise RuntimeError("non-finite rendered RGB")
    total_seconds = perf_counter() - start_total
    return GGXRender(
        rgb, hit, hit_indices, wo, secondary_origins, Li, samples, evaluation,
        contribution, {"sampling_brdf": sampling_seconds, "nerf_query": query_seconds, "total": total_seconds},
        valid_count,
    )
