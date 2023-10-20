"""Hybrid camera composition: finite NeRF segment plus explicit teapot surface."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from time import perf_counter
from typing import Any

import torch

from reimpl.hybrid_teapot.environment_query import EnvironmentNerfQuery
from reimpl.hybrid_teapot.segment_query import SegmentResult, query_environment_segment
from reimpl.hybrid_teapot.teapot_geometry import ExplicitTeapotGeometry, TeapotHit
from reimpl.hybrid_teapot.textured_ggx import evaluate_textured_dielectric_ggx
from reimpl.renderers.ggx_sphere_renderer import deterministic_uniforms
from reimpl.renderers.mirror_sphere_renderer import CameraRays
from reimpl.sampling.cosine_hemisphere import sample_cosine_hemisphere


@dataclass(frozen=True)
class HybridTeapotRender:
    rgb: torch.Tensor
    full_nerf: torch.Tensor
    environment_only: torch.Tensor
    direct_overlay: torch.Tensor
    surface_rgb: torch.Tensor
    pre_segment_color: torch.Tensor
    pre_segment_transmittance: torch.Tensor
    hit: TeapotHit
    metrics: dict


@torch.inference_mode()
def render_hybrid_teapot(
    emitter: Any,
    geometry: ExplicitTeapotGeometry,
    camera_rays: CameraRays,
    spp: int = 16,
    camera_idx: int = 0,
    epsilon: float = 1.0e-4,
    seed: int = 42,
    chunk_size: int = 16384,
) -> HybridTeapotRender:
    start_total = perf_counter()
    h, w = camera_rays.height, camera_rays.width
    origins = camera_rays.origins_external.reshape(-1, 3)
    directions = camera_rays.directions_external.reshape(-1, 3)
    query = EnvironmentNerfQuery(emitter)
    full_nerf = query.query_full_radiance(origins, directions, camera_idx, chunk_size=chunk_size)
    environment = query.query_environment_radiance(origins, directions, camera_idx, chunk_size=chunk_size)
    hit = geometry.intersect(origins, directions)
    indices = torch.nonzero(hit.hit, as_tuple=False).squeeze(-1)
    points, normals = hit.point[indices], hit.normal[indices]
    wo_pixel = torch.nn.functional.normalize(-directions[indices], dim=-1)

    start_segment = perf_counter()
    segment: SegmentResult = query_environment_segment(
        emitter, origins[indices], directions[indices], hit.t[indices], camera_idx, chunk_size
    )
    torch.cuda.synchronize(emitter.device)
    segment_seconds = perf_counter() - start_segment

    normal = normals[:, None, :].expand(-1, spp, 3)
    wo = wo_pixel[:, None, :].expand(-1, spp, 3)
    uniforms = deterministic_uniforms(len(indices), spp, seed, emitter.device)
    samples = sample_cosine_hemisphere(normal, uniforms)
    secondary_origins = (points + epsilon * normals)[:, None, :].expand(-1, spp, 3).contiguous()
    flat_origins = secondary_origins.reshape(-1, 3)
    flat_directions = samples.wi.reshape(-1, 3)
    start_visibility = perf_counter()
    occluded = geometry.occluded(flat_origins, flat_directions, epsilon).reshape(len(indices), spp)
    visibility_seconds = perf_counter() - start_visibility
    valid = ~occluded
    Li = torch.zeros((len(indices), spp, 3), dtype=torch.float32, device=emitter.device)
    start_secondary = perf_counter()
    if valid.any():
        Li.reshape(-1, 3)[valid.reshape(-1)] = query.query_environment_radiance(
            flat_origins[valid.reshape(-1)], flat_directions[valid.reshape(-1)],
            camera_idx, chunk_size=chunk_size,
        )
    torch.cuda.synchronize(emitter.device)
    secondary_seconds = perf_counter() - start_secondary
    base_color = hit.albedo[indices, None, :].expand(-1, spp, 3)
    roughness = hit.roughness[indices, None].expand(-1, spp)
    evaluation = evaluate_textured_dielectric_ggx(normal, samples.wi, wo, base_color, roughness)
    contribution = Li * evaluation.brdf * pi
    contribution = torch.where(valid[..., None], contribution, torch.zeros_like(contribution))
    surface_rgb = contribution.mean(dim=1)

    hybrid = environment.clone()
    hybrid[indices] = segment.color + segment.transmittance * surface_rgb
    direct_overlay = environment.clone(); direct_overlay[indices] = surface_rgb
    rgb = hybrid.reshape(h, w, 3)
    metrics = {
        "resolution": [h, w], "spp": spp, "hit_pixels": len(indices),
        "camera_formula": "C_pre(0->t_s) + T_pre(0->t_s) * L_surface",
        "miss_semantics": "environment-only NeRF whole ray",
        "secondary_semantics": "environment-only NeRF with disable_aabb=True",
        "pre_segment_seconds": segment_seconds, "secondary_query_seconds": secondary_seconds,
        "explicit_visibility_seconds": visibility_seconds,
        "candidate_secondary_rays": len(indices) * spp,
        "queried_secondary_rays": int(valid.sum()), "self_occluded_secondary_rays": int(occluded.sum()),
        "pre_color_mean": float(segment.color.mean()),
        "pre_transmittance_min": float(segment.transmittance.min()),
        "pre_transmittance_mean": float(segment.transmittance.mean()),
        "pre_transmittance_max": float(segment.transmittance.max()),
        "surface_mean": float(surface_rgb.mean()), "surface_max": float(surface_rgb.max()),
        "hybrid_vs_direct_overlay_mae_on_object": float((hybrid[indices] - direct_overlay[indices]).abs().mean()),
        "full_vs_environment_mae": float((full_nerf - environment).abs().mean()),
        "all_finite": bool(torch.isfinite(rgb).all()),
        "total_seconds": perf_counter() - start_total,
    }
    return HybridTeapotRender(
        rgb, full_nerf.reshape(h,w,3), environment.reshape(h,w,3), direct_overlay.reshape(h,w,3),
        surface_rgb, segment.color, segment.transmittance, hit, metrics,
    )

