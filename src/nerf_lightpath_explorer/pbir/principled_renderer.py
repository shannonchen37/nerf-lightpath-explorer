"""Hybrid teapot renderer using Mitsuba's exact principled BSDF semantics."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from nerf_lightpath_explorer.pbir.environment_query import EnvironmentNerfQuery
from nerf_lightpath_explorer.pbir.hybrid_renderer import HybridTeapotRender
from nerf_lightpath_explorer.pbir.segment_query import query_environment_segment
from nerf_lightpath_explorer.renderers.ggx_sphere_renderer import deterministic_uniforms
from nerf_lightpath_explorer.sampling.cosine_hemisphere import sample_cosine_hemisphere


@torch.inference_mode()
def render_principled_teapot(
    emitter: Any, geometry: Any, camera_rays: Any, bsdf: Any, spp: int,
    camera_idx: int = 0, epsilon: float = 1e-4, seed: int = 42,
    chunk_size: int = 16384, mitsuba_camera_vacuum: bool = True,
    uv_override: torch.Tensor | None = None,
    uniforms: torch.Tensor | None = None,
) -> HybridTeapotRender:
    start = perf_counter()
    h, w = camera_rays.height, camera_rays.width
    origins = camera_rays.origins_external.reshape(-1, 3)
    directions = camera_rays.directions_external.reshape(-1, 3)
    query = EnvironmentNerfQuery(emitter)
    full = query.query_full_radiance(origins, directions, camera_idx, chunk_size=chunk_size)
    environment = query.query_environment_radiance(origins, directions, camera_idx, chunk_size=chunk_size)
    hit = geometry.intersect(origins, directions)
    indices = torch.nonzero(hit.hit, as_tuple=False).squeeze(-1)
    points, normals = hit.point[indices], hit.normal[indices]
    wo_pixel = torch.nn.functional.normalize(-directions[indices], dim=-1)
    segment = query_environment_segment(emitter, origins[indices], directions[indices], hit.t[indices], camera_idx, chunk_size)

    normal = normals[:, None, :].expand(-1, spp, 3)
    wo = wo_pixel[:, None, :].expand(-1, spp, 3)
    hit_uv = hit.uv if uv_override is None else uv_override
    if hit_uv.shape != hit.uv.shape:
        raise ValueError("uv_override must have one UV pair per camera ray")
    uv = hit_uv[indices, None, :].expand(-1, spp, 2)
    if uniforms is None:
        uniforms = deterministic_uniforms(len(indices), spp, seed, emitter.device)
    elif uniforms.shape != (len(indices), spp, 2):
        raise ValueError("uniforms must have [hit_pixels,spp,2] shape")
    samples = sample_cosine_hemisphere(normal, uniforms)
    secondary_origins = (points + epsilon * normals)[:, None, :].expand(-1, spp, 3).contiguous()
    flat_o, flat_d = secondary_origins.reshape(-1, 3), samples.wi.reshape(-1, 3)
    occluded = geometry.occluded(flat_o, flat_d, epsilon).reshape(len(indices), spp)
    valid = ~occluded
    li = torch.zeros((len(indices), spp, 3), device=emitter.device)
    if valid.any():
        li.reshape(-1, 3)[valid.reshape(-1)] = query.query_environment_radiance(
            flat_o[valid.reshape(-1)], flat_d[valid.reshape(-1)], camera_idx, chunk_size=chunk_size)
    evaluation = bsdf.evaluate(normal, samples.wi, wo, uv)
    # Mitsuba eval is f*cos; cosine sampling pdf is cos/pi.
    contribution = li * evaluation.weighted / samples.pdf.clamp_min(1e-8)[..., None]
    contribution = torch.where(valid[..., None], contribution, torch.zeros_like(contribution))
    surface = contribution.mean(1)
    hybrid = environment.clone()
    if mitsuba_camera_vacuum:
        # Mitsuba represents NeRF as an environment emitter. Camera-to-mesh
        # transport is therefore vacuum: C_pre=0, T_pre=1.
        hybrid[indices] = surface
    else:
        hybrid[indices] = segment.color + segment.transmittance * surface
    overlay = environment.clone(); overlay[indices] = surface
    metrics = {
        "resolution": [h, w], "spp": spp, "hit_pixels": len(indices),
        "bsdf_mode": "mitsuba_principled", "specular": float(getattr(bsdf, "specular", 1.0)),
        "camera_segment_mode": "mitsuba_vacuum" if mitsuba_camera_vacuum else "finite_nerf_segment",
        "candidate_secondary_rays": len(indices) * spp,
        "queried_secondary_rays": int(valid.sum()), "self_occluded_secondary_rays": int(occluded.sum()),
        "surface_mean": float(surface.mean()), "surface_max": float(surface.max()),
        "pre_color_mean": float(segment.color.mean()), "pre_transmittance_mean": float(segment.transmittance.mean()),
        "all_finite": bool(torch.isfinite(hybrid).all()), "total_seconds": perf_counter() - start,
    }
    return HybridTeapotRender(hybrid.reshape(h,w,3), full.reshape(h,w,3), environment.reshape(h,w,3),
                              overlay.reshape(h,w,3), surface, segment.color, segment.transmittance, hit, metrics)
