"""A minimal explicit perfect-mirror sphere renderer.

No Mitsuba integrator, Monte Carlo sampler, visibility mesh, BSDF library, or
autograd optimization is used. Camera and geometry live directly in the public
emitter-local unit-cube frame of :class:`StandaloneNerfEmitter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from reimpl.geometry.sphere import SphereHit, intersect_sphere


@dataclass(frozen=True)
class CameraRays:
    origins_ns: torch.Tensor
    directions_ns: torch.Tensor
    origins_external: torch.Tensor
    directions_external: torch.Tensor
    height: int
    width: int


@dataclass(frozen=True)
class MirrorRender:
    rgb: torch.Tensor
    background: torch.Tensor
    sphere_hit: SphereHit
    reflected_directions: torch.Tensor
    secondary_origins: torch.Tensor
    timings_seconds: dict[str, float]


def generate_pinhole_rays(emitter: Any, camera_idx: int = 0) -> CameraRays:
    """Independently reproduce Nerfstudio perspective rays at pixel centers."""
    cameras = emitter.dataparser_outputs.cameras.flatten()
    camera = cameras[camera_idx]
    height, width = int(camera.height.item()), int(camera.width.item())
    dtype, device = torch.float32, emitter.device
    # Camera calibration is stored on CPU. Compute the raster projection there,
    # as the parity-reference Cameras implementation does, then transfer the finished
    # rays. This avoids CPU-vs-CUDA rounding drift before the NeRF query.
    camera_device = camera.camera_to_worlds.device
    y, x = torch.meshgrid(
        torch.arange(height, device=camera_device, dtype=dtype) + 0.5,
        torch.arange(width, device=camera_device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    fx, fy = float(camera.fx.item()), float(camera.fy.item())
    cx, cy = float(camera.cx.item()), float(camera.cy.item())
    camera_directions = torch.stack(((x - cx) / fx, -(y - cy) / fy, -torch.ones_like(x)), dim=-1)
    c2w = camera.camera_to_worlds.to(dtype=dtype)
    directions_ns = torch.sum(camera_directions[..., None, :] * c2w[:3, :3], dim=-1)
    directions_ns = directions_ns / torch.linalg.vector_norm(directions_ns, dim=-1, keepdim=True)
    origins_ns = c2w[:3, 3].expand_as(directions_ns).contiguous()
    directions_ns = directions_ns.to(device)
    origins_ns = origins_ns.to(device)
    origins_external, directions_external = emitter.nerfstudio_to_external(
        origins_ns.reshape(-1, 3), directions_ns.reshape(-1, 3)
    )
    # The inverse is only an axis permutation, so normalization is already
    # preserved exactly. Re-normalizing here would introduce a second float32
    # rounding and break bit-level background equivalence on very bright rays.
    return CameraRays(
        origins_ns=origins_ns,
        directions_ns=directions_ns,
        origins_external=origins_external.reshape(height, width, 3),
        directions_external=directions_external.reshape(height, width, 3),
        height=height,
        width=width,
    )


def reflect(incident: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Reflect normalized incident directions about normalized normals."""
    reflected = incident - 2.0 * torch.sum(incident * normal, dim=-1, keepdim=True) * normal
    return torch.nn.functional.normalize(reflected, dim=-1)


@torch.inference_mode()
def render_mirror_sphere(
    emitter: Any,
    camera_rays: CameraRays,
    center: torch.Tensor,
    radius: float,
    camera_idx: int = 0,
    epsilon: float = 1.0e-4,
    chunk_size: int = 16384,
    background: torch.Tensor | None = None,
) -> MirrorRender:
    """Render background rays and replace sphere hits by exact mirror radiance."""
    start_total = perf_counter()
    origins = camera_rays.origins_external.reshape(-1, 3)
    directions = camera_rays.directions_external.reshape(-1, 3)
    start = perf_counter()
    sphere_hit = intersect_sphere(origins, directions, center, radius)
    torch.cuda.synchronize(emitter.device)
    intersection_seconds = perf_counter() - start

    background_seconds = 0.0
    if background is None:
        start = perf_counter()
        background_flat = emitter.query_radiance(origins, directions, camera_idx, chunk_size=chunk_size)
        torch.cuda.synchronize(emitter.device)
        background_seconds = perf_counter() - start
        background = background_flat.reshape(camera_rays.height, camera_rays.width, 3)
    else:
        background_flat = background.reshape(-1, 3)

    hit_indices = torch.nonzero(sphere_hit.hit, as_tuple=False).squeeze(-1)
    hit_points = sphere_hit.point[hit_indices]
    hit_normals = sphere_hit.normal[hit_indices]
    hit_incident = directions[hit_indices]
    reflected = reflect(hit_incident, hit_normals)
    secondary_origins = hit_points + epsilon * hit_normals

    start = perf_counter()
    incident_radiance = emitter.query_radiance(
        secondary_origins, reflected, camera_idx, chunk_size=chunk_size
    )
    torch.cuda.synchronize(emitter.device)
    secondary_seconds = perf_counter() - start

    rgb_flat = background_flat.clone()
    # Perfect mirror: L_o(x, -d_cam) = L_i(x + epsilon*n, reflect(d_cam,n)).
    rgb_flat[hit_indices] = incident_radiance
    rgb = rgb_flat.reshape(camera_rays.height, camera_rays.width, 3)
    total_seconds = perf_counter() - start_total
    return MirrorRender(
        rgb=rgb,
        background=background,
        sphere_hit=sphere_hit,
        reflected_directions=reflected,
        secondary_origins=secondary_origins,
        timings_seconds={
            "intersection": intersection_seconds,
            "background_query": background_seconds,
            "secondary_query": secondary_seconds,
            "total": total_seconds,
        },
    )
