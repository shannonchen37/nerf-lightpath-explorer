"""Camera-ray construction shared by the PBIR validation tools."""

from __future__ import annotations

import torch

from nerf_lightpath_explorer.renderers.mirror_sphere_renderer import CameraRays


def scaled_camera_rays(emitter, resolution: int, camera_idx: int = 0) -> CameraRays:
    """Project a virtual square camera with scaled intrinsics."""
    camera = emitter.dataparser_outputs.cameras.flatten()[camera_idx]
    source_h, source_w = int(camera.height.item()), int(camera.width.item())
    if source_h != source_w:
        raise ValueError("scaled camera helper expects a square camera")
    scale = resolution / source_w
    y, x = torch.meshgrid(
        torch.arange(resolution, dtype=torch.float32) + 0.5,
        torch.arange(resolution, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    fx, fy = float(camera.fx.item()) * scale, float(camera.fy.item()) * scale
    cx, cy = float(camera.cx.item()) * scale, float(camera.cy.item()) * scale
    local = torch.stack(((x - cx) / fx, -(y - cy) / fy, -torch.ones_like(x)), dim=-1)
    c2w = camera.camera_to_worlds.float()
    directions_ns = torch.sum(local[..., None, :] * c2w[:3, :3], dim=-1)
    directions_ns = directions_ns / torch.linalg.vector_norm(directions_ns, dim=-1, keepdim=True)
    origins_ns = c2w[:3, 3].expand_as(directions_ns).contiguous()
    directions_ns, origins_ns = directions_ns.to(emitter.device), origins_ns.to(emitter.device)
    origins_external, directions_external = emitter.nerfstudio_to_external(
        origins_ns.reshape(-1, 3), directions_ns.reshape(-1, 3)
    )
    return CameraRays(
        origins_ns,
        directions_ns,
        origins_external.reshape(resolution, resolution, 3),
        directions_external.reshape(resolution, resolution, 3),
        resolution,
        resolution,
    )
