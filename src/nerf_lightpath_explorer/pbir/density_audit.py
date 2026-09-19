"""Evaluate main/proposal densities at identical points with AABB enabled/disabled."""

from __future__ import annotations

import torch

from nerf_lightpath_explorer.pbir.aabb_control import set_aabb_disabled


def _point_ray_samples(points: torch.Tensor):
    from nerfstudio.cameras.rays import Frustums, RaySamples

    count = len(points)
    origins = points[:, None, :]
    directions = torch.zeros_like(origins)
    starts = torch.zeros((count, 1, 1), dtype=points.dtype, device=points.device)
    ends = torch.zeros_like(starts)
    pixel_area = torch.ones_like(starts)
    frustums = Frustums(origins, directions, starts, ends, pixel_area)
    return RaySamples(frustums=frustums, camera_indices=torch.zeros((count, 1, 1), dtype=torch.long, device=points.device))


@torch.inference_mode()
def audit_density_disable(model, grid_resolution: int = 9) -> dict:
    aabb = model.field.aabb.detach()
    axis = torch.linspace(0.05, 0.95, grid_resolution, device=aabb.device)
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    inside = aabb[0] + xyz * (aabb[1] - aabb[0])
    samples = _point_ray_samples(inside)

    def density(module, disabled: bool) -> torch.Tensor:
        with set_aabb_disabled(model, disabled):
            return module.get_density(samples)[0].detach().float().reshape(-1)

    modules = {"nerfacto_field": model.field}
    modules.update({f"proposal_network_{i}": network for i, network in enumerate(model.proposal_networks)})
    result = {}
    for name, module in modules.items():
        full = density(module, False)
        disabled = density(module, True)
        result[name] = {
            "sample_count_inside_bbox": len(full),
            "full_nonzero_count": int((full > 0).sum()),
            "full_mean": float(full.mean()),
            "full_max": float(full.max()),
            "disabled_nonzero_count": int((disabled > 0).sum()),
            "disabled_max_abs": float(disabled.abs().max()),
            "all_disabled_exact_zero": bool(torch.equal(disabled, torch.zeros_like(disabled))),
        }
    return {
        "aabb_nerfstudio": aabb.cpu().tolist(),
        "aabb_external_emitter_local": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "grid_resolution": grid_resolution,
        "modules": result,
        "all_modules_exact_zero_inside": all(item["all_disabled_exact_zero"] for item in result.values()),
    }

