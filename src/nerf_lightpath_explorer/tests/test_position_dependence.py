"""Empirical position- and direction-dependence diagnostics."""

from __future__ import annotations

import json

import torch

from nerf_lightpath_explorer.tests._common import PROJECT_ROOT, load_emitter


def main() -> None:
    emitter = load_emitter()
    device = emitter.device
    camera = emitter.dataparser_outputs.cameras.to(device)
    camera_idx = 0
    h, w = int(camera.height[camera_idx].item()), int(camera.width[camera_idx].item())
    coords = torch.tensor(
        [[h * 0.35, w * 0.35], [h * 0.50, w * 0.50], [h * 0.65, w * 0.65]],
        dtype=torch.float32,
        device=device,
    )
    indices = torch.full((3, 1), camera_idx, dtype=torch.long, device=device)
    rays = camera.generate_rays(camera_indices=indices, coords=coords)
    external_origins, external_directions = emitter.nerfstudio_to_external(rays.origins, rays.directions)

    fixed_direction = external_directions[1:2].expand(3, 3).contiguous()
    position_offsets = torch.tensor(
        [[-0.08, 0.00, 0.00], [0.00, 0.00, 0.00], [0.08, 0.00, 0.00]],
        dtype=torch.float32,
        device=device,
    )
    position_origins = external_origins[1:2] + position_offsets
    position_rgb = emitter.query_radiance(position_origins, fixed_direction, camera_idx)
    position_differences = torch.stack([
        torch.linalg.vector_norm(position_rgb[0] - position_rgb[1]),
        torch.linalg.vector_norm(position_rgb[0] - position_rgb[2]),
        torch.linalg.vector_norm(position_rgb[1] - position_rgb[2]),
    ])

    fixed_origin = external_origins[1:2].expand(3, 3).contiguous()
    direction_rgb = emitter.query_radiance(fixed_origin, external_directions, camera_idx)
    direction_differences = torch.stack([
        torch.linalg.vector_norm(direction_rgb[0] - direction_rgb[1]),
        torch.linalg.vector_norm(direction_rgb[0] - direction_rgb[2]),
        torch.linalg.vector_norm(direction_rgb[1] - direction_rgb[2]),
    ])

    tolerance = 1e-6
    position_pass = bool(position_differences.max() > tolerance)
    direction_pass = bool(direction_differences.max() > tolerance)
    result = {
        "position_origins": position_origins.cpu().tolist(),
        "fixed_direction": fixed_direction[0].cpu().tolist(),
        "position_rgb": position_rgb.cpu().tolist(),
        "position_pairwise_l2": position_differences.cpu().tolist(),
        "direction_origin": fixed_origin[0].cpu().tolist(),
        "directions": external_directions.cpu().tolist(),
        "direction_rgb": direction_rgb.cpu().tolist(),
        "direction_pairwise_l2": direction_differences.cpu().tolist(),
    }
    output = PROJECT_ROOT / "results/r1_query/dependence_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    for index, rgb in enumerate(position_rgb.cpu().tolist(), start=1):
        print(f"o{index} RGB: {rgb}")
    print(f"position differences: {position_differences.cpu().tolist()}")
    print(f"POSITION_DEPENDENCE: {'PASS' if position_pass else 'FAIL'}")
    for index, rgb in enumerate(direction_rgb.cpu().tolist(), start=1):
        print(f"d{index} RGB: {rgb}")
    print(f"direction differences: {direction_differences.cpu().tolist()}")
    print(f"DIRECTION_DEPENDENCE: {'PASS' if direction_pass else 'FAIL'}")
    if not (position_pass and direction_pass):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
