"""Numerical equivalence, camera-ray, chunking, determinism, and HDR checks."""

from __future__ import annotations

import json

import torch

from reimpl.tests._common import PROJECT_ROOT, deterministic_rays, load_emitter


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    absolute = torch.abs(reference - candidate)
    relative = absolute / torch.clamp(torch.abs(reference), min=1e-6)
    return {
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "max_relative_error": float(relative.max()),
    }


def official_reference(emitter, origins, directions, camera_idx, near):
    from emitters.nerf_op import get_ray_bundle
    from nerfstudio.utils.mi_gl_conversion import mi2gl_left

    transform = torch.from_numpy(mi2gl_left).float().to(emitter.device)
    shifted_origins = origins + near * directions
    ray_bundle = get_ray_bundle(
        shifted_origins,
        directions,
        transform,
        emitter.scene_scale,
        camera_idx,
        emitter.rotater,
    )
    return emitter.model.get_rgb_for_camera_ray_bundle(ray_bundle)


def main() -> None:
    emitter = load_emitter()
    print(f"checkpoint loaded: {emitter.checkpoint_path} (step {emitter.checkpoint_step})")
    print(f"device: {emitter.device}")
    all_metrics = {}
    for ray_count in (1, 8, 128):
        origins, directions = deterministic_rays(ray_count, emitter.device)
        near = torch.linspace(0.0, 0.05, ray_count, device=emitter.device)[:, None]
        official_rgb = official_reference(emitter, origins, directions, 0, near)
        reimpl_rgb = emitter.query_radiance(origins, directions, camera_idx=0, near=near)
        metrics = error_metrics(official_rgb, reimpl_rgb)
        all_metrics[f"official_equivalence_n{ray_count}"] = metrics
        print(f"ray count: {ray_count}")
        print(f"official RGB: {official_rgb[:min(ray_count, 3)].cpu().tolist()}")
        print(f"reimpl RGB: {reimpl_rgb[:min(ray_count, 3)].cpu().tolist()}")
        print(json.dumps(metrics, indent=2))

    camera_idx = 0
    camera = emitter.dataparser_outputs.cameras.to(emitter.device)
    height = int(camera.height[camera_idx].item())
    width = int(camera.width[camera_idx].item())
    coords = torch.tensor([[height // 2, width // 2]], device=emitter.device, dtype=torch.float32)
    indices = torch.tensor([[camera_idx]], device=emitter.device, dtype=torch.long)
    camera_ray_bundle = camera.generate_rays(camera_indices=indices, coords=coords)
    if emitter.rotater is not None:
        camera_ray_bundle.rotater = emitter.rotater.apply_frustums
    camera_reference = emitter.model.get_rgb_for_camera_ray_bundle(camera_ray_bundle)
    external_origins, external_directions = emitter.nerfstudio_to_external(
        camera_ray_bundle.origins, camera_ray_bundle.directions
    )
    camera_candidate = emitter.query_radiance(external_origins, external_directions, camera_idx=camera_idx)
    camera_metrics = error_metrics(camera_reference, camera_candidate)
    all_metrics["camera_ray_equivalence"] = camera_metrics

    origins, directions = deterministic_rays(128, emitter.device)
    full = emitter.query_radiance(origins, directions, camera_idx=0, chunk_size=256)
    for chunk_size in (1, 8, 32):
        all_metrics[f"chunk_{chunk_size}"] = error_metrics(
            full, emitter.query_radiance(origins, directions, camera_idx=0, chunk_size=chunk_size)
        )
    repeated = [emitter.query_radiance(origins, directions, 0, chunk_size=32) for _ in range(3)]
    all_metrics["determinism_1_2"] = error_metrics(repeated[0], repeated[1])
    all_metrics["determinism_1_3"] = error_metrics(repeated[0], repeated[2])

    # Scan real camera rays, which are much more likely than random rays to see HDR emitters.
    hdr_values = []
    for idx in range(min(12, len(camera))):
        h, w = int(camera.height[idx].item()), int(camera.width[idx].item())
        ys = torch.linspace(0, h - 1, 12, device=emitter.device)
        xs = torch.linspace(0, w - 1, 12, device=emitter.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
        indices = torch.full((len(coords), 1), idx, device=emitter.device, dtype=torch.long)
        rays = camera.generate_rays(camera_indices=indices, coords=coords)
        if emitter.rotater is not None:
            rays.rotater = emitter.rotater.apply_frustums
        hdr_values.append(emitter.model.get_rgb_for_camera_ray_bundle(rays))
    hdr = torch.cat(hdr_values).flatten()
    quantiles = torch.quantile(hdr, torch.tensor([0.95, 0.99], device=emitter.device))
    hdr_stats = {
        "min": float(hdr.min()), "max": float(hdr.max()), "mean": float(hdr.mean()),
        "p95": float(quantiles[0]), "p99": float(quantiles[1]),
    }
    all_metrics["hdr"] = hdr_stats

    tolerance = 1e-5
    pass_equivalence = all(v["max_abs_error"] <= tolerance for k, v in all_metrics.items() if k.startswith("official_"))
    pass_camera = camera_metrics["max_abs_error"] <= tolerance
    pass_chunks = all(v["max_abs_error"] <= tolerance for k, v in all_metrics.items() if k.startswith("chunk_"))
    pass_determinism = all(v["max_abs_error"] <= tolerance for k, v in all_metrics.items() if k.startswith("determinism_"))
    passed = pass_equivalence and pass_camera and pass_chunks and pass_determinism and hdr_stats["max"] > 1.0
    output = PROJECT_ROOT / "results/r1_query/equivalence_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_metrics, indent=2))
    print(f"camera-ray metrics: {json.dumps(camera_metrics)}")
    print(f"HDR stats: {json.dumps(hdr_stats)}")
    print(f"R1 QUERY EQUIVALENCE: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
