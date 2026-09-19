"""Export reconstruction raw and brightness-compensated NeRF light clouds."""

from __future__ import annotations

import json
import os
import torch

from nerf_lightpath_explorer.runtime_pipeline import setup_pipeline
from nerf_lightpath_explorer.runtime_paths import MODEL_ROOT as MODEL_RUNTIME, RESULTS_ROOT

OUT = RESULTS_ROOT / "demo" / "light_point_cloud"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(MODEL_RUNTIME)
    try:
        from nerfstudio.model_components.output_light_pc import compensate_pc, extract_light_point_cloud

        _, pipeline, checkpoint, step = setup_pipeline()
        raw, adjoint = extract_light_point_cloud(
            pipeline,
            torch_mi2gl_left=pipeline.torch_mi2gl_left,
            scene_scale=pipeline.scene_scale,
            output_filename=str(OUT / "raw"),
            ray_source=pipeline.config.ray_source,
            adjoint_sampling_strategy="primal",
            camera_scaling_factor=0.25,
            crop_bbox=pipeline.config.crop_bbox,
        )
        compensated = compensate_pc(
            **raw,
            output_filename=str(OUT / "compensated"),
            threshold=pipeline.config.primal_threshold,
        )
    finally:
        os.chdir(previous)

    lum = raw["lum_samples"].reshape(-1)
    summary = {
        "status": "PASS",
        "source": "runtime nerfstudio.model_components.output_light_pc",
        "checkpoint": str(checkpoint),
        "step": int(step),
        "ray_source": pipeline.config.ray_source,
        "camera_scaling_factor": 0.25,
        "crop_bbox": bool(pipeline.config.crop_bbox),
        "primal_threshold": float(pipeline.config.primal_threshold),
        "raw_point_count": int(raw["pos_samples"].shape[0]),
        "compensated_bright_point_count": int(compensated["position"].shape[0]),
        "raw_luminance_mean": float(lum.mean()),
        "raw_luminance_max": float(lum.max()),
        "adjoint_cloud_count": len(adjoint),
        "outputs": {
            "raw": "raw/pc_primal.ply",
            "compensated": "compensated/pc_compensated.ply",
            "frozen_gmm_centers": "gmm_centers.ply",
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
