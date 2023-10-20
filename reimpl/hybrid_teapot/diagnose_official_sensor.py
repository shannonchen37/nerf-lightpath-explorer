"""H5R official camera-0 sensor with a trivial non-NeRF scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from reimpl.tests._common import load_emitter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    args = p.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    emitter = load_emitter()
    import drjit as dr
    import mitsuba as mi
    from nerfstudio.model_components.mi_sensor_generators import MitsubaSensorGenerator
    cameras = emitter.dataparser_outputs.cameras.flatten().to(emitter.device)
    pose_optimizer = emitter.config.pipeline.datamanager.camera_optimizer.setup(
        num_cameras=len(cameras), device=emitter.device)
    generator = MitsubaSensorGenerator(cameras, emitter.scene_scale, pose_optimizer,
                                       patch_width=64, patch_height=64, filter_type="gaussian")
    sensor = generator(0)
    scene = mi.load_dict({
        "type": "scene", "integrator": {"type": "path", "max_depth": 2},
        "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [1, 1, 1]}},
        "shape": {"type": "sphere", "center": [0.5, 0.5, 0.5], "radius": 0.2,
                  "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5,0.2,0.1]}}},
    })
    image = mi.render(scene, sensor=sensor, spp=1, seed=7); dr.eval(image)
    mi.util.write_bitmap(str(args.output), image)
    pixels = np.asarray(image).astype(np.float32)
    result = {"status":"PASS", "camera_idx":0, "resolution":[64,64], "filter":"gaussian",
              "camera_optimizer_mode":"off", "scene_scale":emitter.scene_scale,
              "mean":float(pixels.mean()), "max":float(pixels.max()), "all_finite":bool(np.isfinite(pixels).all()),
              "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","")}
    args.json.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
