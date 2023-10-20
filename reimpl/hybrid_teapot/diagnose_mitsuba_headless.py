"""Pure offscreen Mitsuba CUDA smoke test for H5R runtime diagnosis."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import drjit as dr
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    diagnostic = {
        "hostname": platform.node(),
        "mitsuba_version": mi.__version__,
        "mitsuba_variants": list(mi.variants()),
        "mitsuba_variant": mi.variant(),
        "drjit_version": dr.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "display": os.environ.get("DISPLAY", ""),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", ""),
        "mi_default_variant": os.environ.get("MI_DEFAULT_VARIANT", ""),
        "drjit_liboptix_path": os.environ.get("DRJIT_LIBOPTIX_PATH", ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
            "sampler": {"type": "independent", "sample_count": 1},
            "film": {"type": "hdrfilm", "width": 32, "height": 32,
                     "pixel_format": "rgb", "pixel_filter": {"type": "box"}},
        },
        "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}},
        "shape": {"type": "sphere", "bsdf": {"type": "diffuse",
                  "reflectance": {"type": "rgb", "value": [0.6, 0.2, 0.1]}}},
    })
    image = mi.render(scene, spp=1, seed=5)
    dr.eval(image)
    mi.util.write_bitmap(str(args.output), image)
    pixels = np.asarray(image).astype(np.float32)
    diagnostic.update({"status": "PASS", "shape": list(image.shape),
                       "mean": float(pixels.mean()), "all_finite": bool(np.isfinite(pixels).all())})
    args.json.write_text(json.dumps(diagnostic, indent=2))
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
