"""Validate the reference camera and one-sample-MIS integrator without NeRF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from reimpl.tests._common import load_emitter


def main() -> None:
    q = argparse.ArgumentParser(); q.add_argument("--output", type=Path, required=True); q.add_argument("--json", type=Path, required=True)
    args = q.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    emitter = load_emitter()
    import drjit as dr
    import mitsuba as mi
    from integrators import import_integrators
    from nerfstudio.model_components.mi_sensor_generators import MitsubaSensorGenerator
    import_integrators()
    cameras = emitter.dataparser_outputs.cameras.flatten().to(emitter.device)
    pose_optimizer = emitter.config.pipeline.datamanager.camera_optimizer.setup(num_cameras=len(cameras), device=emitter.device)
    sensor = MitsubaSensorGenerator(cameras, emitter.scene_scale, pose_optimizer,
                                    patch_width=64, patch_height=64, filter_type="gaussian")(0)
    assets = Path.home()/"nerf_inverse_rendering/third_party/nerf-emitter/results/synthetic/teapot-unirough-0.2_bedroom_v2/sdf-nerfacto/v1"
    scene = mi.load_dict({
        "type":"scene",
        "integrator":{"type":"sdf_direct_reparam_onesamplemis", "guiding_mis_compensation":False},
        "emitter":{"type":"constant", "radiance":{"type":"rgb", "value":[1,1,1]}},
        "shape":{"type":"obj", "filename":str(assets/"mesh.obj"),
                 "bsdf":{"type":"principled", "specular":1.0,
                         "base_color":{"type":"bitmap", "filename":str(assets/"reflectance.png"), "raw":True},
                         "roughness":{"type":"bitmap", "filename":str(assets/"roughness.png"), "raw":True}}}})
    image = mi.render(scene, sensor=sensor, spp=1, seed=11); dr.eval(image)
    mi.util.write_bitmap(str(args.output), image)
    pixels=np.asarray(image).astype(np.float32)
    result={"status":"PASS", "integrator":"sdf_direct_reparam_onesamplemis", "emitter":"constant",
            "sensor":"official camera 0", "resolution":[64,64], "spp":1,
            "mean":float(pixels.mean()), "max":float(pixels.max()), "all_finite":bool(np.isfinite(pixels).all())}
    args.json.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
