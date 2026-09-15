"""D1 sampling A/B: cosine, BSDF, reconstructed vMF, and reference MIS."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image,ImageDraw

from reimpl.hybrid_teapot.demo_renderer import center_camera_rays,render_center
from reimpl.hybrid_teapot.mitsuba_principled import MitsubaPrincipledTextureBSDF
from reimpl.hybrid_teapot.reference_guiding import ReferenceVmfMixture
from reimpl.hybrid_teapot.teapot_geometry import ExplicitTeapotGeometry
from reimpl.run_r2_mirror_sphere import display_transform,save_exr,save_png
from reimpl.tests._common import CHECKPOINT_PATH,load_emitter

ROOT=Path.home()/"nerf_inverse_rendering"; OFFICIAL=ROOT/"third_party/nerf-emitter"
ASSETS=OFFICIAL/"results/synthetic/teapot-unirough-0.2_bedroom_v2/sdf-nerfacto/v1"
OUT=ROOT/"results/d1_official_demo"; AB=OUT/"sampling_ab"; RES=128
MODES=("cosine","bsdf","vmf","mis"); SPPS=(16,64,256)
LUMA=torch.tensor([.299,.587,.114]); FIREFLY_ABS_LUMINANCE_ERROR=1.0


def image_metrics(reference:torch.Tensor,image:torch.Tensor,mask:torch.Tensor,highlight:torch.Tensor) -> dict:
    delta=image-reference; lum_delta=(delta*LUMA.to(delta.device)).sum(-1).abs()
    selected=delta[mask]
    return {"rmse":float(torch.sqrt(selected.square().mean())),"mae":float(selected.abs().mean()),
            "highlight_rmse":float(torch.sqrt(delta[highlight].square().mean())),
            "firefly_threshold_abs_luminance_error":FIREFLY_ABS_LUMINANCE_ERROR,
            "firefly_count":int((lum_delta[mask]>FIREFLY_ABS_LUMINANCE_ERROR).sum()),
            "p99_abs_luminance_error":float(torch.quantile(lum_delta[mask],.99)),
            "mean_signed":float(selected.mean())}


def grid(images:dict[str,torch.Tensor],spp:int) -> None:
    panels=[]
    for name in MODES:
        canvas=Image.new("RGB",(RES,RES+20),(16,16,16)); canvas.paste(Image.fromarray(display_transform(images[name])),(0,20))
        ImageDraw.Draw(canvas).text((5,5),f"{name} {spp}spp",fill=(255,255,255)); panels.append(canvas)
    out=Image.new("RGB",(RES*4,RES+20));
    for i,panel in enumerate(panels): out.paste(panel,(i*RES,0))
    out.save(AB/f"grid_{spp}.png")


@torch.inference_mode()
def main() -> None:
    AB.mkdir(parents=True,exist_ok=True); emitter=load_emitter(); device=emitter.device
    previous=Path.cwd(); os.chdir(OFFICIAL)
    try:
        test=emitter.config.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split="test")
    finally: os.chdir(previous)
    camera=test.cameras.flatten()[0]; rays=center_camera_rays(emitter,camera,RES)
    geometry=ExplicitTeapotGeometry(ASSETS/"mesh.obj",ASSETS/"reflectance.png",ASSETS/"roughness.png")
    bsdf=MitsubaPrincipledTextureBSDF(ASSETS/"reflectance.png",ASSETS/"roughness.png",specular=1.0)
    vmf=ReferenceVmfMixture.from_checkpoint(CHECKPOINT_PATH,device)
    origins=rays.origins_external.reshape(-1,3); directions=rays.directions_external.reshape(-1,3)
    oracle_in=AB/"uv_input.npz"; oracle_out=AB/"uv_output.npz"
    np.savez(oracle_in,primary_o=origins.cpu().numpy(),primary_d=directions.cpu().numpy(),
             secondary_o=origins[:1].cpu().numpy(),secondary_d=directions[:1].cpu().numpy())
    subprocess.run([sys.executable,str(ROOT/"reimpl/run_h5_mitsuba_geometry.py"),str(oracle_in),str(oracle_out),str(ASSETS/"mesh.obj")],check=True)
    uv=torch.from_numpy(np.load(oracle_out)["uv"]).to(device).float()

    start=perf_counter(); reference=render_center(emitter,geometry,bsdf,vmf,rays,uv,1024,"mis",0,91024)
    reference_seconds=perf_counter()-start; save_exr(AB/"reference_mis_1024.exr",reference.rgb); save_png(AB/"reference_mis_1024.png",reference.rgb)
    mask=reference.hit; ref_lum=(reference.rgb*LUMA.to(device)).sum(-1)
    highlight=mask&(ref_lum>=torch.quantile(ref_lum[mask],.95))
    result={"status":"PASS","resolution":[RES,RES],"reference":"mis 1024 spp center-primary",
            "reference_seconds":reference_seconds,"highlight_definition":"top 5% reference teapot luminance",
            "firefly_definition":"teapot pixel absolute luminance error versus 1024-spp MIS > 1.0 linear HDR",
            "spp":{}}
    for spp in SPPS:
        images={}; result["spp"][str(spp)]={}
        for mode in MODES:
            start=perf_counter(); render=render_center(emitter,geometry,bsdf,vmf,rays,uv,spp,mode,0,90000+spp)
            elapsed=perf_counter()-start; images[mode]=render.rgb
            save_exr(AB/f"{mode}_{spp}.exr",render.rgb); save_png(AB/f"{mode}_{spp}.png",render.rgb)
            result["spp"][str(spp)][mode]={**image_metrics(reference.rgb,render.rgb,mask,highlight),
                                           "predicted_estimator_variance":render.sample_variance,
                                           "seconds":elapsed,"diagnostics":render.diagnostics}
        grid(images,spp)
        c=result["spp"][str(spp)]["cosine"]; m=result["spp"][str(spp)]["mis"]
        result["spp"][str(spp)]["mis_improvement_vs_cosine"]={
            "variance_reduction_percent":100*(1-m["predicted_estimator_variance"]/c["predicted_estimator_variance"]),
            "rmse_reduction_percent":100*(1-m["rmse"]/c["rmse"]),
            "firefly_reduction_percent":100*(1-m["firefly_count"]/max(c["firefly_count"],1)),
        }
    if not all(result["spp"][str(s)]["mis"]["rmse"]<result["spp"][str(s)]["cosine"]["rmse"] for s in SPPS): result["status"]="FAIL"
    (AB/"sampling_ab_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
