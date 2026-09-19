"""Render the PBIR demo and evaluate its reconstruction metrics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw

from nerf_lightpath_explorer.pbir.demo_renderer import gaussian_splat,jittered_primary_rays,trace_primary,center_camera_rays
from nerf_lightpath_explorer.pbir.mitsuba_principled import MitsubaPrincipledTextureBSDF
from nerf_lightpath_explorer.pbir.spatial_vmf_guiding import SpatialVmfMixture
from nerf_lightpath_explorer.pbir.teapot_geometry import ExplicitTeapotGeometry
from nerf_lightpath_explorer.validate_image_reconstruction import load_exr
from nerf_lightpath_explorer.image_io import display_transform,save_exr,save_png
from nerf_lightpath_explorer.runtime_paths import ASSET_ROOT as ASSETS, CHECKPOINT_PATH, MODEL_ROOT as MODEL_RUNTIME, PROJECT_ROOT as ROOT, RESULTS_ROOT, load_environment as load_emitter

OUT=RESULTS_ROOT/"demo"; RES=256; PRIMARY_SPP=16; SECONDARY_SPP=16
LUMA=torch.tensor([.299,.587,.114]); FIREFLY_THRESHOLD=1.0


def bbox(mask:torch.Tensor,pad:int=8) -> tuple[int,int,int,int]:
    yy,xx=torch.nonzero(mask,as_tuple=True); return max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(RES,int(xx.max())+pad+1),min(RES,int(yy.max())+pad+1)


def save_comparisons(reference:torch.Tensor,ours:torch.Tensor,mask:torch.Tensor) -> None:
    a=display_transform(reference); b=display_transform(ours)
    side=np.concatenate((a,b),axis=1); canvas=Image.fromarray(side); draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,150,20),fill=(15,15,15)); draw.text((5,5),"Reference raw",fill=(255,255,255))
    draw.rectangle((RES,0,RES+150,20),fill=(15,15,15)); draw.text((RES+5,5),"Reconstructed",fill=(255,255,255)); canvas.save(OUT/"side_by_side.png")
    delta=(ours-reference).abs().mean(-1); scale=torch.quantile(delta,.99).clamp_min(1e-8); shown=(delta/scale).clamp(0,1)
    heat=torch.stack((shown,shown.sqrt(),torch.zeros_like(shown)),-1); Image.fromarray((heat.cpu().numpy()*255+.5).astype(np.uint8)).save(OUT/"difference.png")
    relative=(delta/reference.abs().mean(-1).clamp_min(1e-3)).clamp(0,4)/4
    relative_heat=torch.stack((relative,relative.sqrt(),torch.zeros_like(relative)),-1)
    Image.fromarray((relative_heat.cpu().numpy()*255+.5).astype(np.uint8)).save(OUT/"relative_difference.png")
    x0,y0,x1,y1=bbox(mask); Image.fromarray(np.concatenate((a[y0:y1,x0:x1],b[y0:y1,x0:x1]),axis=1)).save(OUT/"teapot_crop.png")
    ref_lum=(reference*LUMA.to(reference.device)).sum(-1); flat=torch.where(mask,ref_lum,torch.full_like(ref_lum,-1)).argmax(); hy,hx=int(flat//RES),int(flat%RES)
    r=32; xa,xb=max(0,hx-r),min(RES,hx+r); ya,yb=max(0,hy-r),min(RES,hy+r)
    Image.fromarray(np.concatenate((a[ya:yb,xa:xb],b[ya:yb,xa:xb]),axis=1)).save(OUT/"highlight_crop.png")
    # A crop centered on the widest silhouette/handle region.
    sy=(y0+y1)//2; xa=max(0,x0-4); xb=min(RES,x0+76); ya=max(0,sy-40); yb=min(RES,sy+40)
    Image.fromarray(np.concatenate((a[ya:yb,xa:xb],b[ya:yb,xa:xb]),axis=1)).save(OUT/"silhouette_crop.png")


@torch.inference_mode()
def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True); emitter=load_emitter(); device=emitter.device
    previous=Path.cwd(); os.chdir(MODEL_RUNTIME)
    try: test=emitter.config.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split="test")
    finally: os.chdir(previous)
    camera=test.cameras.flatten()[0]; geometry=ExplicitTeapotGeometry(ASSETS/"mesh.obj",ASSETS/"reflectance.png",ASSETS/"roughness.png")
    bsdf=MitsubaPrincipledTextureBSDF(ASSETS/"reflectance.png",ASSETS/"roughness.png",specular=1.0)
    vmf=SpatialVmfMixture.from_checkpoint(CHECKPOINT_PATH,device)
    samples=jittered_primary_rays(emitter,camera,RES,PRIMARY_SPP,12001,border_size=2)
    uv_in=OUT/"jittered_uv_input.npz"; uv_out=OUT/"jittered_uv_output.npz"
    np.savez(uv_in,o=samples.origins.cpu().numpy(),d=samples.directions.cpu().numpy())
    subprocess.run([sys.executable,str(ROOT/"src/nerf_lightpath_explorer/build_uv_correspondence.py"),str(uv_in),str(uv_out),str(ASSETS/"mesh.obj")],check=True)
    oracle=np.load(uv_out); uv=torch.from_numpy(oracle["uv"]).to(device).float()
    start=perf_counter(); trace=trace_primary(emitter,geometry,bsdf,vmf,samples.origins,samples.directions,uv,
                                              SECONDARY_SPP,"mis",0,12002)
    ours=gaussian_splat(samples,trace.rgb); torch.cuda.synchronize(device); elapsed=perf_counter()-start
    save_exr(OUT/"ours_final.exr",ours); save_exr(OUT/"demo_raw.exr",ours); save_png(OUT/"ours_final.png",ours)
    reference=load_exr(OUT/"expected_reference.exr",device)
    center=center_camera_rays(emitter,camera,RES); center_hit=geometry.intersect(center.origins_external.reshape(-1,3),center.directions_external.reshape(-1,3)).hit.reshape(RES,RES)
    expected_mask=torch.load(OUT/"expected_spp_256_mask.pt")[...,0].to(device)>.5; interior=center_hit&expected_mask
    delta=ours-reference; lum_error=(delta*LUMA.to(device)).sum(-1).abs(); ref_lum=(reference*LUMA.to(device)).sum(-1)
    highlight=interior&(ref_lum>=torch.quantile(ref_lum[interior],.95))
    try:
        from skimage.metrics import structural_similarity
        ssim=float(structural_similarity(display_transform(reference),display_transform(ours),channel_axis=-1,data_range=255))
    except Exception: ssim=None
    def region(mask):
        d=delta[mask]; peak=reference[mask].max(); rmse=torch.sqrt(d.square().mean())
        return {"pixels":int(mask.sum()),"mae":float(d.abs().mean()),"rmse":float(rmse),
                "psnr_db":float(20*torch.log10(peak/rmse.clamp_min(1e-12))),"mean_signed":float(d.mean()),"max_abs":float(d.abs().max())}
    metrics={"status":"PASS","resolution":[RES,RES],"effective_surface_spp":PRIMARY_SPP*SECONDARY_SPP,
             "primary_spp":PRIMARY_SPP,"secondary_spp_per_primary":SECONDARY_SPP,"seconds":elapsed,
             "sampling":"50% checkpoint vMF + 50% Mitsuba Principled BSDF, balanced MIS",
             "primary_filter":"independent uniform jitter, border=2, Mitsuba Gaussian stddev=0.5/radius=2 splat",
             "regions":{"full":region(torch.ones_like(interior)),"teapot":region(interior),"highlight":region(highlight)},
             "display_ssim":ssim,"silhouette_mismatch_pixels":int((center_hit!=expected_mask).sum()),
             "firefly_definition":"teapot abs luminance error versus reference raw > 1.0",
             "firefly_count":int((lum_error[interior]>FIREFLY_THRESHOLD).sum()),
             "p99_teapot_abs_luminance_error":float(torch.quantile(lum_error[interior],.99)),
             "all_finite":bool(torch.isfinite(ours).all()),"renderer_diagnostics":trace.diagnostics,
             "denoiser":"none; raw MIS result retained, OptiX unavailable"}
    if not metrics["all_finite"]: metrics["status"]="FAIL"
    (OUT/"reconstruction_metrics.json").write_text(json.dumps(metrics,indent=2)); save_comparisons(reference,ours,interior)
    print(json.dumps(metrics,indent=2)); print("DEMO RENDER: "+metrics["status"])


if __name__=="__main__": main()
