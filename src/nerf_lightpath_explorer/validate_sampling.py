"""Numerical validation of vMF PDF and Principled BSDF sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nerf_lightpath_explorer.pbir.mitsuba_principled import MitsubaPrincipledTextureBSDF
from nerf_lightpath_explorer.pbir.spatial_vmf_guiding import SpatialVmfMixture
from nerf_lightpath_explorer.runtime_pipeline import ASSETS,OUTPUT as H5_OUTPUT,setup_pipeline
from nerf_lightpath_explorer.tests._common import CHECKPOINT_PATH

ROOT=Path.home()/"nerf_inverse_rendering"; OUT=ROOT/"results/demo"


@torch.inference_mode()
def main() -> None:
    import drjit as dr
    import mitsuba as mi
    OUT.mkdir(parents=True,exist_ok=True); _,pipeline,_,_=setup_pipeline(); device=pipeline.device
    vmf=SpatialVmfMixture.from_checkpoint(CHECKPOINT_PATH,torch.device(device))
    env=pipeline.sdf_scene.environment(); params=mi.traverse(env)
    parameter_error={
        "position":float((torch.from_numpy(np.asarray(params["position"])).to(device).reshape(64,3)-vmf.position).abs().max()),
        "std":float((torch.from_numpy(np.asarray(params["std"])).to(device).reshape(64)-vmf.std).abs().max()),
        "weight":float((torch.from_numpy(np.asarray(params["weight"])).to(device).reshape(64)-vmf.weight).abs().max()),
    }
    g=torch.Generator(device=device).manual_seed(101); count=4096
    points=.25+.5*torch.rand((count,3),generator=g,device=device)
    directions=torch.nn.functional.normalize(torch.randn((count,3),generator=g,device=device),dim=-1)
    it=dr.zeros(mi.Interaction3f,count); it.p=mi.Point3f(points.cpu().numpy())
    ds=dr.zeros(mi.DirectionSample3f,count); ds.d=mi.Vector3f(directions.cpu().numpy()); ds.n=-ds.d
    expected=torch.from_numpy(np.asarray(env.pdf_direction(it,ds)).astype(np.float32,copy=True)).to(device)
    ours=vmf.pdf(points,directions); pdf_error=(ours-expected).abs()
    pdf_relative=pdf_error/expected.abs().clamp_min(1e-6)

    # Uniform-sphere integration of the mixture PDF at eight positions.
    integration=[]
    for p in points[:8]:
        d=torch.nn.functional.normalize(torch.randn((131072,3),generator=g,device=device),dim=-1)
        integration.append(float(4*torch.pi*vmf.pdf(p.expand(len(d),3),d).mean()))

    bsdf=MitsubaPrincipledTextureBSDF(ASSETS/"reflectance.png",ASSETS/"roughness.png",specular=1.0)
    n=torch.nn.functional.normalize(torch.randn((count,3),generator=g,device=device),dim=-1)
    wo=n.clone(); uv=torch.rand((count,2),generator=g,device=device)
    s1=torch.rand(count,generator=g,device=device); s2=torch.rand((count,2),generator=g,device=device)
    wi,sampled_pdf,_=bsdf.sample(n,wo,uv,s1,s2); evaluated_pdf=bsdf.evaluate(n,wi,wo,uv).pdf
    bsdf_pdf_error=(sampled_pdf-evaluated_pdf).abs()
    result={"status":"PASS","component_count":64,"checkpoint_vs_expected_emitter_max_abs":parameter_error,
            "vmf_pdf_vs_expected":{"mae":float(pdf_error.mean()),"max_abs":float(pdf_error.max()),
                                   "p99_relative":float(torch.quantile(pdf_relative,.99))},
            "vmf_pdf_uniform_sphere_integrals_mc_diagnostic":integration,
            "normalization_note":"analytic vMF components and normalized discrete mixture integrate to one; uniform-sphere MC has high variance for sharp kappa",
            "principled_sample_pdf_vs_eval":{"mae":float(bsdf_pdf_error.mean()),"max_abs":float(bsdf_pdf_error.max())},
            "mis":{"strategy_probability":{"emitter_vmf":.5,"principled_bsdf":.5},
                   "heuristic":"balanced","effective_denominator":"0.5 * (p_emitter + p_bsdf)"}}
    if max(parameter_error.values())>1e-6 or float(pdf_error.mean())>1e-5 or float(torch.quantile(pdf_relative,.99))>2e-4 or float(bsdf_pdf_error.mean())>2e-6 or float(bsdf_pdf_error.max())>3e-5:
        result["status"]="FAIL"
    (OUT/"sampler_unit_tests.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
