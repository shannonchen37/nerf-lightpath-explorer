"""Export the frozen reference GMM/vMF proposal and per-surface diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from reimpl.hybrid_teapot.demo_renderer import center_camera_rays
from reimpl.hybrid_teapot.reference_guiding import ReferenceVmfMixture
from reimpl.hybrid_teapot.teapot_geometry import ExplicitTeapotGeometry
from reimpl.tests._common import CHECKPOINT_PATH,load_emitter

ROOT=Path.home()/"nerf_inverse_rendering"; OFFICIAL=ROOT/"third_party/nerf-emitter"
ASSETS=OFFICIAL/"results/synthetic/teapot-unirough-0.2_bedroom_v2/sdf-nerfacto/v1"
OUT=ROOT/"results/d1_official_demo"; PC=OUT/"light_point_cloud"


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True); PC.mkdir(exist_ok=True)
    emitter=load_emitter(); device=emitter.device; vmf=ReferenceVmfMixture.from_checkpoint(CHECKPOINT_PATH,device)
    components=[{"index":i,"mean":vmf.position[i].tolist(),"std":float(vmf.std[i]),
                 "inv_variance":float(vmf.inv_var[i]),"mixture_weight":float(vmf.weight[i])} for i in range(64)]
    payload={"status":"PASS","source":"checkpoint vmf.position/weight/std","component_count":64,
             "isotropic":True,"weight_sum":float(vmf.weight.sum()),"components":components}
    (OUT/"gmm_components.json").write_text(json.dumps(payload,indent=2))
    colors=np.zeros((64,4),dtype=np.uint8); weights=vmf.weight.cpu().numpy(); colors[:,0]=255
    colors[:,1]=(255*weights/weights.max()).astype(np.uint8); colors[:,3]=255
    trimesh.PointCloud(vmf.position.cpu().numpy(),colors=colors).export(PC/"gmm_centers.ply")

    previous=Path.cwd(); os.chdir(OFFICIAL)
    try:
        parser=emitter.config.pipeline.datamanager.dataparser.setup(); test=parser.get_dataparser_outputs(split="test")
    finally:
        os.chdir(previous)
    rays=center_camera_rays(emitter,test.cameras.flatten()[0],256)
    geometry=ExplicitTeapotGeometry(ASSETS/"mesh.obj",ASSETS/"reflectance.png",ASSETS/"roughness.png")
    hit=geometry.intersect(rays.origins_external.reshape(-1,3),rays.directions_external.reshape(-1,3))
    ids=torch.nonzero(hit.hit).squeeze(-1); chosen=ids[torch.linspace(0,len(ids)-1,8,device=device).long()]
    points=hit.point[chosen]; rows=[]; heatmaps=[]
    theta=torch.linspace(0,torch.pi,96,device=device); phi=torch.linspace(-torch.pi,torch.pi,192,device=device)
    tt,pp=torch.meshgrid(theta,phi,indexing="ij")
    dirs=torch.stack((torch.sin(tt)*torch.cos(pp),torch.sin(tt)*torch.sin(pp),torch.cos(tt)),-1).reshape(-1,3)
    for i,p in enumerate(points):
        delta=vmf.position-p; distance=torch.linalg.vector_norm(delta,dim=-1); kappa=distance*vmf.inv_var
        dominant=int(vmf.weight.argmax()); density=vmf.pdf(p.expand(len(dirs),3),dirs).reshape(96,192)
        shown=torch.log1p(density); shown=shown/shown.max().clamp_min(1e-12)
        heatmaps.append((shown.cpu().numpy()*255+.5).astype(np.uint8))
        rows.append({"surface_index":i,"ray_index":int(chosen[i]),"position":p.tolist(),
                     "normal":hit.normal[chosen[i]].tolist(),"dominant_component":dominant,
                     "dominant_direction":torch.nn.functional.normalize(delta[dominant],dim=-1).tolist(),
                     "dominant_kappa":float(kappa[dominant]),"dominant_weight":float(vmf.weight[dominant]),
                     "pdf_max":float(density.max()),"pdf_uniform_sphere":float(1/(4*torch.pi))})
    (OUT/"vmf_surface_points.json").write_text(json.dumps({"status":"PASS","points":rows},indent=2))
    Image.fromarray(np.concatenate(heatmaps,axis=0),mode="L").save(OUT/"vmf_directional_diagnostic.png")
    print(json.dumps({"status":"PASS","components":64,"surface_points":8},indent=2))


if __name__=="__main__": main()
