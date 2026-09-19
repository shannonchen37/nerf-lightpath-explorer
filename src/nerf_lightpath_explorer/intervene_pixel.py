"""Exact frozen-path and deterministic full-pixel material interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from nerf_lightpath_explorer.pbir.aabb_control import set_aabb_disabled
from nerf_lightpath_explorer.pbir.demo_renderer import jittered_primary_rays
from nerf_lightpath_explorer.pbir.environment_query import EnvironmentNerfQuery
from nerf_lightpath_explorer.pbir.mitsuba_principled import MitsubaPrincipledInterventionBSDF
from nerf_lightpath_explorer.pbir.spatial_vmf_guiding import SpatialVmfMixture
from nerf_lightpath_explorer.pbir.teapot_geometry import ExplicitTeapotGeometry, _bilinear
from nerf_lightpath_explorer.runtime_paths import ASSET_ROOT as ASSETS, CHECKPOINT_PATH, MODEL_ROOT as MODEL_RUNTIME, RESULTS_ROOT, load_environment as load_emitter

DEMO_RESULTS = RESULTS_ROOT / "demo"; PIXEL_TRACES = RESULTS_ROOT / "pixel_traces" / "traces"
OUT = RESULTS_ROOT / "causal_explorer" / "interventions"
LUMA = torch.tensor([.299, .587, .114]); RES = 256; PRIMARY_SPP = 16; SECONDARY_SPP = 16
PRIMARY_SEED = 12001; PATH_SEED = 12002


def parse_color(value):
    if value is None: return None
    parts = tuple(float(x) for x in value.split(","))
    if len(parts) != 3 or any(x < 0 or x > 1 for x in parts): raise ValueError("base color must be r,g,b in [0,1]")
    return parts


def key_for(mode, pixel, roughness, base_color, specular):
    payload = json.dumps([mode, pixel, roughness, base_color, specular], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def vecsum(x): return x.sum(0) if len(x) else torch.zeros(3, device=x.device)


def stats(contribution, diffuse, specular, source_vmf, visible, background):
    total = contribution.sum(0) + background
    lum = (contribution * LUMA.to(contribution.device)).sum(-1)
    order = torch.argsort(lum, descending=True); denom = float((total * LUMA.to(total.device)).sum().clamp_min(1e-20))
    top = {str(n): float(lum[order[:n]].sum() / denom) for n in (1, 5, 10, 20, 50)}
    return {"rgb": total.tolist(), "diffuse_rgb": diffuse.sum(0).tolist(), "specular_rgb": specular.sum(0).tolist(),
            "bsdf_sampled_rgb": contribution[~source_vmf].sum(0).tolist(), "vmf_sampled_rgb": contribution[source_vmf].sum(0).tolist(),
            "visible_rgb": contribution[visible].sum(0).tolist(), "occluded_zeroed_rgb": contribution[~visible].sum(0).tolist(),
            "background_rgb": background.tolist(), "top_fractions": top,
            "diffuse_fraction": float((diffuse.sum(0)*LUMA.to(total.device)).sum()/max(float(((diffuse+specular).sum(0)*LUMA.to(total.device)).sum()),1e-20)),
            "specular_fraction": float((specular.sum(0)*LUMA.to(total.device)).sum()/max(float(((diffuse+specular).sum(0)*LUMA.to(total.device)).sum()),1e-20))}


def material_values(raw, surfaces, base_color, roughness, device):
    base = torch.tensor([s["base_color"] for s in surfaces], device=device, dtype=torch.float32)
    rough = torch.tensor([s["roughness"] for s in surfaces], device=device, dtype=torch.float32)
    if base_color is not None: base[:] = torch.tensor(base_color, device=device)
    if roughness is not None: rough[:] = roughness
    return base, rough


def frozen(raw, base_color, roughness, specular, device):
    paths = raw["secondary_samples"]; surface_by_id = {int(s["ray_id"]): s for s in raw["primary_surface_samples"]}
    surfaces = [surface_by_id[int(p["primary_ray_id"])] for p in paths]
    n = torch.tensor([s["shading_normal"] for s in surfaces], device=device); wo = torch.tensor([s["wo"] for s in surfaces], device=device)
    uv = torch.tensor([s["uv"] for s in surfaces], device=device); wi = torch.tensor([p["wi"] for p in paths], device=device)
    Li = torch.tensor([p["Li"] for p in paths], device=device); visible = torch.tensor([p["visible"] for p in paths], device=device, dtype=torch.bool)
    epdf = torch.tensor([p["emitter_pdf"] for p in paths], device=device); source_vmf = torch.tensor([p["source"] == "emitter_vmf" for p in paths], device=device)
    primary = raw["primary_reconstruction_samples"]; wsum = sum(float(x["filter_weight"]) for x in primary)
    weight = {int(x["ray_id"]): float(x["filter_weight"])/wsum/SECONDARY_SPP for x in primary}
    scale = torch.tensor([weight[int(p["primary_ray_id"])] for p in paths], device=device)
    background = sum((torch.tensor(x["radiance"], device=device)*float(x["filter_weight"])/wsum for x in primary if not x["hit"]), torch.zeros(3,device=device))
    bsdf = MitsubaPrincipledInterventionBSDF(ASSETS/"reflectance.png", ASSETS/"roughness.png",
                                              base_color=base_color, roughness=roughness, specular=specular)
    ev = bsdf.evaluate(n, wi, wo, uv); base, rough = material_values(raw, surfaces, base_color, roughness, device)
    lobes = bsdf.evaluate_lobes(n, wi, wo, uv, base, rough); denom = .5*(epdf+ev.pdf)
    c = torch.where(visible[:,None], Li*ev.weighted/denom.clamp_min(1e-12)[:,None], torch.zeros_like(Li))*scale[:,None]
    dc = torch.where(visible[:,None], Li*lobes["diffuse_weighted"]/denom.clamp_min(1e-12)[:,None], torch.zeros_like(Li))*scale[:,None]
    sc = torch.where(visible[:,None], Li*lobes["specular_weighted"]/denom.clamp_min(1e-12)[:,None], torch.zeros_like(Li))*scale[:,None]
    result = stats(c, dc, sc, source_vmf, visible, background)
    changed_lum = (c*LUMA.to(device)).sum(-1); ids=torch.argsort(changed_lum,descending=True)[:20]
    result["top_path_changes"]=[{"path_id":f"{paths[i]['primary_ray_id']}:{paths[i]['sample_id']}","rgb":c[i].tolist()} for i in ids.tolist()]
    result["invariants"]={"same_wi":True,"same_Li":True,"same_visibility":True,
                          "wi_sha256":hashlib.sha256(wi.cpu().numpy().tobytes()).hexdigest(),
                          "Li_sha256":hashlib.sha256(Li.cpu().numpy().tobytes()).hexdigest(),
                          "visibility_sha256":hashlib.sha256(visible.cpu().numpy().tobytes()).hexdigest()}
    return result


@torch.inference_mode()
def full(raw, base_color, roughness, specular, emitter):
    torch.manual_seed(33001); torch.cuda.manual_seed_all(33001)
    device=emitter.device; query=EnvironmentNerfQuery(emitter)
    previous=Path.cwd(); os.chdir(MODEL_RUNTIME)
    try: camera=emitter.config.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split="test").cameras.flatten()[0]
    finally: os.chdir(previous)
    geometry=ExplicitTeapotGeometry(ASSETS/"mesh.obj",ASSETS/"reflectance.png",ASSETS/"roughness.png")
    bsdf=MitsubaPrincipledInterventionBSDF(ASSETS/"reflectance.png",ASSETS/"roughness.png",base_color=base_color,roughness=roughness,specular=specular)
    vmf=SpatialVmfMixture.from_checkpoint(CHECKPOINT_PATH,device); primary=jittered_primary_rays(emitter,camera,RES,PRIMARY_SPP,PRIMARY_SEED,2)
    uv_all=torch.from_numpy(np.load(DEMO_RESULTS/"jittered_uv_output.npz")["uv"]).to(device).float(); hit=geometry.intersect(primary.origins,primary.directions)
    hit_ids=torch.nonzero(hit.hit).squeeze(-1); rank=torch.full((len(hit.hit),),-1,dtype=torch.long,device=device); rank[hit_ids]=torch.arange(len(hit_ids),device=device)
    generator=torch.Generator(device=device).manual_seed(PATH_SEED); uniforms=torch.rand((len(hit_ids),SECONDARY_SPP,4),generator=generator,device=device)
    selected=torch.tensor([int(p["ray_id"]) for p in raw["primary_reconstruction_samples"]],device=device); weights=torch.tensor([float(p["filter_weight"]) for p in raw["primary_reconstruction_samples"]],device=device); weights/=weights.sum()
    radiance=torch.zeros((len(selected),3),device=device); all_c=[]; all_dc=[]; all_sc=[]; all_src=[]; all_vis=[]
    misses=~hit.hit[selected]
    if misses.any(): radiance[misses]=query.query_environment_radiance(primary.origins[selected[misses]],primary.directions[selected[misses]],0)
    ids=selected[~misses]
    if len(ids):
        ranks=rank[ids]; p=hit.point[ids]; n=hit.normal[ids]; wo=torch.nn.functional.normalize(-primary.directions[ids],dim=-1); uv=uv_all[ids]; u=uniforms[ranks]
        nn=n[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3); ww=wo[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3); uuv=uv[:,None,:].expand(-1,SECONDARY_SPP,2).reshape(-1,2); points=p[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3)
        es=vmf.sample(points,u[...,1:3].reshape(-1,2)); bw,_,_=bsdf.sample(nn,ww,uuv,u[...,1].reshape(-1),u[...,2:4].reshape(-1,2)); choose=u[...,0].reshape(-1)<.5; wi=torch.where(choose[:,None],es.direction,bw)
        occluded=geometry.occluded(points+1e-4*nn,wi,1e-4); ev=bsdf.evaluate(nn,wi,ww,uuv); epdf=vmf.pdf(points,wi); denom=.5*(epdf+ev.pdf); valid=(~occluded)&(denom>0)
        Li=torch.zeros_like(wi); Li[valid]=query.query_environment_radiance((points+1e-4*nn)[valid],wi[valid],0)
        base=torch.from_numpy(_bilinear(geometry.reflectance,uv.cpu().numpy())).to(device); rough=torch.from_numpy(_bilinear(geometry.roughness_texture,uv.cpu().numpy())).to(device)
        if base_color is not None: base[:]=torch.tensor(base_color,device=device)
        if roughness is not None: rough[:]=roughness
        basef=base[:,None,:].expand(-1,SECONDARY_SPP,3).reshape(-1,3); roughf=rough[:,None].expand(-1,SECONDARY_SPP).reshape(-1); lobes=bsdf.evaluate_lobes(nn,wi,ww,uuv,basef,roughf)
        c=torch.where(valid[:,None],Li*ev.weighted/denom.clamp_min(1e-12)[:,None],torch.zeros_like(Li)); dc=torch.where(valid[:,None],Li*lobes["diffuse_weighted"]/denom.clamp_min(1e-12)[:,None],torch.zeros_like(Li)); sc=torch.where(valid[:,None],Li*lobes["specular_weighted"]/denom.clamp_min(1e-12)[:,None],torch.zeros_like(Li))
        surf=c.reshape(len(ids),SECONDARY_SPP,3).mean(1); radiance[~misses]=surf; primary_scale=weights[~misses,None,None]/SECONDARY_SPP
        all_c=c.reshape(len(ids),SECONDARY_SPP,3)*primary_scale; all_dc=dc.reshape(len(ids),SECONDARY_SPP,3)*primary_scale; all_sc=sc.reshape(len(ids),SECONDARY_SPP,3)*primary_scale
        all_c=all_c.reshape(-1,3); all_dc=all_dc.reshape(-1,3); all_sc=all_sc.reshape(-1,3); all_src=choose; all_vis=valid
    background=(radiance[misses]*weights[misses,None]).sum(0); result=stats(all_c,all_dc,all_sc,all_src,all_vis,background)
    result["rgb"]=(radiance*weights[:,None]).sum(0).tolist(); result["path_direction_sha256"]=hashlib.sha256(wi.cpu().numpy().tobytes()).hexdigest() if len(ids) else None
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("frozen","full"),required=True); ap.add_argument("--pixel",required=True); ap.add_argument("--roughness",type=float); ap.add_argument("--base-color"); ap.add_argument("--specular",type=float,default=1.0); ap.add_argument("--repeat",type=int,default=1); args=ap.parse_args()
    pixel=tuple(map(int,args.pixel.split(","))); base=parse_color(args.base_color)
    if args.roughness is not None and not 0<=args.roughness<=1: raise ValueError("roughness must be in [0,1]")
    if not 0<=args.specular<=1: raise ValueError("specular must be in [0,1]")
    raw=json.loads((PIXEL_TRACES/f"pixel_{pixel[0]}_{pixel[1]}.json").read_text()); OUT.mkdir(parents=True,exist_ok=True)
    emitter=load_emitter() if args.mode=="full" else None; device=emitter.device if emitter else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runs=[]
    for _ in range(args.repeat): runs.append(frozen(raw,base,args.roughness,args.specular,device) if args.mode=="frozen" else full(raw,base,args.roughness,args.specular,emitter))
    reproducibility=max(abs(runs[0]["rgb"][i]-r["rgb"][i]) for r in runs[1:] for i in range(3)) if len(runs)>1 else 0.0
    payload={"schema_version":"d3.1","status":"PASS","mode":"FROZEN_PATH" if args.mode=="frozen" else "FULL_PIXEL_RERENDER",
             "pixel":list(pixel),"intervention":{"roughness":args.roughness,"base_color":base,"specular":args.specular,"F0":.08*args.specular},
             "original":{"rgb":raw["reconstructed_rgb"],"material":raw["material"]},"result":runs[0],"repeat_count":len(runs),
             "reproducibility_max_abs":reproducibility,"causal_semantics":("same wi/Li/visibility; recompute material BSDF, dependent BSDF PDF/MIS, contribution" if args.mode=="frozen" else "material changes BSDF proposal, wi, visibility, NeRF Li, PDF/MIS, and contribution")}
    name=f"{args.mode}_{pixel[0]}_{pixel[1]}_{key_for(args.mode,pixel,args.roughness,base,args.specular)}.json"; (OUT/name).write_text(json.dumps(payload,indent=2)); print(json.dumps({"status":"PASS","output":str(OUT/name),"rgb":payload["result"]["rgb"],"reproducibility_max_abs":reproducibility},indent=2))


if __name__=="__main__": main()
