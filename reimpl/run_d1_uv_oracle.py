"""Memory-conscious Mitsuba UV oracle for D1 jittered primary rays."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("output",type=Path); p.add_argument("mesh",type=Path)
    a=p.parse_args(); rays=np.load(a.input,mmap_mode="r")
    scene=mi.load_dict({"type":"scene","shape":{"type":"obj","filename":str(a.mesh),"bsdf":{"type":"diffuse"}}})
    count=len(rays["o"]); uv=np.zeros((count,2),np.float32); valid=np.zeros(count,bool); primitive=np.full(count,-1,np.int64)
    chunk=262144
    for start in range(0,count,chunk):
        end=min(start+chunk,count); ray=mi.Ray3f(mi.Point3f(rays["o"][start:end]),mi.Vector3f(rays["d"][start:end]))
        si=scene.ray_intersect(ray); valid[start:end]=np.asarray(si.is_valid()); uv[start:end]=np.asarray(si.uv).astype(np.float32)
        primitive[start:end]=np.asarray(si.prim_index).astype(np.int64)
    np.savez(a.output,valid=valid,uv=uv,primitive_index=primitive)


if __name__=="__main__": main()
