"""Local D3 causal explorer server with cached GPU1 exact tracing/interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT=Path(__file__).resolve().parents[2]; D3=ROOT/"results/d3_causal_explorer"; TRACES=D3/"traces"; INTERVENTIONS=D3/"interventions"
DEFAULT_REMOTE_ENV="export CUDA_VISIBLE_DEVICES=1 MI_DEFAULT_VARIANT=cuda_ad_rgb PYTHONPATH=."


def intervention_key(mode,pixel,roughness,base,specular):
    value=json.dumps([mode,pixel,roughness,base,specular],separators=(",",":"))
    return hashlib.sha256(value.encode()).hexdigest()[:16]


class Handler(SimpleHTTPRequestHandler):
    remote=None
    remote_root=None
    remote_python="python"
    remote_env=DEFAULT_REMOTE_ENV
    def translate_path(self,path):
        clean=urlparse(path).path.lstrip("/")
        if not clean: clean="reimpl/d3_causal_explorer/index.html"
        return str(ROOT/clean)
    def json_response(self,payload,status=200):
        data=json.dumps(payload).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
    def remote_run(self,command):
        if not self.remote or not self.remote_root:
            raise RuntimeError("trace is not bundled; configure --remote and --remote-root for on-demand GPU recomputation")
        prefix=f"cd {shlex.quote(self.remote_root)} && {self.remote_env}"
        subprocess.run(["ssh",self.remote,f"{prefix} && {command}"],check=True)
    def sync(self,remote,local):
        local.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(["rsync","-a",f"{self.remote}:{self.remote_root}/{remote}",str(local)],check=True)
    def do_GET(self):
        p=urlparse(self.path); q=parse_qs(p.query)
        try:
            if p.path=="/api/trace":
                x,y=int(q["x"][0]),int(q["y"][0]); assert 0<=x<256 and 0<=y<256
                target=TRACES/f"pixel_{x}_{y}.json"
                if not target.exists():
                    self.remote_run(f"{shlex.quote(self.remote_python)} reimpl/run_d2_generate_traces.py --pixels {x},{y} && {shlex.quote(self.remote_python)} reimpl/run_d3_build_and_validate.py")
                    self.sync(f"results/d3_causal_explorer/traces/{target.name}",target)
                data=target.read_bytes(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
            if p.path=="/api/intervention":
                mode=q.get("mode",["frozen"])[0]; assert mode in ("frozen","full")
                x,y=int(q["x"][0]),int(q["y"][0]); assert 0<=x<256 and 0<=y<256
                rough=float(q["roughness"][0]) if q.get("roughness") else None; assert rough is None or 0<=rough<=1
                spec=float(q.get("specular",["1"])[0]); assert 0<=spec<=1
                base=None
                if q.get("base_color"):
                    base=tuple(float(v) for v in q["base_color"][0].split(",")); assert len(base)==3 and all(0<=v<=1 for v in base)
                key=intervention_key(mode,(x,y),rough,base,spec); name=f"{mode}_{x}_{y}_{key}.json"; target=INTERVENTIONS/name
                if not target.exists():
                    args=f"--mode {mode} --pixel {x},{y} --specular {spec}"
                    if rough is not None: args+=f" --roughness {rough}"
                    if base is not None: args+=f" --base-color {base[0]},{base[1]},{base[2]}"
                    self.remote_run(f"{shlex.quote(self.remote_python)} reimpl/run_d3_intervention.py {args}")
                    self.sync(f"results/d3_causal_explorer/interventions/{name}",target)
                data=target.read_bytes(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
            if p.path=="/api/sampler":
                self.json_response(json.loads((ROOT/"results/d1_official_demo/sampling_ab/sampling_ab_metrics.json").read_text())); return
            if p.path=="/api/gmm":
                self.json_response(json.loads((ROOT/"results/d1_official_demo/gmm_components.json").read_text())); return
        except Exception as exc:
            self.json_response({"status":"ERROR","error":str(exc)},500); return
        super().do_GET()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=8766)
    p.add_argument("--remote",default=os.environ.get("NERF_EXPLORER_REMOTE"))
    p.add_argument("--remote-root",default=os.environ.get("NERF_EXPLORER_REMOTE_ROOT"))
    p.add_argument("--remote-python",default=os.environ.get("NERF_EXPLORER_REMOTE_PYTHON","python"))
    p.add_argument("--remote-env",default=os.environ.get("NERF_EXPLORER_REMOTE_ENV",DEFAULT_REMOTE_ENV))
    a=p.parse_args(); Handler.remote=a.remote; Handler.remote_root=a.remote_root; Handler.remote_python=a.remote_python; Handler.remote_env=a.remote_env
    print(f"D3 explorer: http://127.0.0.1:{a.port}"); ThreadingHTTPServer(("127.0.0.1",a.port),Handler).serve_forever()


if __name__=="__main__": main()
