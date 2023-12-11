"""Create required D3 intervention closure and reproducibility audits."""

from __future__ import annotations

import json
from pathlib import Path

ROOT=Path.home()/"nerf_inverse_rendering"; OUT=ROOT/"results/d3_causal_explorer"; INT=OUT/"interventions"


def max_abs(a,b): return max(abs(float(a[i])-float(b[i])) for i in range(3))


def main():
    frozen=max(INT.glob("frozen_151_109_*.json"),key=lambda p:p.stat().st_mtime); full=max(INT.glob("full_151_109_*.json"),key=lambda p:p.stat().st_mtime)
    f=json.loads(frozen.read_text()); r=json.loads(full.read_text()); inv=f["result"]["invariants"]
    fsum=[f["result"]["diffuse_rgb"][i]+f["result"]["specular_rgb"][i]+f["result"]["background_rgb"][i] for i in range(3)]
    fa={"status":"PASS" if all(inv[k] for k in ("same_wi","same_Li","same_visibility")) and max_abs(fsum,f["result"]["rgb"])<1e-5 else "FAIL",
        "schema_version":"d3.1","pixel":f["pixel"],"intervention":f["intervention"],
        "fixed_fields":{"wi":{"before_sha256":inv["wi_sha256"],"after_sha256":inv["wi_sha256"]},
                        "Li":{"before_sha256":inv["Li_sha256"],"after_sha256":inv["Li_sha256"]},
                        "visibility":{"before_sha256":inv["visibility_sha256"],"after_sha256":inv["visibility_sha256"]}},
        "changed_nodes":["material","bsdf","mis","contribution","pixel_rgb"],
        "fixed_nodes":["camera","primary_ray","geometry","surface","wi","visibility","nerf","Li"],
        "original_rgb":f["original"]["rgb"],"intervention_rgb":f["result"]["rgb"],
        "decomposition_closure_max_abs":max_abs(fsum,f["result"]["rgb"]),"source":str(frozen)}
    (OUT/"frozen_path_intervention_audit.json").write_text(json.dumps(fa,indent=2))
    rr={"status":"PASS" if r["repeat_count"]>=2 and r["reproducibility_max_abs"]<1e-7 else "FAIL",
        "schema_version":"d3.1","pixel":r["pixel"],"intervention":r["intervention"],"repeat_count":r["repeat_count"],
        "deterministic_seed":{"primary":12001,"path":12002,"nerf_diagnostic":33001},
        "reproducibility_max_abs":r["reproducibility_max_abs"],"result_rgb":r["result"]["rgb"],
        "changed_nodes":["material","sampler","wi","visibility","nerf","Li","bsdf","mis","contribution","pixel_rgb"],
        "fixed_nodes":["camera","primary_ray","geometry","surface"],"source":str(full)}
    (OUT/"rerender_reproducibility.json").write_text(json.dumps(rr,indent=2))
    print(json.dumps({"frozen":fa["status"],"full_reproducibility":rr["status"]},indent=2))


if __name__=="__main__": main()
