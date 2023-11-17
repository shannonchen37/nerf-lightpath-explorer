"""Acceptance checks for the four required D2 pixel classes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT=Path.home()/"nerf_inverse_rendering"; OUT=ROOT/"results/d2_pixel_inspector"


def main() -> None:
    pixels=json.loads((OUT/"acceptance_pixels.json").read_text()); checks={}
    for label,(x,y) in pixels.items():
        trace=json.loads((OUT/"traces"/f"pixel_{x}_{y}.json").read_text())
        selected=trace.get("nerf_selected_sample"); nerf=trace["nerf_selected_ray"]
        li_retrace_drift=max(abs(a-b) for a,b in zip(selected["Li"],nerf["final_Li"])) if selected else 0.0
        volume_reconstruction=max(abs(a-b) for a,b in zip(nerf["sum_weighted_rgb"],nerf["final_Li"]))
        checks[label]={"pixel":[x,y],"primary_hit":trace["primary_ray"]["hit"],
                       "secondary_samples":len(trace["secondary_samples"]),"volume_samples":nerf["sample_count"],
                       "pixel_reconstruction_max_abs":trace["reconstruction_max_abs"],
                       "stored_Li_vs_diagnostic_retrace_drift":li_retrace_drift,
                       "diagnostic_volume_sum_vs_final_Li_max_abs":volume_reconstruction,
                       "material_present":trace["material"] is not None,
                       "pdf_mis_fields_present":not selected or all(k in selected for k in ("bsdf_pdf","emitter_pdf","mis_weight","contribution"))}
    status="PASS" if all(v["pixel_reconstruction_max_abs"]<1e-5 and v["diagnostic_volume_sum_vs_final_Li_max_abs"]<1e-5 and v["volume_samples"]>0 and v["pdf_mis_fields_present"] for v in checks.values()) else "FAIL"
    result={"status":status,"classes":checks,"viewer":"reimpl/d2_viewer/server.py","trace_json_export":True,
            "exact_trace":"same D1 primary/path seeds, stored Li/contributions, and Gaussian reconstruction",
            "nerf_volume_trace":"explicit diagnostic retrace; internal volume sum must reproduce that retrace Li, while stateful proposal positions may differ from stored final-render Li",
            "fast_mode":"top-contribution subset for responsive inspection"}
    (OUT/"acceptance.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if status!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
