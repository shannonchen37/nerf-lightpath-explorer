"""Build explorer traces and validate attribution/UI consistency without rerendering."""

from __future__ import annotations

import json
import random
from nerf_lightpath_explorer.causal_explorer.schema import build_pixel_trace, lum, mul
from nerf_lightpath_explorer.runtime_paths import RESULTS_ROOT

RAW = RESULTS_ROOT / "pixel_traces" / "traces"
OUT = RESULTS_ROOT / "causal_explorer"
GMM = RESULTS_ROOT / "demo" / "gmm_components.json"


def max_abs(v): return max(abs(float(x)) for x in v)


def main() -> None:
    trace_out = OUT / "traces"; trace_out.mkdir(parents=True, exist_ok=True)
    gmm = json.loads(GMM.read_text()); built = []; raw_by_pixel = {}
    for path in sorted(RAW.glob("pixel_*_*.json")):
        raw = json.loads(path.read_text())
        if raw["secondary_samples"] and "diffuse_contribution" not in raw["secondary_samples"][0]:
            continue
        d3 = build_pixel_trace(raw, gmm); pixel = tuple(d3["pixel"]); raw_by_pixel[pixel] = raw; built.append(d3)
        (trace_out / path.name).write_text(json.dumps(d3, separators=(",", ":")))

    teapot = [t for t in built if t["primary"]["hit"]]
    closures = []
    for t in teapot:
        closures.append({"pixel": t["pixel"],
                         "attribution_max_abs": max_abs(t["contribution_summary"]["attribution_residual"]),
                         "lobe_max_abs": max_abs(t["contribution_summary"]["lobe_residual"]),
                         "recorded_reconstruction_max_abs": float(t["reconstruction"]["max_abs"])})
    closure = {"status": "PASS" if len(teapot) >= 20 and max((max(x["attribution_max_abs"], x["lobe_max_abs"]) for x in closures), default=1) < 1e-5 else "FAIL",
               "schema_version": "d3.1", "teapot_pixel_count": len(teapot), "tolerance": 1e-5,
               "max_attribution_abs": max((x["attribution_max_abs"] for x in closures), default=None),
               "max_lobe_abs": max((x["lobe_max_abs"] for x in closures), default=None), "pixels": closures}
    (OUT / "attribution_closure.json").write_text(json.dumps(closure, indent=2))

    candidates = [(t, p) for t in built for p in t["paths"]]
    rng = random.Random(20260915); checks = []
    for t, p in rng.sample(candidates, min(50, len(candidates))):
        raw = raw_by_pixel[tuple(t["pixel"])]; source = raw["secondary_samples"][p["source_index"]]
        primary = raw["primary_reconstruction_samples"]
        total_weight = sum(float(x["filter_weight"]) for x in primary)
        source_primary = next(x for x in primary if int(x["ray_id"]) == int(source["primary_ray_id"]))
        expected_pixel = mul(source["contribution"], float(source_primary["filter_weight"]) / total_weight / raw["final_renderer"]["secondary_spp"])
        errors = {
            "wi": max(abs(p["wi"][i] - source["wi"][i]) for i in range(3)),
            "Li": max(abs(p["Li"][i] - source["Li"][i]) for i in range(3)),
            "bsdf_pdf": abs(p["bsdf_pdf"] - source["bsdf_pdf"]),
            "emitter_pdf": abs(p["emitter_pdf"] - source["emitter_pdf"]),
            "mis_weight": abs(p["mis_weight"] - source["mis_weight"]),
            "pixel_contribution": max(abs(p["pixel_contribution"][i] - expected_pixel[i]) for i in range(3)),
        }
        checks.append({"pixel": t["pixel"], "path_id": p["id"], "errors": errors})
    ui = {"status": "PASS" if len(checks) == 50 and max((max(c["errors"].values()) for c in checks), default=1) < 1e-12 else "FAIL",
          "schema_version": "d3.1", "checked_paths": len(checks), "tolerance": 1e-12,
          "max_error": max((max(c["errors"].values()) for c in checks), default=None), "checks": checks}
    (OUT / "ui_path_consistency.json").write_text(json.dumps(ui, indent=2))

    presets = {"background": (16, 16), "diffuse": (141, 153), "highlight": (151, 109), "silhouette": (25, 112)}
    preset_results = {}
    for name, pixel in presets.items():
        match = next((t for t in built if tuple(t["pixel"]) == pixel), None)
        preset_results[name] = {"pixel": list(pixel), "present": match is not None,
                                "reconstruction_max_abs": match["reconstruction"]["max_abs"] if match else None,
                                "shading_regime": match["why"]["shading_regime"] if match else None}
    all_paths = [p for t in built for p in t["paths"]]
    high_spec = max(all_paths, key=lambda p: lum(p["specular_pixel_contribution"]), default=None)
    occluded = next((p for p in all_paths if p["occluded"]), None)
    vmf_pixel = max(built, key=lambda t: lum(t["contribution_summary"]["vmf_sampled"]) / max(lum(t["contribution_summary"]["path_total"]), 1e-20), default=None)
    extra = {
        "high_contribution_specular_path": {"path_id": high_spec["id"], "luminance": high_spec["luminance"]} if high_spec else None,
        "occluded_path": {"path_id": occluded["id"], "contribution": occluded["pixel_contribution"]} if occluded else None,
        "vmf_dominant_pixel": {"pixel": vmf_pixel["pixel"], "vmf_fraction": lum(vmf_pixel["contribution_summary"]["vmf_sampled"]) / max(lum(vmf_pixel["contribution_summary"]["path_total"]), 1e-20)} if vmf_pixel else None,
    }
    acceptance = {"status": "PASS" if closure["status"] == ui["status"] == "PASS" and all(x["present"] for x in preset_results.values()) and all(extra.values()) else "FAIL",
                  "schema_version": "d3.1", "trace_count": len(built), "presets": preset_results,
                  "extra_cases": extra, "attribution_closure": closure["status"], "ui_path_consistency": ui["status"]}
    (OUT / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    print(json.dumps({"status": acceptance["status"], "traces": len(built), "teapot": len(teapot),
                      "closure": closure["status"], "ui": ui["status"]}, indent=2))


if __name__ == "__main__": main()
