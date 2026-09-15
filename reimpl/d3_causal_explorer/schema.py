"""Stable d3.1 trace schema and exact attribution derived from saved D1 paths."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "d3.1"
LUMA = (0.299, 0.587, 0.114)


def add(a, b): return [float(a[i] + b[i]) for i in range(3)]
def sub(a, b): return [float(a[i] - b[i]) for i in range(3)]
def mul(a, s): return [float(a[i] * s) for i in range(3)]
def lum(a): return float(sum(a[i] * LUMA[i] for i in range(3)))
def zero(): return [0.0, 0.0, 0.0]


def sum_rgb(rows, key):
    out = zero()
    for row in rows: out = add(out, row[key])
    return out


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_pixel_trace(raw: dict, gmm: dict) -> dict:
    secondary_spp = int(raw["final_renderer"]["secondary_spp"])
    primary = raw["primary_reconstruction_samples"]
    weight_sum = sum(float(p["filter_weight"]) for p in primary)
    primary_weight = {int(p["ray_id"]): float(p["filter_weight"]) / weight_sum for p in primary}
    surface_by_ray = {int(s["ray_id"]): s for s in raw.get("primary_surface_samples", [])}

    background = zero()
    for p in primary:
        if not p["hit"]:
            background = add(background, mul(p["radiance"], primary_weight[int(p["ray_id"])]))

    paths = []
    for source_index, s in enumerate(raw["secondary_samples"]):
        ray_id = int(s["primary_ray_id"])
        scale = primary_weight[ray_id] / secondary_spp
        contribution = mul(s["contribution"], scale)
        diffuse = mul(s["diffuse_contribution"], scale)
        specular = mul(s["specular_contribution"], scale)
        surface = surface_by_ray[ray_id]
        component = int(s["component"]) if s["source"] == "emitter_vmf" else None
        row = {
            "id": f"{ray_id}:{int(s['sample_id'])}", "source_index": source_index,
            "primary_ray_id": ray_id, "sample_id": int(s["sample_id"]),
            "source": "vMF/emitter" if s["source"] == "emitter_vmf" else "BSDF",
            "visible": bool(s["visible"]), "occluded": bool(s["occluded"]),
            "origin": surface["position"], "wi": s["wi"], "Li": s["Li"],
            "bsdf_f_cos": s["bsdf_f_cos"], "diffuse_f_cos": s["diffuse_f_cos"],
            "specular_f_cos": s["specular_f_cos"], "cos_theta": float(s["cos_theta"]),
            "NoV": float(s["NoV"]), "bsdf_pdf": float(s["bsdf_pdf"]),
            "emitter_pdf": float(s["emitter_pdf"]), "proposal_pdf": float(s["proposal_pdf"]),
            "selection_probability": float(s["selection_probability"]), "mis_weight": float(s["mis_weight"]),
            "component": component, "vmf_mu": s["vmf_mu"] if component is not None else None,
            "vmf_kappa": float(s["vmf_kappa"]) if component is not None else None,
            "microfacet_diagnostic": {"label": "DIAGNOSTIC — not a complete principled decomposition",
                                       "D": float(s["D"]), "F": s["F"], "G": float(s["G"])},
            "sample_contribution": s["contribution"], "pixel_contribution": contribution,
            "diffuse_pixel_contribution": diffuse, "specular_pixel_contribution": specular,
            "luminance": lum(contribution), "Li_luminance": lum(s["Li"]),
        }
        row["lobe"] = "specular" if lum(specular) >= lum(diffuse) else "diffuse"
        paths.append(row)

    paths.sort(key=lambda p: p["luminance"], reverse=True)
    reconstructed = [float(x) for x in raw["reconstructed_rgb"]]
    total_luminance = max(lum(reconstructed), 1e-20)
    cumulative = 0.0
    for rank, p in enumerate(paths, 1):
        p["rank"] = rank
        p["energy_fraction"] = p["luminance"] / total_luminance
        cumulative += p["energy_fraction"]
        p["cumulative_fraction"] = cumulative

    path_total = sum_rgb(paths, "pixel_contribution")
    diffuse = sum_rgb(paths, "diffuse_pixel_contribution")
    specular = sum_rgb(paths, "specular_pixel_contribution")
    source_bsdf = sum_rgb((p for p in paths if p["source"] == "BSDF"), "pixel_contribution")
    source_vmf = sum_rgb((p for p in paths if p["source"] == "vMF/emitter"), "pixel_contribution")
    visible = sum_rgb((p for p in paths if p["visible"]), "pixel_contribution")
    occluded = sum_rgb((p for p in paths if p["occluded"]), "pixel_contribution")
    attributed = add(background, path_total)
    residual = sub(reconstructed, attributed)
    lobe_residual = sub(path_total, add(diffuse, specular))

    def top_fraction(n): return sum(p["energy_fraction"] for p in paths[:n])
    top = {str(n): top_fraction(n) for n in (1, 5, 10, 12, 20, 50, 100, 200)}
    needed = {}
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        needed[str(threshold)] = next((p["rank"] for p in paths if p["cumulative_fraction"] >= threshold), None)

    component_rgb = {}
    for p in paths:
        if p["component"] is not None:
            key = str(p["component"]); component_rgb[key] = add(component_rgb.get(key, zero()), p["pixel_contribution"])
    components = []
    gmm_by_id = {str(c["index"]): c for c in gmm["components"]}
    for key, rgb in sorted(component_rgb.items(), key=lambda item: lum(item[1]), reverse=True):
        c = gmm_by_id[key]
        components.append({"component": int(key), "pixel_contribution": rgb,
                           "fraction": lum(rgb) / total_luminance,
                           "centroid": c["mean"], "spatial_std": c["std"],
                           "mixture_weight": c["mixture_weight"]})

    diffuse_l, specular_l = lum(diffuse), lum(specular)
    lobe_sum = max(diffuse_l + specular_l, 1e-20)
    specular_fraction = specular_l / lobe_sum
    if not raw["primary_ray"]["hit"]: regime = "Background NeRF pixel"
    elif specular_fraction >= 0.6: regime = "Specular-dominated"
    elif specular_fraction <= 0.4: regime = "Diffuse-dominated"
    else: regime = "Mixed diffuse/specular"

    dominant = components[0] if components else None
    why = {
        "pixel": raw["pixel"], "surface": "Teapot" if raw["primary_ray"]["hit"] else "Background NeRF",
        "shading_regime": regime, "material": "Parity-validated recovered principled" if raw["material"] else None,
        "roughness": raw["material"]["roughness"] if raw["material"] else None,
        "F0": raw["material"]["F0"] if raw["material"] else None,
        "dominant_lighting": ({"label": "Bright environment region", "gmm_component": dominant["component"],
                               "centroid": dominant["centroid"]} if dominant else
                              {"label": "Direct background NeRF camera ray"}),
        "top_12_fraction": top["12"], "specular_fraction": specular_fraction if paths else None,
        "diffuse_fraction": diffuse_l / lobe_sum if paths else None,
        "recorded_rgb": raw["recorded_final_rgb"], "reconstructed_rgb": reconstructed,
        "reconstruction_max_abs": float(raw["reconstruction_max_abs"]),
    }

    teapot_hit = bool(raw["primary_ray"]["hit"])
    dag = {
        "nodes": (["camera", "primary_ray", "geometry", "surface", "material", "sampler", "wi",
                   "visibility", "nerf", "Li", "bsdf", "mis", "contribution", "pixel_rgb"] if teapot_hit
                  else ["camera", "primary_ray", "nerf", "Li", "pixel_rgb"]),
        "edges": ([["camera", "primary_ray"], ["primary_ray", "geometry"], ["geometry", "surface"],
                   ["surface", "material"], ["surface", "sampler"], ["sampler", "wi"],
                   ["wi", "visibility"], ["wi", "nerf"], ["nerf", "Li"],
                   ["visibility", "contribution"], ["Li", "bsdf"], ["material", "bsdf"],
                   ["bsdf", "mis"], ["sampler", "mis"], ["mis", "contribution"],
                   ["contribution", "pixel_rgb"]] if teapot_hit
                  else [["camera", "primary_ray"], ["primary_ray", "nerf"], ["nerf", "Li"], ["Li", "pixel_rgb"]]),
        "intervention_dependencies": {
            "frozen_material": {"changes": ["material", "bsdf", "mis", "contribution", "pixel_rgb"],
                                "fixed": ["camera", "primary_ray", "geometry", "surface", "wi", "visibility", "nerf", "Li"]},
            "full_material": {"changes": ["material", "sampler", "wi", "visibility", "nerf", "Li", "bsdf", "mis", "contribution", "pixel_rgb"],
                              "fixed": ["camera", "primary_ray", "geometry", "surface"]},
        },
    }
    return {
        "schema_version": SCHEMA_VERSION, "trace_semantics": "EXACT D1 saved paths and Gaussian reconstruction",
        "diagnostic_boundary": "NeRF volume samples are a diagnostic single-ray retrace, not stored D1 internal samples",
        "pixel": raw["pixel"], "why": why,
        "primary": {"pixel": raw["pixel"], "sample_coordinate": [raw["pixel"][0] + .5, raw["pixel"][1] + .5],
                    "origin": raw["primary_ray"]["origin"], "direction": raw["primary_ray"]["direction"],
                    "hit": raw["primary_ray"]["hit"], "t_hit": raw["primary_ray"].get("t_hit")},
        "surface": ({k: raw["primary_ray"][k] for k in ("world_position", "emitter_local_position", "triangle_id",
                     "barycentric", "uv", "geometric_normal", "shading_normal")} if raw["primary_ray"]["hit"] else None),
        "material": raw["material"], "paths": paths, "component_attribution": components,
        "contribution_summary": {
            "total_reconstructed": reconstructed, "background_direct": background, "path_total": path_total,
            "diffuse": diffuse, "specular": specular, "bsdf_sampled": source_bsdf, "vmf_sampled": source_vmf,
            "visible": visible, "occluded_zeroed": occluded, "attributed_sum": attributed,
            "attribution_residual": residual, "lobe_residual": lobe_residual,
            "top_fractions": top, "paths_needed": needed, "path_count": len(paths),
        },
        "reconstruction": {"recorded": raw["recorded_final_rgb"], "reconstructed": reconstructed,
                           "absolute_error": [abs(x) for x in sub(raw["recorded_final_rgb"], reconstructed)],
                           "max_abs": raw["reconstruction_max_abs"], "filter": raw["final_renderer"]["filter"]},
        "nerf_diagnostic": {**raw["nerf_selected_ray"], "display_label": "DIAGNOSTIC NeRF RETRACE — not stored final D1 volume sequence",
                            "path_id": (f"{raw['nerf_selected_sample']['primary_ray_id']}:{raw['nerf_selected_sample']['sample_id']}" if raw["nerf_selected_sample"] else None),
                            "stored_selected_path_Li": raw["nerf_selected_sample"]["Li"] if raw["nerf_selected_sample"] else raw["nerf_selected_ray"]["final_Li"]},
        "causal_graph": dag, "raw_trace_sha256": _sha(raw),
        "advanced": {"final_renderer": raw["final_renderer"], "vMF": raw["vMF"], "raw_trace_available": True},
    }


def load_and_build(raw_path: Path, gmm_path: Path) -> dict:
    return build_pixel_trace(json.loads(raw_path.read_text()), json.loads(gmm_path.read_text()))
