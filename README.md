<p align="center">
  <img src="demo.png" alt="NeRF LightPath Explorer interactive causal transport interface" width="100%">
</p>

# NeRF LightPath Explorer

An interactive causal light-transport explorer for **NeRF-Emitter / physics-based inverse rendering (PBIR)**. Instead of exposing renderer internals as raw JSON, the interface answers a more useful question: **why does this pixel have this color?**

The explorer follows the actual rendering chain:

```text
Camera → Primary ray → Surface → Material → Sampling → wi
       → Visibility → Environment NeRF Li → BSDF → MIS
       → Path contribution → Pixel RGB
```

## Interactive design

The UI is organized around five investigation questions:

| View | What it explains |
|---|---|
| **WHY?** | Pixel classification, recovered material, contribution decomposition, Top-N paths and exact reconstruction |
| **WHERE?** | Interactive WebGL path view, visible/occluded rays, vMF directions and GMM bright-region linkage |
| **HOW?** | A selected path's sampling → visibility → Li → BSDF → MIS → contribution calculation, with real numbers |
| **INSIDE NeRF** | A clearly labelled diagnostic volume retrace with sigma, transmittance, weights and RGB accumulation |
| **WHAT IF?** | Frozen-path material interventions, deterministic full-pixel rerenders and attribution-only interventions |

Selection is bidirectional: clicking a virtualized table row highlights the exact 3D ray, while clicking a ray selects the corresponding path and updates every dependent panel. Pixel, path, mode and tab are encoded in the URL for reproducible investigations.

## Try the bundled explorer

The repository includes compact exact traces for Background, Diffuse, Highlight and Silhouette presets, so the main explanatory workflow runs without a GPU:

```bash
python3 reimpl/d3_causal_explorer/server.py --port 8766
```

Open [http://127.0.0.1:8766/](http://127.0.0.1:8766/), then choose a preset or use a URL such as:

```text
http://127.0.0.1:8766/?pixel=151,109&tab=how&path=464223:3
```

Material recomputation for arbitrary parameters requires the official renderer assets, checkpoint and a CUDA-capable environment. Optional remote recomputation is configured without hard-coded infrastructure:

```bash
python3 reimpl/d3_causal_explorer/server.py \
  --remote <ssh-host> \
  --remote-root <remote-project-root> \
  --remote-python <remote-python>
```

The same settings can be provided through `NERF_EXPLORER_REMOTE`, `NERF_EXPLORER_REMOTE_ROOT`, `NERF_EXPLORER_REMOTE_PYTHON`, and `NERF_EXPLORER_REMOTE_ENV`.

## What was implemented

- Explicit teapot geometry takeover with environment-only NeRF radiance queries.
- Official-compatible Mitsuba 3.4.1 principled BSDF/PDF behavior.
- Official-style BSDF + vMF one-sample MIS transport.
- Exact Gaussian-filtered pixel attribution down to individual saved paths.
- Diffuse/specular, BSDF/vMF, visible/occluded and per-component decomposition.
- WebGL rendering of the teapot wireframe, primary ray, normals and thousands of secondary rays.
- Frozen-path `do(material)` interventions and deterministic single-pixel full rerenders.
- Stable `d3.1` trace schema and automated closure/reproducibility checks.
- Presentation mode and exportable investigation bundles.

## Verified result

For the bundled highlight pixel `(151, 109)`:

- 4,304 exact secondary paths;
- `68.97%` specular and `31.03%` diffuse energy;
- pixel reconstruction max error: `3.8147e-6`;
- 50/90/99% cumulative energy requires 102/567/1,486 paths.

Across 26 validated teapot pixels, the maximum attribution residual is `4.8441e-7`; diffuse + specular closure is `2.1250e-8`. Fifty randomly selected UI paths match backend `wi`, `Li`, PDFs, MIS and contribution with zero serialization error. Frozen-path invariants pass, and repeated full rerenders are deterministic (`max_abs = 0`).

Validation artifacts are under [`results/d3_causal_explorer`](results/d3_causal_explorer), with methodology in [`reports/d3_causal_light_transport_explorer.md`](reports/d3_causal_light_transport_explorer.md) and the schema definition in [`reports/d3_trace_schema.md`](reports/d3_trace_schema.md).

## Data semantics

All contribution ranking and reconstruction values come from saved exact D1 paths or deterministic exact retracing. The displayed estimator matches the implementation:

```text
C_sample = V × Li × (f·cosθ) × w_balance / (q × p_selected)
C_pixel  = C_sample × primary_filter_weight / Σ(filter_weight) / secondary_spp
```

The NeRF volume inspector is intentionally marked **DIAGNOSTIC**: its sample sequence explains one deterministic retrace but is not presented as the stored internal sample sequence of the final whole-image render. D/F/G values are likewise labelled microfacet diagnostics rather than a fabricated complete decomposition of Disney Principled.

## Repository layout

```text
reimpl/d3_causal_explorer/   Browser UI, stable schema and server
reimpl/hybrid_teapot/        Geometry, NeRF query, guiding and BSDF integration
reimpl/run_d2_*.py           Exact trace generation and baseline acceptance
reimpl/run_d3_*.py           Attribution, interventions and D3 validation
results/                     Bundled presets and compact validation artifacts
reports/                     Technical report and d3.1 schema documentation
demo.png                     Interface overview
```

The official `gerwang/nerf-emitter` source, scene assets and model checkpoints are not redistributed. Full regeneration expects them under the layout referenced by the scripts.

## Acknowledgements

This project builds on [gerwang/nerf-emitter](https://github.com/gerwang/nerf-emitter), Mitsuba 3, Nerfstudio and the NeRF ecosystem. Please follow the original projects' licenses and citation requirements when reproducing the renderer.

---

**Project period:** September 2023 – January 2024. The completed research prototype was later organized for public release on GitHub.
