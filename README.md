<p align="center">
  <img src="demo.png" alt="NeRF LightPath Explorer interactive causal transport interface" width="100%">
</p>

# NeRF LightPath Explorer

An independent reconstruction and interactive extension of earlier research work on decomposed NeRF–surface light transport for physics-based inverse rendering (PBIR). **This repository is not a fork of NeRF-Emitter.** The later public [`gerwang/nerf-emitter`](https://github.com/gerwang/nerf-emitter) implementation is used here as an external reference for numerical and visual parity validation.

## Project origin

This repository reconstructs and extends a research prototype I previously worked on during September 2023 – January 2024. The current public repository independently organizes that work into a validated implementation and adds a causal, interactive light-transport analysis layer. The completed prototype was later packaged for public release; the public NeRF-Emitter repository serves as a reference implementation and parity target, not as this project's codebase or fork base.

## What problem this solves

Inverse-rendering pipelines can produce a pixel without making its physical causes inspectable. NeRF LightPath Explorer connects the final RGB value back to the camera ray, explicit surface, recovered material, sampled incident directions, visibility, environment radiance, BSDF response and MIS estimator. It supports observation, attribution and controlled intervention using the renderer's actual saved paths.

## Core pipeline

```text
Captured HDR scene
        ↓
     Full NeRF
        ↓  remove object-region density
Environment-only NeRF ───────────────┐
                                     │ environment radiance Li
Explicit editable geometry/material ├→ PBIR light transport
                                     │
                                     └→ exact pixel/path trace
                                                ↓
                                   causal interactive explorer
```

The NeRF represents environment radiance. The explicit object represents editable geometry and material. Light transport between them is physically evaluated by the PBIR renderer.

## Interactive causal explorer

The interface is organized around five investigation questions:

| View | What it explains |
|---|---|
| **WHY?** | Pixel classification, recovered material, contribution decomposition, Top-N paths and exact reconstruction |
| **WHERE?** | Interactive WebGL paths, visibility, vMF directions and GMM bright-region linkage |
| **HOW?** | A selected path's sampling → visibility → Li → BSDF → MIS → contribution calculation with real values |
| **INSIDE NeRF** | A clearly labelled diagnostic volume retrace with density, transmittance, weights and RGB accumulation |
| **WHAT IF?** | Frozen-path material interventions, deterministic full-pixel rerenders and attribution-only interventions |

Selection is bidirectional: a virtualized table row highlights the exact 3D ray, and selecting a ray updates its table row and every dependent panel. Pixel, path, mode and tab are encoded in the URL for reproducible investigations. Presentation mode and exportable investigation bundles support demos and supplementary material.

## What this repository contributes

- Independent reconstruction of a decomposed NeRF-environment + explicit-surface PBIR pipeline.
- Explicit environment-only NeRF querying after object-region density removal.
- Exact Gaussian-filtered pixel and individual light-path attribution.
- Causal decomposition of pixel formation into material, proposal, visibility, radiance, BSDF and MIS terms.
- Bidirectional virtualized path-table and WebGL 3D inspection.
- Diffuse/specular, BSDF/vMF, visible/occluded and per-component contribution analysis.
- Frozen-path material interventions that isolate downstream response.
- Deterministic single-pixel full-rerender interventions.
- The interactive WHY / WHERE / HOW / INSIDE NeRF / WHAT IF workflow.

Standard components such as NeRF, Mitsuba, PyTorch, GGX and vMF distributions are dependencies or established techniques, not claimed as original contributions.

## Reference validation

The independently reconstructed renderer was validated against the public [`gerwang/nerf-emitter`](https://github.com/gerwang/nerf-emitter) reference implementation using:

- the public teapot benchmark and reference checkpoint;
- geometry-hit and UV parity;
- environment-query and visibility parity;
- Mitsuba Principled BSDF/PDF parity;
- image-level numerical and visual parity;
- vMF proposal and one-sample MIS behavior.

During parity testing, the external reference implementation was kept unmodified. Its source is not vendored into this repository.

## Try the explorer

Compact exact traces for Background, Diffuse, Highlight and Silhouette pixels are bundled, so the explanatory workflow runs without a GPU:

```bash
python3 reimpl/d3_causal_explorer/server.py --port 8766
```

Open [http://127.0.0.1:8766/](http://127.0.0.1:8766/) or restore a shared investigation directly:

```text
http://127.0.0.1:8766/?pixel=151,109&tab=how&path=464223:3
```

The historical directory name `reimpl/` contains the independently reconstructed PBIR components used by this repository. It is retained to preserve script paths and validation provenance; it does not indicate that this repository is a downstream fork.

Arbitrary material recomputation requires the reference benchmark assets, checkpoint and a CUDA environment. Optional remote execution is configured without embedded infrastructure:

```bash
python3 reimpl/d3_causal_explorer/server.py \
  --remote <ssh-host> \
  --remote-root <remote-project-root> \
  --remote-python <remote-python>
```

Equivalent environment variables are `NERF_EXPLORER_REMOTE`, `NERF_EXPLORER_REMOTE_ROOT`, `NERF_EXPLORER_REMOTE_PYTHON` and `NERF_EXPLORER_REMOTE_ENV`.

## Verified results

For the bundled highlight pixel `(151, 109)`:

- 4,304 exact secondary paths;
- `68.97%` specular and `31.03%` diffuse energy;
- pixel reconstruction max error: `3.8147e-6`;
- 50/90/99% cumulative energy requires 102/567/1,486 paths.

Across 26 validated teapot pixels, the maximum attribution residual is `4.8441e-7`; diffuse + specular closure is `2.1250e-8`. Fifty selected UI paths match backend `wi`, `Li`, PDFs, MIS and contribution with zero serialization error. Frozen-path invariants pass, and repeated full rerenders are deterministic (`max_abs = 0`). These results validate the reconstructed implementation independently against the public parity target.

Validation artifacts are under [`results/d3_causal_explorer`](results/d3_causal_explorer), with methodology in [`reports/d3_causal_light_transport_explorer.md`](reports/d3_causal_light_transport_explorer.md) and the schema in [`reports/d3_trace_schema.md`](reports/d3_trace_schema.md).

## Data semantics and honesty boundaries

Contribution ranking and reconstruction come from saved exact paths or deterministic exact retracing:

```text
C_sample = V × Li × (f·cosθ) × w_balance / (q × p_selected)
C_pixel  = C_sample × primary_filter_weight / Σ(filter_weight) / secondary_spp
```

The NeRF volume inspector is marked **DIAGNOSTIC**: it explains one deterministic retrace but is not presented as the stored internal sample sequence of the final whole-image render. D/F/G values are similarly labelled microfacet diagnostics rather than a fabricated complete decomposition of Disney Principled.

## Repository layout

```text
reimpl/d3_causal_explorer/   Browser UI, stable schema and server
reimpl/hybrid_teapot/        Reconstructed geometry, radiance query, guiding and BSDF integration
reimpl/run_d2_*.py           Exact tracing and acceptance checks
reimpl/run_d3_*.py           Attribution, interventions and D3 validation
results/                     Bundled presets and compact validation artifacts
reports/                     Technical, schema and public-narrative audits
demo.png                     Interface overview
```

## Dependencies

Mitsuba 3, Nerfstudio, PyTorch, Open3D, Trimesh, NumPy and Pillow are actual software dependencies where noted. Each remains subject to its own license. The public NeRF-Emitter repository has the distinct role of external validation reference; it is not treated as a dependency codebase and is not redistributed here.

## Reference implementation and acknowledgements

The public [`gerwang/nerf-emitter`](https://github.com/gerwang/nerf-emitter) repository is acknowledged as the external implementation used to reproduce the teapot benchmark and check numerical and visual parity. This repository does not claim authorship of NeRF-Emitter and does not vendor its source, scene assets or checkpoints. Users performing parity validation should follow that project's license and citation requirements, as well as those of the software dependencies listed above.
