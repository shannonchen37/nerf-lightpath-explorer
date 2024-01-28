# D3 Causal Light Transport Explorer Report

## Verdict

**D3 CAUSAL LIGHT TRANSPORT EXPLORER: PASS**

The D3 layer changes D2 from a trace/table inspector into a causal explorer organized by user questions: WHY, WHERE, HOW, INSIDE NeRF and WHAT IF. It consumes exact D1/D2 path data, adds exact contribution attribution and two controlled intervention modes, and leaves the official tracked source and D1 estimator untouched.

## Validation summary

| Check | Result | Evidence |
|---|---:|---|
| D1 baseline | PASS | Frozen prior correctness baseline |
| D2 four-class acceptance after trace extension | PASS | `results/d2_pixel_inspector/acceptance.json` |
| D3 schema/acceptance | PASS | 32 traces; all four presets and three extra cases present |
| Attribution closure | PASS | 26 teapot pixels; max residual `4.8441e-7` (< `1e-5`) |
| Diffuse + specular closure | PASS | max residual `2.1250e-8` |
| UI/backend selected-path consistency | PASS | 50 paths, max error `0.0` |
| Frozen-path intervention | PASS | same `wi`, Li and visibility hashes; decomposition residual `1.1921e-7` |
| Full-rerender reproducibility | PASS | two repeats, max RGB difference `0.0` |
| Official tracked source | PASS | commit `8b927077a402515d4be48d747e2cae5f7b06e127`; `git diff --quiet` exit `0` |
| Chrome end-to-end | PASS | WHY/WHERE/HOW/NeRF/WHAT IF/background/presentation and live GPU1 interventions exercised |

The official repository contains an existing untracked `scenes/` directory; this is not a tracked source modification.

## Highlight example `(151,109)`

The page derives the following from its exact trace rather than hard-coding values:

- 4,304 secondary paths; reconstructed RGB `(6.199532, 5.869241, 2.306535)`;
- specular/diffuse fractions `68.9698% / 31.0302%`;
- dominant attributed bright region: GMM component `#11`;
- Top 1/5/10/12/20/50 contributions: `1.179% / 5.184% / 9.195% / 10.633% / 16.009% / 31.630%`;
- paths needed for 50/80/90/95/99%: `102 / 326 / 567 / 819 / 1486`;
- final recorded-versus-reconstructed max error `3.8147e-6`.

The selected top path `464223:3` is shown as vMF/emitter component `#37`, visible, with its exact `wi`, Li, PDFs, MIS weight, diffuse/specular response and pixel contribution. The expanded formula reproduces the stored sample contribution numerically.

## Intervention result

For `do(roughness=0.6)` at `(151,109)`:

- Frozen paths: original RGB `(6.199532, 5.869241, 2.306535)` → `(3.392422, 3.149653, 0.803173)`; `wi`, Li and visibility hashes are identical.
- Full rerender: RGB `(2.645401, 2.441569, 0.613748)`; two deterministic repeats are identical (`max_abs=0`).

Frozen mode changes material, BSDF, dependent MIS, contribution and pixel RGB while holding camera, primary ray, geometry, surface, `wi`, visibility, NeRF and Li fixed. Full mode additionally changes sampler, `wi`, visibility, NeRF Li and downstream PDFs/MIS while retaining camera, primary ray, geometry and surface.

## Required questions

1. **D3 compared with D2:** D3 explains causal dependence and supports controlled recomputation; D2 only exposed exact traces and path rows.
2. **Can a highlight pixel answer why it is bright?** Yes. It reports the shading regime, material state, diffuse/specular split, dominant GMM bright region, top contributors and exact reconstruction.
3. **Can it quantify Top N?** Yes. Top fractions, cumulative plot, threshold counts and Top-10/50/All partial reconstruction are calculated from exact pixel contributions.
4. **Can it identify vMF/GMM directions?** Yes. The spherical direction view overlays exact BSDF/vMF samples, component centers/weights and GMM spatial centroids; component selection filters paths and highlights the 3D marker.
5. **Does a selected path show the full chain?** Yes: sampling → visibility → Li → official BSDF response → MIS → sample/pixel contribution, including exact numeric substitution.
6. **Can it enter NeRF diagnostic volume trace?** Yes. It displays all 48 diagnostic samples, curves, click-selected sample position and Li closure, with a prominent non-exact-internal-trace warning.
7. **Can it prove pixel = sum(paths)?** Yes. Gaussian-weighted attribution closes over 26 teapot pixels with max residual `4.8441e-7`; background-direct is included separately.
8. **Did frozen roughness intervention succeed?** Yes. Fixed-data hashes match and the recomputed decomposition closes.
9. **Did full rerender succeed?** Yes. The selected pixel rerenders on GPU1 with deterministic seeds and repeat error `0.0`.
10. **Which causal nodes change?** Frozen: material/BSDF/MIS/contribution/pixel change; camera/geometry/surface/wi/visibility/NeRF/Li stay fixed. Full: material/sampler/wi/visibility/NeRF/Li/BSDF/MIS/contribution/pixel change; camera/primary ray/geometry/surface stay fixed.
11. **Do Background/Diffuse/Highlight/Silhouette pass?** Yes. Their reconstruction errors are respectively `5.9605e-8`, `3.5763e-7`, `3.8147e-6`, and `3.8743e-7`. Background adapts to a reduced camera → NeRF → Li → pixel DAG and hides nonexistent material interventions.
12. **Viewer address:** `http://127.0.0.1:8766/`.

## UI and honesty boundaries

The 3D view uses WebGL, a deterministic 6,000-edge display wireframe of the official mesh, contribution-mapped opacity, orbit/pan/zoom, top-N/isolate controls and CPU ray picking for table ↔ 3D linkage. The path table is virtualized. URL state preserves pixel/path/mode/tab. Investigation export emits trace/selected-path/summary/intervention JSON plus canvas PNGs and a standalone HTML report.

No light-source semantic names are inferred: the UI says only “bright environment region” and component/direction IDs. D/F/G are labelled microfacet diagnostics. The NeRF sample view is labelled diagnostic and never represented as the saved final D1 internal sequence.
