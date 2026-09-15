# D3 Trace Schema (`d3.1`)

## Scope and provenance

`d3.1` is an explanation schema built from saved D1/D2 exact paths. It does not alter the D1 estimator. Exact attribution uses the same primary/path seeds and the same Gaussian reconstruction weights as D1. The source D2 file hash is stored as `raw_trace_sha256`.

The `nerf_diagnostic` branch is intentionally separate: it is a deterministic single-ray, 48-sample diagnostic retrace and is **not** the stored whole-ray internal sample sequence from the final D1 render.

## Top-level `PixelTrace`

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Always `d3.1` |
| `trace_semantics` | string | Exact D1 path/reconstruction provenance |
| `diagnostic_boundary` | string | Explicit exact-versus-diagnostic warning |
| `pixel` | `[int,int]` | Image coordinate `(x,y)` |
| `why` | object | Derived, non-hard-coded causal summary |
| `primary` | `PrimaryRay` | Center-ray semantic summary |
| `surface` | `SurfaceHit?` | Null for background pixels |
| `material` | `MaterialState?` | Null for background pixels |
| `paths` | `SecondaryPath[]` | Exact saved secondary paths, contribution-sorted |
| `component_attribution` | array | Exact per-vMF/GMM contribution aggregation |
| `contribution_summary` | object | RGB decompositions, top fractions and thresholds |
| `reconstruction` | object | Recorded/reconstructed RGB and error |
| `nerf_diagnostic` | `NeRFDiagnosticTrace` | Clearly labelled diagnostic volume trace |
| `causal_graph` | object | Pixel-specific DAG and intervention dependencies |
| `advanced` | object | Renderer/vMF metadata for debug mode |

## `PrimaryRay`, `SurfaceHit`, and `MaterialState`

`PrimaryRay` contains pixel, sample coordinate, camera origin/direction, hit flag and `t_hit`. `SurfaceHit` contains world/emitter-local position, triangle ID, barycentric coordinates, UV, geometric normal and shading normal. `MaterialState` contains exact sampled base color, roughness, specular, F0, IOR, metallic and BSDF type.

For a miss, `surface` and `material` are null and the causal DAG reduces to `camera → primary_ray → nerf → Li → pixel_rgb`.

## `SecondaryPath`

Each path has a stable ID `<primary_ray_id>:<sample_id>` and stores:

- sampling identity: source, component, `vmf_mu`, `vmf_kappa`;
- geometry: origin, `wi`, `cos_theta`, `NoV`;
- visibility: `visible`, `occluded`;
- environment: exact saved `Li` and its luminance;
- response: full parity-validated Mitsuba principled `bsdf_f_cos`, plus diffuse/specular values whose sum closes to the full response;
- sampling/MIS: `bsdf_pdf`, `emitter_pdf`, selected proposal PDF, source-selection probability and balance weight;
- estimator values: sample contribution, Gaussian-filtered pixel contribution, luminance, energy fraction, rank and cumulative fraction;
- `microfacet_diagnostic`: D/F/G, explicitly labelled diagnostic and not presented as a complete principled decomposition.

The exact displayed one-sample estimator is:

```text
C_sample = V × Li × (f·cosθ) × w_balance / (q × p_selected)
C_pixel  = C_sample × primary_filter_weight / Σ(primary_filter_weight) / secondary_spp
```

## `ContributionSummary`

The schema provides RGB totals for background-direct, all secondary paths, diffuse, specular, BSDF-sampled, vMF/emitter-sampled, visible and occluded-zeroed paths. It also stores the attributed sum, attribution residual, lobe residual, Top 1/5/10/12/20/50/100/200 fractions, and path counts needed for 50/80/90/95/99% cumulative luminance.

## `NeRFDiagnosticTrace`

The diagnostic trace contains `path_id`, external origin/direction, sample `t`, positions, sigma, alpha, transmittance, weights, RGB, per-sample RGB contribution, summed weighted RGB, diagnostic `final_Li`, and the exact saved selected-path Li for comparison. `display_label` and `diagnostic_boundary` prevent it from being confused with an exact internal final-render trace.

## `InterventionResult`

Two material modes share schema version `d3.1`:

- `frozen_path`: holds camera, geometry, surface, all `wi`, `Li`, visibility and sample identities fixed; recomputes material response, dependent BSDF PDF/MIS, contribution and pixel RGB.
- `full_pixel_rerender`: holds camera/geometry/surface and deterministic seeds fixed; resamples proposal directions, visibility and NeRF Li before recomputing downstream values.

Attribution-only component removal and hypothetical `V=0` are UI calculations and are explicitly not claimed to edit the NeRF scene.
