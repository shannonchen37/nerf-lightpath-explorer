# Public Narrative and Project Provenance Audit

## Conclusion

The public narrative now identifies this repository as an independent reconstruction, organization, validation and interactive extension of an earlier research prototype developed during September 2023 – January 2024. The public `gerwang/nerf-emitter` repository is consistently described as an external reference implementation and numerical/visual parity target, not a project codebase, fork base, upstream implementation or code foundation.

## Wording changes

| File | Previous public wording | Revised wording | Reason |
|---|---|---|---|
| `README.md` | Interactive explorer “for NeRF-Emitter” and “builds on” framing | Independent reconstruction and extension; explicit non-fork statement | Establish the project identity before describing the reference target |
| `README.md` | Reference implementation and software dependencies grouped together | Dedicated Project origin, Contributions, Reference validation, Dependencies and Acknowledgements sections | Distinguish the parity target from actual software dependencies |
| `reimpl/__init__.py` | “built around the official NeRF-Emitter baseline” | “Independent reconstruction of decomposed NeRF–surface PBIR transport” | Remove inaccurate code-foundation framing |
| `reimpl/nerf_emitter_query.py` | Query API presented primarily as an official-checkpoint wrapper | Independent HDR query with optional public-reference parity validation | Make the reconstructed pipeline the subject and validation the context |
| `reimpl/hybrid_teapot/official_guiding.py` | Public module/class name centered on “official” | New `reference_guiding.py` and `ReferenceVmfMixture`; compatibility wrapper retained | Improve the public API without breaking historical imports |
| `reimpl/hybrid_teapot/*.py` and D1/D3 runners | “official-style”, “official-compatible” and “official route” wording | “reference-compatible”, “parity-validated” and “reference benchmark” wording | Describe specific comparison semantics without implying ownership or ancestry |
| `reimpl/d3_causal_explorer/index.html` | “official-style MIS” | “reference-validated MIS” | Correct the user-visible interface label |
| `reports/d3_causal_light_transport_explorer.md` | “official tracked source” and “official repo untouched” | External reference implementation kept unmodified during parity validation | Keep reproducibility evidence while making workspace boundaries clear |
| `reports/d3_trace_schema.md` | “official Mitsuba principled” | “parity-validated Mitsuba principled” | State why the response is mentioned |

## Naming classification

Changed public API names:

- `official_guiding.py` now delegates to `reference_guiding.py`.
- `OfficialVmfMixture` is superseded by `ReferenceVmfMixture`.
- `from_reference_checkpoint` and `load_reference_nerf_environment` are the preferred query APIs.

Retained compatibility and provenance names:

- `official_guiding.py`, `OfficialVmfMixture`, `from_official_checkpoint` and `load_official_nerf_emitter` remain backward-compatible aliases.
- `official_repo_root`, the `OFFICIAL` local constant, diagnostic filenames, historical result directory names, test identifiers and frozen artifact filenames remain where changing them could break scripts, APIs or trace provenance.
- Reference-specific stored metadata remains unchanged inside frozen experiment artifacts.
- The historical top-level directory `reimpl/` remains because a repository-wide move would break script paths and historical documentation; the README defines its independent-reconstruction meaning.

These retained identifiers refer to an external parity fixture or preserve compatibility. They do not describe the project's ancestry.

## Integrity checks

- Renderer equations, sampling behavior, checkpoint handling and experiment execution paths were not changed.
- Numerical JSON, trace payloads, acceptance artifacts, images and benchmark outputs were not edited.
- `demo.png` and all files under `results/` were hash-compared before and after the cleanup.
- Existing commit IDs were preserved; this cleanup is an ordinary new commit.
