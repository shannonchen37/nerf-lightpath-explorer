"""Export a compact wireframe of the reference teapot for the D3 WebGL view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path.home() / "nerf_inverse_rendering"
MESH = ROOT / "third_party/nerf-emitter/results/synthetic/teapot-unirough-0.2_bedroom_v2/sdf-nerfacto/v1/mesh.obj"
OUT = ROOT / "reimpl/d3_causal_explorer/assets/teapot_edges.json"
MAX_EDGES = 6000


def main() -> None:
    mesh = trimesh.load(MESH, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("reference mesh.obj did not load as one Trimesh")
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) > MAX_EDGES:
        # Deterministic coverage over the full list; this is display geometry only.
        edges = edges[np.linspace(0, len(edges) - 1, MAX_EDGES, dtype=np.int64)]
    lines = np.asarray(mesh.vertices, dtype=np.float32)[edges].tolist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(lines, separators=(",", ":")))
    print(json.dumps({"mesh": str(MESH), "unique_edges": int(len(mesh.edges_unique)), "exported_edges": len(lines), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
