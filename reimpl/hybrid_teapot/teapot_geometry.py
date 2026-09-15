"""Explicit reference-teapot intersection and texture lookup via Open3D BVH."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image


@dataclass(frozen=True)
class TeapotHit:
    hit: torch.Tensor
    t: torch.Tensor
    point: torch.Tensor
    normal: torch.Tensor
    uv: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    primitive_id: torch.Tensor


def _load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(image)


def _load_scalar(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(image)


def _bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    u = np.mod(uv[:, 0], 1.0) * (width - 1)
    v = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (height - 1)
    x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = u - x0, v - y0
    if texture.ndim == 3:
        wx, wy = wx[:, None], wy[:, None]
    result = (
        texture[y0, x0] * (1 - wx) * (1 - wy)
        + texture[y0, x1] * wx * (1 - wy)
        + texture[y1, x0] * (1 - wx) * wy
        + texture[y1, x1] * wx * wy
    )
    return result.astype(np.float32)


class ExplicitTeapotGeometry:
    """CPU BVH wrapper; inputs/outputs are torch tensors on the caller device."""

    def __init__(self, mesh_path: Path, reflectance_path: Path, roughness_path: Path) -> None:
        mesh = trimesh.load(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("expected one Trimesh")
        if mesh.visual.uv is None:
            raise RuntimeError("reference teapot mesh has no UV coordinates")
        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.faces = np.asarray(mesh.faces, dtype=np.uint32)
        self.vertex_uv = np.asarray(mesh.visual.uv, dtype=np.float32)
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        self.reflectance = _load_rgb(reflectance_path)
        self.roughness_texture = _load_scalar(roughness_path)
        tensor_mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.faces)
        )
        self.scene = o3d.t.geometry.RaycastingScene()
        self.geometry_id = self.scene.add_triangles(tensor_mesh)
        self.metadata = {
            "mesh_path": str(mesh_path), "reflectance_path": str(reflectance_path),
            "roughness_path": str(roughness_path), "vertices": len(self.vertices),
            "faces": len(self.faces), "bounds": np.stack((self.vertices.min(0), self.vertices.max(0))).tolist(),
            "reflectance_resolution": list(self.reflectance.shape[:2]),
            "roughness_resolution": list(self.roughness_texture.shape[:2]),
        }

    def intersect(self, origins: torch.Tensor, directions: torch.Tensor) -> TeapotHit:
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[-1] != 3:
            raise ValueError("origins/directions must have identical [N,3] shape")
        device = origins.device
        rays = torch.cat((origins, directions), dim=-1).detach().cpu().float().numpy()
        answer = self.scene.cast_rays(o3d.core.Tensor(rays))
        t_np = answer["t_hit"].numpy().astype(np.float32)
        primitive_np = answer["primitive_ids"].numpy().astype(np.int64)
        bary_uv = answer["primitive_uvs"].numpy().astype(np.float32)
        hit_np = np.isfinite(t_np)
        safe_primitive = np.where(hit_np, primitive_np, 0)
        faces = self.faces[safe_primitive]
        bary = np.stack((1.0 - bary_uv[:, 0] - bary_uv[:, 1], bary_uv[:, 0], bary_uv[:, 1]), axis=-1)
        uv_np = (self.vertex_uv[faces] * bary[..., None]).sum(axis=1)
        normal_np = (self.vertex_normals[faces] * bary[..., None]).sum(axis=1)
        normal_np /= np.maximum(np.linalg.norm(normal_np, axis=-1, keepdims=True), 1.0e-12)
        # Ensure the geometric shading normal faces the incident camera ray.
        flip = (normal_np * rays[:, 3:]).sum(axis=-1) > 0
        normal_np[flip] *= -1
        albedo_np = _bilinear(self.reflectance, uv_np)
        roughness_np = _bilinear(self.roughness_texture, uv_np)
        albedo_np[~hit_np] = 0
        roughness_np[~hit_np] = 0
        uv_np[~hit_np] = 0
        normal_np[~hit_np] = 0
        t_safe = np.where(hit_np, t_np, np.inf).astype(np.float32)
        t = torch.from_numpy(t_safe).to(device)
        hit = torch.from_numpy(hit_np).to(device)
        point = origins + torch.where(hit, t, torch.zeros_like(t))[:, None] * directions
        point = torch.where(hit[:, None], point, torch.zeros_like(point))
        return TeapotHit(
            hit=hit, t=t, point=point,
            normal=torch.from_numpy(normal_np).to(device),
            uv=torch.from_numpy(uv_np).to(device),
            albedo=torch.from_numpy(albedo_np).to(device),
            roughness=torch.from_numpy(roughness_np).to(device),
            primitive_id=torch.from_numpy(np.where(hit_np, primitive_np, -1)).to(device),
        )

    def occluded(self, origins: torch.Tensor, directions: torch.Tensor, epsilon: float = 1.0e-4) -> torch.Tensor:
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[-1] != 3:
            raise ValueError("origins/directions must have identical [N,3] shape")
        rays = torch.cat((origins, directions), dim=-1).detach().cpu().float().numpy()
        t = self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        return torch.from_numpy(np.isfinite(t) & (t > epsilon)).to(origins.device)
