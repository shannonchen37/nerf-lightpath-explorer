"""Standalone whole-ray HDR query API for an official NeRF-Emitter checkpoint.

This module deliberately does not construct the official Mitsuba pipeline or a
trainer. It reconstructs only the data-dependent Nerfacto model, loads its state,
and reproduces the ray conversion performed by ``emitters/nerf_emitter_op.py``.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import torch

TensorLikeIndex = Union[int, torch.Tensor]
TensorLikeNear = Union[float, torch.Tensor]


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _bootstrap_official_imports(official_repo_root: Path) -> None:
    for path in (official_repo_root, official_repo_root / "differentiable-sdf-rendering" / "python"):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


class StandaloneNerfEmitter:
    """A forward-only wrapper around the official whole-ray SdfNerfacto model.

    External convention:
        Mitsuba emitter-local coordinates. Positions use the unit-cube convention
        consumed by the official NeRF emitter after applying its inverse
        ``to_world`` transform. Directions are emitter-local vectors. ``near`` is
        expressed in the same ray parameterization and shifts the origin before
        conversion.

    Internal convention:
        Nerfstudio/OpenGL training coordinates, after scene scaling and the
        official ``mi2gl_left`` axis permutation.
    """

    _MIN_STABLE_TCNN_CHUNK = 128

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: Any,
        dataparser_outputs: Any,
        rotater: Optional[torch.nn.Module],
        scene_scale: float,
        device: torch.device,
        checkpoint_path: Path,
        checkpoint_step: int,
        official_repo_root: Path,
    ) -> None:
        self.model = model
        self.config = config
        self.dataparser_outputs = dataparser_outputs
        self.rotater = rotater
        self.scene_scale = float(scene_scale)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_step = checkpoint_step
        self.official_repo_root = official_repo_root
        self._mi2gl_left = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        self._gl2mi_left = torch.linalg.inv(self._mi2gl_left)

    @classmethod
    def from_official_checkpoint(
        cls,
        config_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cuda",
        official_repo_root: Optional[Union[str, Path]] = None,
    ) -> "StandaloneNerfEmitter":
        """Reconstruct and load only the official query-time Nerfacto model."""
        config_path = Path(config_path).expanduser().resolve()
        if official_repo_root is None:
            official_repo_root = config_path.parents[5]
        official_repo_root = Path(official_repo_root).expanduser().resolve()
        _bootstrap_official_imports(official_repo_root)

        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        with _working_directory(official_repo_root):
            import mitsuba as mi
            import yaml
            from nerfstudio.engine.trainer import TrainerConfig
            from nerfstudio.field_components.rotater import Rotater

            if mi.variant() is None:
                mi.set_variant("cuda_ad_rgb" if requested_device.type == "cuda" else "llvm_ad_rgb")
            config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
            if not isinstance(config, TrainerConfig):
                raise TypeError(f"Expected TrainerConfig, got {type(config)!r}")

            parser = config.pipeline.datamanager.dataparser.setup()
            dataparser_outputs = parser.get_dataparser_outputs(split="train")
            metadata = dataparser_outputs.metadata
            rotater = None
            if "rotations" in metadata:
                rotater = Rotater(
                    metadata["rotations"],
                    dataparser_outputs.dataparser_scale,
                    transform_matrices=metadata.get("rotation_transform_matrices"),
                    rotation_aabb=metadata.get("rotation_aabb"),
                )

            grad_scaler = torch.cuda.amp.GradScaler(enabled=bool(config.mixed_precision))
            model = config.pipeline.model.setup(
                scene_box=dataparser_outputs.scene_box,
                num_train_data=len(dataparser_outputs.image_filenames),
                metadata=metadata,
                device=requested_device,
                grad_scaler=grad_scaler,
                mixed_precision=bool(config.mixed_precision),
                rotater=rotater,
            ).to(requested_device)

            if checkpoint_path is None:
                checkpoint_directory = config_path.parent / "nerfstudio_models"
                checkpoints = sorted(checkpoint_directory.glob("step-*.ckpt"))
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoint in {checkpoint_directory}")
                checkpoint_path = checkpoints[-1]
            checkpoint_path = Path(checkpoint_path).expanduser().resolve()
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_step = int(checkpoint["step"])

            model_state = {}
            for key, value in checkpoint["pipeline"].items():
                if key.startswith("_model.module."):
                    model_state[key[len("_model.module."):]] = value
                elif key.startswith("_model."):
                    model_state[key[len("_model."):]] = value
            incompatible = model.load_state_dict(model_state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Model checkpoint mismatch: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
            model.update_to_step(checkpoint_step)
            model.eval()

        return cls(
            model=model,
            config=config,
            dataparser_outputs=dataparser_outputs,
            rotater=rotater,
            scene_scale=dataparser_outputs.dataparser_scale,
            device=requested_device,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            official_repo_root=official_repo_root,
        )

    def _validate_rays(self, origins: torch.Tensor, directions: torch.Tensor) -> None:
        for name, tensor in (("origins", origins), ("directions", directions)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} must have dtype float32")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
            if tensor.ndim != 2 or tensor.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [N,3], got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
        if origins.shape != directions.shape or origins.shape[0] < 1:
            raise ValueError("origins and directions must have the same non-empty [N,3] shape")
        if torch.any(torch.linalg.vector_norm(directions, dim=-1) == 0):
            raise ValueError("directions must be nonzero")

    def _camera_indices(self, camera_idx: TensorLikeIndex, ray_count: int) -> torch.Tensor:
        if isinstance(camera_idx, int):
            indices = torch.full((ray_count, 1), camera_idx, dtype=torch.long, device=self.device)
        elif isinstance(camera_idx, torch.Tensor):
            indices = camera_idx.to(device=self.device, dtype=torch.long)
            if indices.ndim == 1:
                indices = indices[:, None]
            if indices.shape == (1, 1):
                indices = indices.expand(ray_count, 1)
            if indices.shape != (ray_count, 1):
                raise ValueError(f"camera_idx must broadcast to [N,1], got {tuple(indices.shape)}")
        else:
            raise TypeError("camera_idx must be int or torch.Tensor")
        camera_count = len(self.dataparser_outputs.image_filenames)
        if torch.any(indices < 0) or torch.any(indices >= camera_count):
            raise ValueError(f"camera_idx must be in [0,{camera_count - 1}]")
        return indices

    def _near_column(self, near: TensorLikeNear, ray_count: int) -> torch.Tensor:
        near_tensor = torch.as_tensor(near, dtype=torch.float32, device=self.device)
        if near_tensor.ndim == 0:
            return near_tensor.expand(ray_count, 1)
        if near_tensor.ndim == 1:
            near_tensor = near_tensor[:, None]
        if near_tensor.shape == (1, 1):
            near_tensor = near_tensor.expand(ray_count, 1)
        if near_tensor.shape != (ray_count, 1):
            raise ValueError(f"near must broadcast to [N,1], got {tuple(near_tensor.shape)}")
        if not torch.isfinite(near_tensor).all():
            raise ValueError("near contains non-finite values")
        return near_tensor

    def external_to_nerfstudio(
        self, origins: torch.Tensor, directions: torch.Tensor, near: TensorLikeNear = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the exact official near shift, scene scaling, and axis transform."""
        self._validate_rays(origins, directions)
        near_column = self._near_column(near, origins.shape[0])
        shifted_origins = origins + near_column * directions
        scaled_origins = (shifted_origins * 2.0 - 1.0) * self.scene_scale
        ns_origins = (self._mi2gl_left[:3, :3] @ scaled_origins.T).T
        ns_directions = (self._mi2gl_left[:3, :3] @ directions.T).T
        return ns_origins.contiguous(), ns_directions.contiguous()

    def nerfstudio_to_external(
        self, origins: torch.Tensor, directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse conversion used by the camera-ray equivalence diagnostic."""
        self._validate_rays(origins, directions)
        unpermuted_origins = (self._gl2mi_left[:3, :3] @ origins.T).T
        external_origins = (unpermuted_origins / self.scene_scale + 1.0) * 0.5
        external_directions = (self._gl2mi_left[:3, :3] @ directions.T).T
        return external_origins.contiguous(), external_directions.contiguous()

    def make_ray_bundle(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        camera_idx: TensorLikeIndex,
        near: TensorLikeNear = 0.0,
    ) -> Any:
        """Convert public rays and construct the official internal RayBundle."""
        from nerfstudio.cameras.rays import RayBundle

        ns_origins, ns_directions = self.external_to_nerfstudio(origins, directions, near)
        ray_bundle = RayBundle(
            origins=ns_origins,
            directions=ns_directions,
            pixel_area=torch.ones_like(ns_origins[..., :1]),
            camera_indices=self._camera_indices(camera_idx, origins.shape[0]),
        )
        if self.rotater is not None:
            ray_bundle.rotater = self.rotater.apply_frustums
        return ray_bundle

    def query_radiance(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        camera_idx: TensorLikeIndex,
        near: TensorLikeNear = 0.0,
        chunk_size: int = 16384,
    ) -> torch.Tensor:
        """Return volume-rendered linear HDR RGB with shape ``[N,3]``.

        The official model's query entry is intentionally retained. It runs in
        deterministic eval mode and includes collider, proposal sampling, field
        evaluation, density weights, and RGB accumulation.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._validate_rays(origins, directions)
        camera_indices = self._camera_indices(camera_idx, origins.shape[0])
        near_column = self._near_column(near, origins.shape[0])
        # tiny-cuda-nn selects different kernels for extremely small batches;
        # their half-precision reduction order creates avoidable ~1e-3 drift.
        # Promote tiny requested chunks to the smallest empirically stable batch.
        effective_chunk_size = max(chunk_size, self._MIN_STABLE_TCNN_CHUNK)
        rgb_chunks = []
        for start in range(0, origins.shape[0], effective_chunk_size):
            end = min(start + effective_chunk_size, origins.shape[0])
            ray_bundle = self.make_ray_bundle(
                origins[start:end],
                directions[start:end],
                camera_indices[start:end],
                near_column[start:end],
            )
            rgb_chunks.append(self.model.get_rgb_for_camera_ray_bundle(ray_bundle))
        rgb = torch.cat(rgb_chunks, dim=0)
        if rgb.shape != (origins.shape[0], 3):
            raise RuntimeError(f"Expected RGB [N,3], got {tuple(rgb.shape)}")
        if not torch.isfinite(rgb).all():
            raise RuntimeError("Query produced non-finite RGB")
        return rgb

    @torch.inference_mode()
    def debug_query_environment_ray(
        self, origin: torch.Tensor, direction: torch.Tensor, camera_idx: int, near: float = 0.0
    ) -> dict:
        """Trace one ray through the unchanged model and expose volume samples.

        This diagnostic method intentionally does not replace ``query_radiance``
        in normal rendering.
        """
        from nerfstudio.field_components.field_heads import FieldHeadNames
        from nerfstudio.model_components.losses import scale_gradients_by_distance_squared

        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError("debug ray origin/direction must be [3]")
        ray_bundle=self.make_ray_bundle(origin[None].float(),direction[None].float(),camera_idx,near)
        model=self.model
        if model.collider is not None:
            ray_bundle=model.collider(ray_bundle)
        ray_samples,_,_=model.proposal_sampler(ray_bundle,density_fns=model.density_fns)
        if model.rotater is not None:
            ray_samples.camera_indices=model.rotater.map_rotation_ids(ray_samples.camera_indices)
        field_outputs=model.field.forward(ray_samples,compute_normals=model.config.predict_normals)
        if model.config.use_gradient_scaling:
            field_outputs=scale_gradients_by_distance_squared(field_outputs,ray_samples)
        density=field_outputs[FieldHeadNames.DENSITY]
        rgb_samples=field_outputs[FieldHeadNames.RGB]
        weights=ray_samples.get_weights(density)
        final=model.renderer_rgb(rgb=rgb_samples,weights=weights)
        starts=ray_samples.frustums.starts; ends=ray_samples.frustums.ends; delta=ends-starts
        alpha=1-torch.exp(-density*delta)
        trans=torch.cumprod(torch.cat((torch.ones_like(alpha[...,:1,:]),1-alpha+1e-7),dim=-2),dim=-2)[...,:-1,:]
        contribution=weights*rgb_samples
        return {
            "origin_external":origin.detach().cpu().tolist(),"direction_external":direction.detach().cpu().tolist(),
            "near":float(near),"camera_idx":int(camera_idx),"environment_only":True,
            "trace_kind":"diagnostic single-ray retrace; stateful stratified proposal samples are not the stored final-render samples",
            "t":((starts+ends)*.5)[0,:,0].detach().cpu().tolist(),
            "sigma":density[0,:,0].detach().cpu().tolist(),"alpha":alpha[0,:,0].detach().cpu().tolist(),
            "transmittance":trans[0,:,0].detach().cpu().tolist(),"weight":weights[0,:,0].detach().cpu().tolist(),
            "rgb_i":rgb_samples[0].detach().cpu().tolist(),"contribution_i":contribution[0].detach().cpu().tolist(),
            "sum_weighted_rgb":contribution.sum(-2)[0].detach().cpu().tolist(),
            "final_Li":final[0].detach().cpu().tolist(),"sample_count":int(density.shape[-2]),
        }


def load_official_nerf_emitter(
    config_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cuda",
) -> StandaloneNerfEmitter:
    """Convenience wrapper around :meth:`from_official_checkpoint`."""
    return StandaloneNerfEmitter.from_official_checkpoint(config_path, checkpoint_path, device)
