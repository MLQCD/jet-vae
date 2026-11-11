"""Torch dataset over the ``.npy`` cache written by ``jetvae.data.preprocess``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["CachedParticleShapeDataset", "collate_cached_batch"]


class CachedParticleShapeDataset(Dataset):
    def __init__(self, particles_path: str | Path, mask_path: str | Path) -> None:
        self.particles = np.load(particles_path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        if self.particles.ndim != 3 or self.particles.shape[-1] != 3:
            raise ValueError(f"Expected particles [N, K, 3], got {self.particles.shape}")
        if self.mask.shape != self.particles.shape[:2]:
            raise ValueError(f"Expected mask {self.particles.shape[:2]}, got {self.mask.shape}")
        if len(self.particles) == 0:
            raise ValueError("CachedParticleShapeDataset cannot be empty")

    def __len__(self) -> int:
        return int(self.particles.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        particles = torch.from_numpy(np.array(self.particles[index], copy=True)).float()
        return {
            "particles": particles,
            "momenta": particles,
            "mask": torch.from_numpy(np.array(self.mask[index], copy=True)).bool(),
        }


def collate_cached_batch(batch: list[dict[str, torch.Tensor]]):
    return {k: torch.stack([item[k] for item in batch], dim=0)
            for k in ("particles", "momenta", "mask")}
