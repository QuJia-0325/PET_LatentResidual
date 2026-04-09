from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class PETLatentResidual4HopDataset(Dataset):
    """Aligned 4-hop dataset with both latent states and raw PET images.

    Assumption:
    - `latents_{split}.pt` and `preprocessed_data_*.pt` share the same slice order.
    - `x_0` is the NORMAL target for every low-dose PT file.
    """

    TIMEPOINTS = ["D50", "D20", "D10", "D4", "NORMAL"]
    PAIRS = [
        ("D50", "D20"),
        ("D20", "D10"),
        ("D10", "D4"),
        ("D4", "NORMAL"),
    ]

    def __init__(
        self,
        latent_path: str,
        raw_data_dir: str,
        split: str = "train",
        clamp_max: float = 10.0,
        t_map: Dict[str, float] | None = None,
        rollout_timepoints: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.clamp_max = clamp_max
        self.t_map = t_map or {
            "D50": 2.0,
            "D20": 5.0,
            "D10": 10.0,
            "D4": 25.0,
            "NORMAL": 100.0,
        }
        self.rollout_timepoints = rollout_timepoints or list(self.TIMEPOINTS)

        latent_blob = torch.load(latent_path, map_location="cpu")
        self.latents: Dict[str, torch.Tensor] = {}
        for tp in self.TIMEPOINTS:
            if tp not in latent_blob:
                raise KeyError(f"Missing latent timepoint {tp!r} in {latent_path}")
            self.latents[tp] = latent_blob[tp]

        self.images = self._load_images(raw_data_dir, split)
        self.num_slices = self.latents["NORMAL"].shape[0]
        self.index: List[Tuple[int, int]] = []
        for pair_idx in range(len(self.PAIRS)):
            for slice_idx in range(self.num_slices):
                self.index.append((pair_idx, slice_idx))

    def _normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.clamp_max)
        x = (x / (self.clamp_max / 2.0)) - 1.0
        return x

    def _load_images(self, raw_data_dir: str, split: str) -> Dict[str, torch.Tensor]:
        images: Dict[str, torch.Tensor] = {}
        for tp in ["D50", "D20", "D10", "D4"]:
            path = os.path.join(raw_data_dir, f"preprocessed_data_{tp}.pt")
            blob = torch.load(path, map_location="cpu")
            images[tp] = self._normalize_image(blob[split]["x_T"]).float()

        normal_path = os.path.join(raw_data_dir, "preprocessed_data_D50.pt")
        normal_blob = torch.load(normal_path, map_location="cpu")
        images["NORMAL"] = self._normalize_image(normal_blob[split]["x_0"]).float()
        return images

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair_idx, slice_idx = self.index[idx]
        src_tp, dst_tp = self.PAIRS[pair_idx]

        z_src = self.latents[src_tp][slice_idx].clone()
        z_dst = self.latents[dst_tp][slice_idx].clone()
        x_src = self.images[src_tp][slice_idx].clone().float()
        x_dst = self.images[dst_tp][slice_idx].clone().float()

        z_rollout = torch.stack(
            [self.latents[tp][slice_idx].clone() for tp in self.rollout_timepoints],
            dim=0,
        )
        x_rollout = torch.stack(
            [self.images[tp][slice_idx].clone() for tp in self.rollout_timepoints],
            dim=0,
        )

        return {
            "z_src": z_src,
            "z_dst": z_dst,
            "x_src": x_src,
            "x_dst": x_dst,
            "t_src": torch.tensor(self.t_map[src_tp], dtype=torch.float32),
            "t_dst": torch.tensor(self.t_map[dst_tp], dtype=torch.float32),
            "hop_idx": torch.tensor(pair_idx, dtype=torch.long),
            "pair_idx": torch.tensor(pair_idx, dtype=torch.long),
            "slice_idx": torch.tensor(slice_idx, dtype=torch.long),
            "z_rollout": z_rollout,
            "x_rollout": x_rollout,
        }
