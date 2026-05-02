from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(max(num_bytes, 0))
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{x:.1f}TB"


def _torch_load_with_heartbeat(
    path: str,
    tag: str,
    map_location: str = "cpu",
    heartbeat_sec: float = 20.0,
    mmap: bool = False,
):
    size_str = "unknown"
    try:
        size_str = _format_bytes(os.path.getsize(path))
    except OSError:
        pass

    mmap_suffix = " mmap=True" if mmap else ""
    print(f"[io] loading {tag}{mmap_suffix}: {path} ({size_str})", flush=True)
    t0 = time.time()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(heartbeat_sec):
            elapsed = time.time() - t0
            print(f"[io] still loading {tag} ({elapsed:.1f}s): {path}", flush=True)

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        obj = torch.load(path, map_location=map_location, mmap=mmap)
    finally:
        stop_event.set()
        thread.join(timeout=0.2)

    elapsed = time.time() - t0
    print(f"[io] loaded {tag} in {elapsed:.1f}s: {path}", flush=True)
    return obj


class PETFirstHopAligned4HopDataset(Dataset):
    """Aligned latent+image dataset for first-hop latent transport training."""

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
        verify_alignment: bool = True,
        alignment_check_num_samples: int = 16,
        alignment_audit_json: str | None = None,
        include_x_rollout_first: bool = True,
        include_full_x_rollout: bool = False,
        image_size: int = 224,
        image_timepoints: Sequence[str] | None = None,
        latent_mmap: bool = False,
    ) -> None:
        super().__init__()
        self.split = split
        self.clamp_max = float(clamp_max)
        self.image_size = int(image_size)
        self.t_map = t_map or {
            "D50": 2.0,
            "D20": 5.0,
            "D10": 10.0,
            "D4": 25.0,
            "NORMAL": 100.0,
        }
        self.rollout_timepoints = rollout_timepoints or list(self.TIMEPOINTS)
        self.verify_alignment = bool(verify_alignment)
        self.alignment_check_num_samples = int(alignment_check_num_samples)
        self.alignment_audit_json = alignment_audit_json
        self.include_x_rollout_first = bool(include_x_rollout_first)
        self.include_full_x_rollout = bool(include_full_x_rollout)
        self.latent_mmap = bool(latent_mmap)

        if self.alignment_check_num_samples <= 0:
            raise ValueError("alignment_check_num_samples must be positive")
        if not self.rollout_timepoints:
            raise ValueError("rollout_timepoints must not be empty")
        for tp in self.rollout_timepoints:
            if tp not in self.TIMEPOINTS:
                raise ValueError(f"Unsupported rollout timepoint: {tp}")
        if self.rollout_timepoints[0] != "D50":
            raise ValueError(
                "first-hop dataset requires rollout_timepoints[0] == 'D50' "
                f"so x_rollout_first is unambiguously sourced from D50, got {self.rollout_timepoints[0]!r}"
            )
        if self.include_full_x_rollout:
            if len(self.rollout_timepoints) < 2 or self.rollout_timepoints[1] != "D20":
                raise ValueError(
                    "first-hop dataset requires rollout_timepoints[1] == 'D20' when include_full_x_rollout=true "
                    "so x_rollout[1] remains the first prediction target"
                )

        latent_blob = _torch_load_with_heartbeat(
            latent_path,
            tag=f"latents({split})",
            map_location="cpu",
            mmap=self.latent_mmap,
        )
        self.latents: Dict[str, torch.Tensor] = {}
        for tp in self.TIMEPOINTS:
            if tp not in latent_blob:
                raise KeyError(f"Missing latent timepoint {tp!r} in {latent_path}")
            self.latents[tp] = latent_blob[tp].float()

        self.image_timepoints = self._resolve_required_image_timepoints(image_timepoints)
        self.images = self._load_images(raw_data_dir, split, self.image_timepoints)
        self.num_slices = int(self.latents["NORMAL"].shape[0])
        self._verify_shapes()

        self.index: List[Tuple[int, int]] = []
        for pair_idx in range(len(self.PAIRS)):
            for slice_idx in range(self.num_slices):
                self.index.append((pair_idx, slice_idx))

        self.alignment_summary: Dict[str, Dict[str, float]] = {}
        if self.verify_alignment:
            self.alignment_summary = self._run_alignment_checks()

    def _verify_shapes(self) -> None:
        latent_n = self.latents["NORMAL"].shape[0]

        for tp in self.TIMEPOINTS:
            if self.latents[tp].shape[0] != latent_n:
                raise RuntimeError(f"Latent slice count mismatch at {tp}")
            if tp in self.images and self.images[tp].shape[0] != latent_n:
                raise RuntimeError(
                    f"Image slice count mismatch at {tp}: "
                    f"latents NORMAL={latent_n}, images {tp}={self.images[tp].shape[0]}"
                )

    def _normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.clamp_max)
        x = (x / (self.clamp_max / 2.0)) - 1.0
        return x

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [N, C, H, W], got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h == self.image_size and w == self.image_size:
            return x
        return F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)

    def _resolve_required_image_timepoints(self, image_timepoints: Sequence[str] | None) -> List[str]:
        if image_timepoints is not None:
            requested = [str(tp) for tp in image_timepoints]
        else:
            # The training objective consumes D50 as hop0 pixel input and D20
            # as hop0 image target. Full chain image metrics opt into all images.
            requested = ["D50", "D20"]
            if self.include_full_x_rollout:
                requested.extend(self.rollout_timepoints)
            if self.verify_alignment and not self.alignment_audit_json:
                requested.extend(["D50", "D20", "NORMAL"])

        allowed = set(self.TIMEPOINTS)
        deduped: List[str] = []
        for tp in requested:
            if tp not in allowed:
                raise ValueError(f"Unsupported image timepoint: {tp}")
            if tp not in deduped:
                deduped.append(tp)
        if "D50" not in deduped or "D20" not in deduped:
            raise ValueError("first-hop dataset requires image_timepoints to include D50 and D20")
        return deduped

    def _load_images(self, raw_data_dir: str, split: str, image_timepoints: Sequence[str]) -> Dict[str, torch.Tensor]:
        images: Dict[str, torch.Tensor] = {}
        requested = set(image_timepoints)
        print(f"[io] image timepoints({split})={list(image_timepoints)}", flush=True)
        for tp in ["D50", "D20", "D10", "D4"]:
            if tp not in requested:
                continue
            path = os.path.join(raw_data_dir, f"preprocessed_data_{tp}.pt")
            blob = _torch_load_with_heartbeat(
                path,
                tag=f"raw({split},{tp})",
                map_location="cpu",
            )
            x = self._normalize_image(blob[split]["x_T"]).float()
            images[tp] = self._resize_if_needed(x)

        # NORMAL target shares x_0 in the raw PT files; D50 file is used as canonical source.
        if "NORMAL" in requested:
            normal_path = os.path.join(raw_data_dir, "preprocessed_data_D50.pt")
            normal_blob = _torch_load_with_heartbeat(
                normal_path,
                tag=f"raw({split},NORMAL_from_D50_x0)",
                map_location="cpu",
            )
            normal = self._normalize_image(normal_blob[split]["x_0"]).float()
            images["NORMAL"] = self._resize_if_needed(normal)
        return images

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean()
        std = x.std(unbiased=False).clamp_min(1e-8)
        return (x - mu) / std

    @staticmethod
    def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
        az = PETFirstHopAligned4HopDataset._zscore(a)
        bz = PETFirstHopAligned4HopDataset._zscore(b)
        return float((az * bz).mean().item())

    def _run_alignment_checks(self) -> Dict[str, Dict[str, float]]:
        if self.alignment_audit_json:
            return self._load_alignment_audit()
        return self._run_scalar_alignment_checks()

    def _load_alignment_audit(self) -> Dict[str, Dict[str, float]]:
        if not os.path.exists(self.alignment_audit_json):
            raise RuntimeError(f"alignment_audit_json not found: {self.alignment_audit_json}")
        with open(self.alignment_audit_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        metric = payload.get("meta", {}).get("psnr_metric", "")
        if metric != "src.utils.metrics.calc_psnr_clip3":
            raise RuntimeError(
                "alignment audit must use src.utils.metrics.calc_psnr_clip3, "
                f"got: {metric!r}"
            )

        split_res = payload.get("results", {}).get(self.split, None)
        if split_res is None:
            raise RuntimeError(
                f"alignment audit missing split={self.split!r} in {self.alignment_audit_json}"
            )

        check_tps = ["D50", "D20", "NORMAL"]
        summary: Dict[str, Dict[str, float]] = {}
        for tp in check_tps:
            tp_res = split_res.get(tp, None)
            if tp_res is None:
                raise RuntimeError(
                    f"alignment audit missing timepoint {tp!r} for split={self.split!r}"
                )
            if not bool(tp_res.get("alignment_signal_positive", False)):
                raise RuntimeError(
                    "alignment audit failed for "
                    f"split={self.split}, timepoint={tp}: {tp_res}"
                )

            psnr = tp_res.get("psnr_clip3", {})
            matched = psnr.get("matched", {})
            roll1 = psnr.get("roll1", {})
            rand = psnr.get("random", {})
            summary[tp] = {
                "psnr_clip3_matched_mean": float(matched.get("mean", float("nan"))),
                "psnr_clip3_roll1_mean": float(roll1.get("mean", float("nan"))),
                "psnr_clip3_random_mean": float(rand.get("mean", float("nan"))),
                "psnr_gap_vs_roll1_db": float(tp_res.get("psnr_gap_vs_roll1_db", float("nan"))),
                "psnr_gap_vs_random_db": float(tp_res.get("psnr_gap_vs_random_db", float("nan"))),
                "n_eval": float(tp_res.get("n_eval", float("nan"))),
            }
        return summary

    def _run_scalar_alignment_checks(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        check_tps = ["D50", "D20", "NORMAL"]
        n = self.num_slices
        k = min(self.alignment_check_num_samples, n)
        idx = torch.linspace(0, n - 1, steps=k).round().long()

        for tp in check_tps:
            lat_scalar = self.latents[tp].mean(dim=(1, 2, 3))
            img_scalar = self.images[tp].mean(dim=(1, 2, 3))

            lat_s = lat_scalar[idx]
            img_s = img_scalar[idx]
            img_shift = img_s.roll(1, dims=0)

            lat_z = self._zscore(lat_s)
            img_z = self._zscore(img_s)
            img_shift_z = self._zscore(img_shift)

            matched_l1 = float((lat_z - img_z).abs().mean().item())
            mismatched_l1 = float((lat_z - img_shift_z).abs().mean().item())
            matched_corr = self._corr(lat_s, img_s)
            mismatched_corr = self._corr(lat_s, img_shift)

            strict_l1_ok = matched_l1 <= (mismatched_l1 * 0.90)
            strict_corr_ok = matched_corr >= (mismatched_corr + 0.10)
            weak_l1_ok = matched_l1 < mismatched_l1
            weak_corr_ok = matched_corr > mismatched_corr

            if not ((strict_l1_ok and strict_corr_ok) or (weak_l1_ok and weak_corr_ok)):
                raise RuntimeError(
                    "Alignment content check failed at "
                    f"{tp}: matched_l1={matched_l1:.6f}, mismatched_l1={mismatched_l1:.6f}, "
                    f"matched_corr={matched_corr:.6f}, mismatched_corr={mismatched_corr:.6f}"
                )

            if not (strict_l1_ok and strict_corr_ok):
                print(
                    "[alignment][warn] borderline content check at "
                    f"{tp}: matched_l1={matched_l1:.6f}, mismatched_l1={mismatched_l1:.6f}, "
                    f"matched_corr={matched_corr:.6f}, mismatched_corr={mismatched_corr:.6f}"
                )

            summary[tp] = {
                "decoded_vs_raw_l1_matched": matched_l1,
                "decoded_vs_raw_l1_mismatched": mismatched_l1,
                "decoded_vs_raw_corr_matched": matched_corr,
                "decoded_vs_raw_corr_mismatched": mismatched_corr,
                "num_samples": float(k),
            }
        return summary

    def __len__(self) -> int:
        return len(self.index)

    def _image_or_zero(self, tp: str, slice_idx: int) -> torch.Tensor:
        if tp in self.images:
            return self.images[tp][slice_idx].clone()
        # Non-hop0 pair rows do not consume x_src/x_dst in the current model/loss.
        # Return a correctly shaped placeholder so mixed-pair batches collate.
        template = next(iter(self.images.values()))
        return torch.zeros_like(template[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair_idx, slice_idx = self.index[idx]
        src_tp, dst_tp = self.PAIRS[pair_idx]

        z_src = self.latents[src_tp][slice_idx].clone()
        z_dst = self.latents[dst_tp][slice_idx].clone()
        x_src = self._image_or_zero(src_tp, slice_idx)
        x_dst = self._image_or_zero(dst_tp, slice_idx)

        z_rollout = torch.stack([self.latents[tp][slice_idx].clone() for tp in self.rollout_timepoints], dim=0)

        out = {
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
        }

        if self.include_x_rollout_first:
            first_tp = self.rollout_timepoints[0]
            out["x_rollout_first"] = self.images[first_tp][slice_idx].clone()

        if self.include_full_x_rollout:
            missing = [tp for tp in self.rollout_timepoints if tp not in self.images]
            if missing:
                raise RuntimeError(
                    "include_full_x_rollout=true requires all rollout images to be loaded; "
                    f"missing={missing}, loaded={sorted(self.images)}"
                )
            x_rollout = torch.stack([self.images[tp][slice_idx].clone() for tp in self.rollout_timepoints], dim=0)
            out["x_rollout"] = x_rollout

        return out


class Hop0OnlyViewDataset(Dataset):
    """Dataset view that only keeps hop0 (D50->D20) samples."""

    def __init__(self, base: PETFirstHopAligned4HopDataset) -> None:
        self.base = base
        self.indices = [i for i, (pair_idx, _) in enumerate(base.index) if pair_idx == 0]
        if not self.indices:
            raise RuntimeError("No hop0 samples found in base dataset")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base[self.indices[idx]]
