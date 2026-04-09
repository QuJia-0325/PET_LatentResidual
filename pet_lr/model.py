from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .bootstrap import ensure_rae_importable


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = self.depthwise(x)
        h = self.pointwise(h)
        h = self.norm(h)
        h = h * (1.0 + gamma) + beta
        h = self.act(h)
        return x + h


class ResidualRefineHead(nn.Module):
    """Small local head that refines decoder drafts, not a second generator."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        max_residual: float = 0.20,
        num_hops: int = 4,
    ) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.hop_embed = nn.Embedding(num_hops, hidden_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([FiLMResidualBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)])
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _build_cond(self, t_src: torch.Tensor, t_dst: torch.Tensor, hop_idx: torch.Tensor) -> torch.Tensor:
        log_dt = torch.log(t_dst.clamp(min=1e-6)) - torch.log(t_src.clamp(min=1e-6))
        time_feat = torch.stack([t_src, t_dst, log_dt], dim=-1)
        return self.time_mlp(time_feat) + self.hop_embed(hop_idx.long())

    def forward(
        self,
        x_src_draft: torch.Tensor,
        x_dst_draft: torch.Tensor,
        t_src: torch.Tensor,
        t_dst: torch.Tensor,
        hop_idx: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._build_cond(t_src, t_dst, hop_idx)
        x = torch.cat([x_src_draft, x_dst_draft], dim=1)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, cond)
        residual = torch.tanh(self.out(h)) * self.max_residual
        return residual


class LatentResidualRefinementModel(nn.Module):
    """Frozen latent backbone + frozen RAE decoder + small residual head."""

    _PAIR_V_STD = [0.009634, 0.002946, 0.000775, 0.000141]

    def __init__(
        self,
        cfg: Dict,
        device: torch.device,
    ) -> None:
        super().__init__()
        ensure_rae_importable(cfg["rae_root"])
        self.backbone, self.target_normalize = build_backbone(cfg)
        self.rae = build_rae(cfg, device)

        train_cfg = cfg.get("training", {})
        self.freeze_backbone = bool(train_cfg.get("freeze_backbone", True))
        self.freeze_rae = bool(train_cfg.get("freeze_rae", True))

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if self.freeze_rae:
            self.rae.requires_grad_(False)
            self.rae.eval()

        head_cfg = cfg.get("residual_head", {})
        self.residual_head = ResidualRefineHead(
            hidden_channels=int(head_cfg.get("hidden_channels", 64)),
            num_blocks=int(head_cfg.get("num_blocks", 4)),
            max_residual=float(head_cfg.get("max_residual", 0.20)),
            num_hops=4,
        )

    def _maybe_latent_context(self):
        return torch.no_grad() if self.freeze_backbone else nullcontext()

    def decode_crop(self, z: torch.Tensor) -> torch.Tensor:
        x = self.rae.decode(z)
        _, _, h, w = x.shape
        top = max((h - 192) // 2, 0)
        left = max((w - 192) // 2, 0)
        return x[:, 0:1, top:top + 192, left:left + 192]

    def _sigma_for_hop(self, hop_idx: torch.Tensor) -> torch.Tensor:
        std = torch.tensor(self._PAIR_V_STD, dtype=torch.float32, device=hop_idx.device)
        return std[hop_idx.long()].view(-1, 1, 1, 1)

    def predict_latent_step(
        self,
        z_src: torch.Tensor,
        t_src: torch.Tensor,
        t_dst: torch.Tensor,
        hop_idx: torch.Tensor,
    ) -> torch.Tensor:
        with self._maybe_latent_context():
            v_pred = self.backbone(z_src, t_src, t_dst=t_dst, hop_idx=hop_idx)
            dt = (t_dst - t_src).view(-1, 1, 1, 1)
            if self.target_normalize:
                sigma = self._sigma_for_hop(hop_idx)
                v_pred = v_pred.float() * sigma.float()
                z_pred = z_src.float() + v_pred * dt.float()
            else:
                z_pred = z_src + v_pred * dt
        return z_pred

    def forward_step(
        self,
        z_src: torch.Tensor,
        t_src: torch.Tensor,
        t_dst: torch.Tensor,
        hop_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_pred = self.predict_latent_step(z_src, t_src, t_dst, hop_idx)
        with self._maybe_latent_context():
            x_src_draft = self.decode_crop(z_src)
            x_dst_draft = self.decode_crop(z_pred)
        residual = self.residual_head(x_src_draft, x_dst_draft, t_src, t_dst, hop_idx)
        x_refined = torch.clamp(x_dst_draft + residual, min=-1.0, max=1.0)
        return {
            "z_pred": z_pred,
            "x_src_draft": x_src_draft,
            "x_dst_draft": x_dst_draft,
            "residual": residual,
            "x_refined": x_refined,
        }


def build_backbone(cfg: Dict) -> Tuple[nn.Module, bool]:
    from src.pet_flow.models.pet_flow_dit_hop import PETFlowDiTDHHopAware

    backbone_cfg = cfg.get("backbone", {})
    model_cfg = backbone_cfg.get("model", {})
    data_cfg = backbone_cfg.get("data", {})
    model = PETFlowDiTDHHopAware(
        input_size=model_cfg.get("input_size", 14),
        patch_size=model_cfg.get("patch_size", 1),
        in_channels=model_cfg.get("in_channels", 768),
        hidden_size=model_cfg.get("hidden_size", [384, 2048]),
        depth=model_cfg.get("depth", [12, 2]),
        num_heads=model_cfg.get("num_heads", [6, 16]),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        timepoints=data_cfg.get("timepoints", ["D50", "D20", "D10", "D4", "NORMAL"]),
        t_map=data_cfg.get(
            "t_map",
            {"D50": 2.0, "D20": 5.0, "D10": 10.0, "D4": 25.0, "NORMAL": 100.0},
        ),
    )
    ckpt = torch.load(backbone_cfg["checkpoint_path"], map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    return model, bool(ckpt.get("target_normalize", False))


def build_rae(cfg: Dict, device: torch.device) -> nn.Module:
    from src.pet_flow.inference_pet_flow import load_rae_model

    rae = load_rae_model(cfg, device)
    return rae
