from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bootstrap import ensure_rae_importable


def _num_parameters(module: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters()))


def _inverse_softplus_scalar(y: float) -> float:
    """Numerically stable inverse softplus for scalar initialization."""
    y = float(y)
    if y <= 0.0:
        # Large negative maps softplus close to 0.
        return -20.0
    if y > 20.0:
        # softplus(x) ~= x in this regime.
        return y
    return math.log(math.expm1(y))


class FirstHopPixelEncoder(nn.Module):
    """Weak local pixel encoder used only for hop0 latent forcing."""

    def __init__(
        self,
        latent_channels: int = 768,
        latent_size: int = 16,
        stem_channels: int = 32,
        hidden_channels: int = 64,
        proj_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, stem_channels),
            nn.SiLU(),
            nn.Conv2d(stem_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((latent_size, latent_size))
        self.proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        # Default: deterministic zero init.
        # Branch activation is then controlled by a tiny non-zero gate.
        if float(proj_init_std) > 0.0:
            nn.init.normal_(self.proj.weight, mean=0.0, std=float(proj_init_std))
            nn.init.zeros_(self.proj.bias)
        else:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected x in [B,1,H,W], got {tuple(x.shape)}")
        h = self.stem(x)
        h = self.pool(h)
        return self.proj(h)


class SpatialAlignmentProjector(nn.Module):
    """iREPA-style spatial projector for representation alignment.

    Unlike the original REPA which uses an MLP projector (losing spatial structure),
    this uses Conv2d to preserve and leverage spatial relationships between tokens.
    For PET imaging where anatomical structure preservation is critical, the spatial
    projector provides better alignment quality.

    Reference: iREPA (improved REPA) — spatial-aware projector for dense prediction.
    """

    def __init__(
        self,
        in_channels: int = 384,
        out_channels: int = 768,
        hidden_channels: int = 256,
        spatial_size: int = 16,
    ) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project backbone hidden state to alignment space.

        Args:
            h: Hidden state [B, N, D] (token sequence) or [B, D, H, W] (spatial)

        Returns:
            Projected features [B, out_channels, H, W]
        """
        if h.dim() == 3:
            b, n, d = h.shape
            s = self.spatial_size
            h = h.transpose(1, 2).reshape(b, d, s, s)
        return self.projector(h)


class HopResidualVelocityHead(nn.Module):
    """Small hop-specific residual correction on top of shared velocity output."""

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 192,
        num_hops: int = 4,
        last_init_std: float = 0.0,
        lambda_hop_init: float = 1.0e-3,
        lambda_hop_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_hops = int(num_hops)
        self.lambda_hop_floor = float(lambda_hop_floor)
        if self.lambda_hop_floor < 0.0:
            raise ValueError(f"lambda_hop_floor must be >= 0, got {self.lambda_hop_floor}")

        lambda_hop_init = float(lambda_hop_init)
        if lambda_hop_init < self.lambda_hop_floor:
            raise ValueError(
                f"lambda_hop_init ({lambda_hop_init}) must be >= lambda_hop_floor ({self.lambda_hop_floor})"
            )
        lambda_raw_init = _inverse_softplus_scalar(lambda_hop_init - self.lambda_hop_floor)
        self.lambda_hop_raw = nn.Parameter(
            torch.full((self.num_hops,), float(lambda_raw_init), dtype=torch.float32)
        )

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
                )
                for _ in range(self.num_hops)
            ]
        )
        for head in self.heads:
            last = head[-1]
            # Default: deterministic zero init.
            # Branch activation is then controlled by tiny non-zero lambda_hop.
            if float(last_init_std) > 0.0:
                nn.init.normal_(last.weight, mean=0.0, std=float(last_init_std))
                nn.init.zeros_(last.bias)
            else:
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def lambda_hop_value(self) -> torch.Tensor:
        # Keep hop residual branch strictly positive with configurable non-zero floor.
        return self.lambda_hop_floor + F.softplus(self.lambda_hop_raw)

    @property
    def lambda_hop(self) -> torch.Tensor:
        # Backward-compatible accessor used by logging code.
        return self.lambda_hop_value()

    def forward(self, v_shared: torch.Tensor, hop_idx: torch.Tensor) -> torch.Tensor:
        if v_shared.dim() != 4:
            raise ValueError(f"Expected velocity in [B,C,H,W], got {tuple(v_shared.shape)}")
        hop_idx = hop_idx.long()
        residual = torch.zeros_like(v_shared)
        lambda_hop = self.lambda_hop_value()
        for hop in range(self.num_hops):
            mask = hop_idx == hop
            if torch.any(mask):
                v_sub = v_shared[mask]
                v_corr = self.heads[hop](v_sub)
                residual[mask] = lambda_hop[hop] * v_corr
        return residual


class PETFlowDiTFirstHop(nn.Module):
    """First-hop latent transport model with hop0 pixel forcing and hop residuals."""

    _PAIR_V_STD = [0.009634, 0.002946, 0.000775, 0.000141]

    def __init__(self, cfg: Dict, device: torch.device) -> None:
        super().__init__()
        ensure_rae_importable(cfg["rae_root"])
        self.cfg = cfg
        self.backbone, self.target_normalize = build_backbone(cfg)
        self.rae = build_rae(cfg, device)

        train_cfg = cfg.get("training", {})
        self.freeze_backbone = bool(train_cfg.get("freeze_backbone", False))
        self.freeze_rae = bool(train_cfg.get("freeze_rae", True))

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        if self.freeze_rae:
            self.rae.requires_grad_(False)
            self.rae.eval()

        model_cfg = cfg.get("backbone", {}).get("model", {})
        first_cfg = cfg.get("first_hop", {})
        self.latent_channels = int(model_cfg.get("in_channels", 768))
        self.latent_size = int(model_cfg.get("input_size", 16))
        self.num_hops = int(first_cfg.get("num_hops", 4))
        self.image_size = int(cfg.get("data", {}).get("image_size", cfg.get("rae", {}).get("encoder_input_size", 224)))

        self.pixel_encoder = FirstHopPixelEncoder(
            latent_channels=self.latent_channels,
            latent_size=self.latent_size,
            stem_channels=int(first_cfg.get("pixel_encoder_stem_channels", 32)),
            hidden_channels=int(first_cfg.get("pixel_encoder_hidden_channels", 64)),
            proj_init_std=float(first_cfg.get("pixel_proj_init_std", 0.0)),
        )
        self._assert_pixel_encoder_capacity(first_cfg)
        self.hop_residual_head = HopResidualVelocityHead(
            in_channels=self.latent_channels,
            hidden_channels=int(first_cfg.get("hop_residual_hidden_channels", 192)),
            num_hops=self.num_hops,
            last_init_std=float(first_cfg.get("hop_residual_last_init_std", 0.0)),
            lambda_hop_init=float(first_cfg.get("lambda_hop_init", 1.0e-3)),
            lambda_hop_floor=float(first_cfg.get("lambda_hop_floor", 0.0)),
        )

        self.pixel_gate_floor = float(first_cfg.get("pixel_gate_floor", 0.0))
        if self.pixel_gate_floor < 0.0:
            raise ValueError(f"pixel_gate_floor must be >= 0, got {self.pixel_gate_floor}")
        pixel_gate_init = float(first_cfg.get("pixel_gate_init", 1.0e-3))
        if pixel_gate_init < self.pixel_gate_floor:
            raise ValueError(
                f"pixel_gate_init ({pixel_gate_init}) must be >= pixel_gate_floor ({self.pixel_gate_floor})"
            )
        pixel_gate_raw_init = _inverse_softplus_scalar(pixel_gate_init - self.pixel_gate_floor)
        self.g_pix_raw = nn.Parameter(torch.tensor(float(pixel_gate_raw_init), dtype=torch.float32))

        pair_v_std = first_cfg.get("pair_v_std", self._PAIR_V_STD)
        if len(pair_v_std) != self.num_hops:
            raise ValueError(f"pair_v_std length ({len(pair_v_std)}) must equal num_hops ({self.num_hops})")
        self.register_buffer("pair_v_std", torch.tensor(pair_v_std, dtype=torch.float32), persistent=False)

        # --- iREPA-style spatial alignment (Scheme A) ---
        align_cfg = first_cfg.get("alignment", {})
        self.alignment_enabled = bool(align_cfg.get("enabled", False))
        self._hook_handle = None
        self._hooked_hidden = None
        if self.alignment_enabled:
            backbone_hidden = int(model_cfg.get("hidden_size", [384, 2048])[0])
            align_out = int(align_cfg.get("proj_out_channels", self.latent_channels))
            align_hidden = int(align_cfg.get("proj_hidden_channels", 256))
            self.alignment_projector = SpatialAlignmentProjector(
                in_channels=backbone_hidden,
                out_channels=align_out,
                hidden_channels=align_hidden,
                spatial_size=self.latent_size,
            )
            self._align_layer_idx = int(align_cfg.get("layer_idx", 6))
            self._install_backbone_hook()

    def _assert_pixel_encoder_capacity(self, first_cfg: Dict) -> None:
        max_ratio = float(first_cfg.get("pixel_encoder_max_ratio", 0.05))
        if max_ratio <= 0.0:
            raise ValueError(f"pixel_encoder_max_ratio must be > 0, got {max_ratio}")
        backbone_params = _num_parameters(self.backbone)
        pixel_params = _num_parameters(self.pixel_encoder)
        if backbone_params <= 0:
            raise RuntimeError("Shared backbone has no parameters; cannot enforce pixel encoder capacity cap")
        ratio = float(pixel_params) / float(backbone_params)
        if ratio > max_ratio:
            raise RuntimeError(
                "FirstHopPixelEncoder violates weak-expression cap: "
                f"{pixel_params} params / {backbone_params} backbone params = {ratio:.4%}, "
                f"max allowed = {max_ratio:.4%}"
            )

    def _install_backbone_hook(self) -> None:
        """Install a forward hook on a backbone DiT block to capture hidden states.

        Uses a non-invasive hook so that the backbone code (in the RAE repo)
        does not need any modification.
        """
        target_idx = self._align_layer_idx
        # DiT^DH backbone stores blocks as self.blocks (ModuleList)
        blocks = None
        for attr in ("blocks", "layers", "dit_blocks"):
            if hasattr(self.backbone, attr):
                blocks = getattr(self.backbone, attr)
                break
        if blocks is None:
            raise RuntimeError(
                "Cannot find backbone block list for alignment hook. "
                "Expected one of: backbone.blocks, backbone.layers, backbone.dit_blocks"
            )
        if target_idx >= len(blocks):
            raise ValueError(
                f"alignment.layer_idx={target_idx} but backbone only has {len(blocks)} blocks"
            )

        def _hook_fn(module, input, output):
            # DiT blocks typically output a tensor [B, N, D] or tuple.
            if isinstance(output, tuple):
                self._hooked_hidden = output[0]
            else:
                self._hooked_hidden = output

        self._hook_handle = blocks[target_idx].register_forward_hook(_hook_fn)
        print(
            f"[alignment] installed hook on backbone block[{target_idx}] "
            f"(out of {len(blocks)} blocks)",
            flush=True,
        )

    @property
    def g_pix(self) -> torch.Tensor:
        # Backward-compatible accessor used by training regularizer code.
        return self.g_pix_raw

    def _maybe_backbone_context(self):
        # Keep autograd enabled even when backbone params are frozen:
        # gradients must still flow to hop0 pixel forcing inputs.
        return nullcontext()

    def assert_decoder_frozen(self) -> None:
        for name, p in self.rae.named_parameters():
            if p.requires_grad:
                raise RuntimeError(f"Decoder param must be frozen but is trainable: {name}")

    def sigma_for_hop(self, hop_idx: torch.Tensor) -> torch.Tensor:
        return self.pair_v_std[hop_idx.long()].view(-1, 1, 1, 1)

    def gate_pix_value(self) -> torch.Tensor:
        # Keep forcing gate strictly positive with configurable non-zero floor.
        return self.pixel_gate_floor + F.softplus(self.g_pix_raw)

    def _apply_hop0_pixel_forcing(
        self,
        z_src: torch.Tensor,
        hop_idx: torch.Tensor,
        x_src_img: torch.Tensor | None,
    ) -> torch.Tensor:
        if x_src_img is None:
            return z_src

        if x_src_img.dim() == 3:
            x_src_img = x_src_img.unsqueeze(1)
        if x_src_img.dim() != 4:
            raise ValueError(f"Expected x_src_img in [B,1,H,W], got {tuple(x_src_img.shape)}")
        if x_src_img.shape[1] != 1:
            x_src_img = x_src_img[:, 0:1]

        hop0_mask = hop_idx.long() == 0
        if not torch.any(hop0_mask):
            return z_src

        z_forced = z_src.clone()
        x_hop0 = x_src_img[hop0_mask]
        pix_feat = self.pixel_encoder(x_hop0)
        z_forced[hop0_mask] = z_forced[hop0_mask] + self.gate_pix_value() * pix_feat
        return z_forced

    def predict_latent_step(
        self,
        z_src: torch.Tensor,
        t_src: torch.Tensor,
        t_dst: torch.Tensor,
        hop_idx: torch.Tensor,
        x_src_img: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        hop_idx = hop_idx.long()
        z_in = self._apply_hop0_pixel_forcing(z_src, hop_idx, x_src_img)

        with self._maybe_backbone_context():
            v_shared = self.backbone(z_in, t_src, t_dst=t_dst, hop_idx=hop_idx)
        v_hop = self.hop_residual_head(v_shared, hop_idx)
        v_total_raw = v_shared + v_hop

        dt = (t_dst - t_src).view(-1, 1, 1, 1)
        if self.target_normalize:
            sigma = self.sigma_for_hop(hop_idx)
            v_total = v_total_raw.float() * sigma.float()
            z_pred = z_src.float() + v_total * dt.float()
        else:
            z_pred = z_src + v_total_raw * dt

        pix_delta = z_in - z_src

        # --- iREPA alignment output ---
        align_proj = None
        if self.alignment_enabled and self._hooked_hidden is not None:
            # Ensure float32 for projector even if AMP produces float16 hidden states
            hooked = self._hooked_hidden.float()
            align_proj = self.alignment_projector(hooked)
            self._hooked_hidden = None  # clear for next call

        return {
            "z_pred": z_pred,
            "z_in": z_in,
            "v_shared": v_shared,
            "v_hop": v_hop,
            "v_total_raw": v_total_raw,
            "gate_pix": self.gate_pix_value(),
            "gate_pix_raw": self.g_pix_raw,
            "lambda_hop": self.hop_residual_head.lambda_hop_value(),
            "lambda_hop_raw": self.hop_residual_head.lambda_hop_raw,
            "pix_delta_abs": pix_delta.abs().mean(),
            "v_hop_abs": v_hop.abs().mean(),
            "align_proj": align_proj,
        }

    def decode_crop(self, z: torch.Tensor, crop_size: int | None = None) -> torch.Tensor:
        x = self.rae.decode(z)
        if x.shape[1] > 1:
            x = x[:, 0:1]
        target = int(crop_size or self.image_size)
        h, w = x.shape[-2:]
        if h == target and w == target:
            return x
        if h < target or w < target:
            raise ValueError(f"Cannot center-crop decode output {(h, w)} to {(target, target)}")
        top = (h - target) // 2
        left = (w - target) // 2
        return x[:, :, top:top + target, left:left + target]


def build_backbone(cfg: Dict) -> Tuple[nn.Module, bool]:
    from src.pet_flow.models.pet_flow_dit_hop import PETFlowDiTDHHopAware

    backbone_cfg = cfg.get("backbone", {})
    model_cfg = backbone_cfg.get("model", {})
    data_cfg = backbone_cfg.get("data", {})
    model = PETFlowDiTDHHopAware(
        input_size=model_cfg.get("input_size", 16),
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
    state = ckpt.get("model_ema", ckpt.get("model"))
    if state is None:
        raise KeyError("Backbone checkpoint must contain `model` or `model_ema`")
    model.load_state_dict(state, strict=True)
    return model, bool(ckpt.get("target_normalize", False))


def build_rae(cfg: Dict, device: torch.device) -> nn.Module:
    from src.pet_flow.inference_pet_flow import load_rae_model

    rae = load_rae_model(cfg, device)
    return rae
