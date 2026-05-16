"""V18 — RAE decoder LoRA finetune helper.

Reuses RAE's own LinearWithLoRA implementation (RAE/src/utils/lora.py) to avoid
divergent code paths. Only adds two pieces:

1. wrap_decoder_with_lora(rae, cfg): scoped LoRA injection limited to the
   last_n_blocks of `rae.decoder.decoder_layers` (a ViT-MAE decoder block
   list). RAE encoder LoRA is loaded separately via inference_pet_flow.py
   and is orthogonal to this.

2. compute_kl_pullback_loss(rae_lora, rae_frozen, z_gt, crop_size):
   MSE(decode_lora(z_gt) - decode_frozen(z_gt)) to prevent decoder drift.

Architecture facts confirmed by reading the RAE source (RAE/RAE/src/stage1/):

- rae.encoder = DINOv2 (already LoRA-tuned in RAE pretraining; checkpoint
  loaded from rae_cfg.checkpoint_path). We DO NOT touch encoder here.
- rae.decoder = GeneralDecoder, contains:
    rae.decoder.decoder_embed       (nn.Linear)
    rae.decoder.decoder_layers      (nn.ModuleList[ViTMAELayer], 12 layers for ViTB)
    rae.decoder.decoder_norm        (nn.LayerNorm)
    rae.decoder.decoder_pred        (nn.Linear)   # the patch->pixel projection
- Each ViTMAELayer contains:
    layer.attention.attention.query   (nn.Linear)
    layer.attention.attention.key     (nn.Linear)
    layer.attention.attention.value   (nn.Linear)
    layer.attention.output.dense      (nn.Linear)
    layer.intermediate.dense          (nn.Linear)   == fc1
    layer.output.dense                (nn.Linear)   == fc2

LoRA contract:
  * On wrap, ALL rae params are frozen (encoder + decoder), then per-layer
    LinearWithLoRA wrapping unlocks only lora_A/lora_B in the last N decoder
    layers.
  * LinearWithLoRA does zero-init B -> step 0 output identical to V7 best.pt.
  * RAE encoder LoRA params (loaded from RAE pretrained ckpt) stay frozen.

For the KL pull-back loss we build a SECOND, fully frozen RAE via the
existing load_rae_model() function (re-imported from
src.pet_flow.inference_pet_flow). This second copy has its own encoder
LoRA loaded but never touched.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _resolve_attr_path(obj: object, dotted: str) -> object:
    """Walk `a.b.c` style path; raises with helpful message on miss."""
    cur = obj
    # Allow paths starting with 'rae.' or just '.decoder...'
    if dotted.startswith("rae."):
        dotted = dotted[len("rae."):]
    for piece in dotted.split("."):
        if piece == "":
            continue
        if not hasattr(cur, piece):
            raise AttributeError(
                f"decoder_lora.target_root path failed at '{piece}' (full path '{dotted}'). "
                f"Inspect `print(rae)` and update target_root in yaml. "
                f"Expected for ViT-MAE decoder: 'rae.decoder.decoder_layers'."
            )
        cur = getattr(cur, piece)
    return cur


def wrap_decoder_with_lora(
    rae: nn.Module,
    cfg: Dict,
) -> Tuple[List[nn.Parameter], List[str]]:
    """Wrap last_n_blocks of `rae`'s decoder with LoRA adapters.

    Uses RAE's native LinearWithLoRA (zero-init B -> step 0 output unchanged).

    Returns:
      (lora_params, wrapped_names) -- pass lora_params to optimizer as a new
      param group; wrapped_names goes to startup log for verification.
    """
    lora_cfg = cfg["training"].get("decoder_lora", {})
    if not bool(lora_cfg.get("enabled", False)):
        return [], []

    # Import RAE's own LoRA tooling. We do this lazily so non-LoRA runs
    # don't pay the import cost.
    from src.utils.lora import LinearWithLoRA  # type: ignore

    target_root = str(lora_cfg.get("target_root", "rae.decoder.decoder_layers"))
    last_n = int(lora_cfg.get("last_n_blocks", 2))
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.0))
    init_zero = bool(lora_cfg.get("init_scale_zero", True))
    target_keywords = tuple(lora_cfg.get("target_keywords", (
        "attention.attention.query",
        "attention.attention.key",
        "attention.attention.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    )))

    # Step 1: freeze ALL rae params (encoder + decoder). This is defensive;
    # RAE encoder LoRA params loaded from pretrained ckpt remain frozen here
    # -- V18 only touches decoder, never encoder.
    for p in rae.parameters():
        p.requires_grad_(False)

    # Step 2: resolve decoder block list
    blocks = _resolve_attr_path(rae, target_root)
    if not hasattr(blocks, "__len__"):
        raise RuntimeError(
            f"decoder_lora.target_root '{target_root}' resolved to "
            f"{type(blocks).__name__}, which has no len(). Expected nn.ModuleList."
        )
    if last_n > len(blocks):
        raise ValueError(
            f"decoder_lora.last_n_blocks={last_n} > total blocks={len(blocks)}"
        )

    if not init_zero:
        # RAE's LinearWithLoRA always zero-inits B (see lora.py:34). If user
        # really wants non-zero B init, they have to monkey-patch reset_parameters.
        # We don't support that here; warn loudly.
        print(
            f"[decoder_lora] WARN: init_scale_zero=false ignored. "
            f"RAE LinearWithLoRA hard-codes zero-init for B. "
            f"V18 step 0 output WILL equal V7 best.pt output exactly.",
            flush=True,
        )

    # Step 3: wrap matching Linear in the last N blocks
    lora_params: List[nn.Parameter] = []
    wrapped_names: List[str] = []

    for block_idx in range(len(blocks) - last_n, len(blocks)):
        block = blocks[block_idx]
        block_wraps = []
        for sub_name, sub_mod in list(block.named_modules()):
            if not isinstance(sub_mod, nn.Linear):
                continue
            if not any(kw in sub_name for kw in target_keywords):
                continue
            # locate parent for in-place replacement
            parent_path = sub_name.rsplit(".", 1)
            if len(parent_path) == 1:
                parent = block
                attr = parent_path[0]
            else:
                parent = block.get_submodule(parent_path[0])
                attr = parent_path[1]
            adapter = LinearWithLoRA(sub_mod, rank=rank, alpha=alpha, dropout=dropout)
            adapter.to(sub_mod.weight.device)
            setattr(parent, attr, adapter)
            lora_params.append(adapter.lora_A)
            lora_params.append(adapter.lora_B)
            wrapped_names.append(f"decoder_layers[{block_idx}].{sub_name}")
            block_wraps.append(sub_name)
        if not block_wraps:
            print(
                f"[decoder_lora] WARN block[{block_idx}]: no Linear matched "
                f"keywords {target_keywords}. Skipped.",
                flush=True,
            )

    if not lora_params:
        raise RuntimeError(
            "decoder_lora is enabled but ZERO Linear modules were wrapped. "
            "Inspect `print(rae.decoder.decoder_layers[-1])` and update "
            "decoder_lora.target_keywords in yaml. Expected matches per ViTMAELayer: "
            "attention.attention.{query,key,value}, attention.output.dense, "
            "intermediate.dense, output.dense (6 Linear per layer)."
        )

    # Sanity: confirm only the wrapped LoRA params have requires_grad=True on rae
    trainable_rae = [n for n, p in rae.named_parameters() if p.requires_grad]
    expected_count = len(lora_params)
    actual_count = len(trainable_rae)
    if actual_count != expected_count:
        sample = trainable_rae[:5]
        raise RuntimeError(
            f"[decoder_lora] consistency check failed: rae has {actual_count} "
            f"trainable params but we expected {expected_count} LoRA params. "
            f"First 5 trainable: {sample}. "
            f"This means freeze step failed or RAE has non-LoRA trainable params elsewhere."
        )

    print(
        f"[decoder_lora] wrapped {len(wrapped_names)} Linear modules across "
        f"last {last_n} decoder layers of {target_root}; "
        f"trainable LoRA params (A+B per Linear) = {sum(p.numel() for p in lora_params):,}; "
        f"first 3 wrapped: {wrapped_names[:3]}",
        flush=True,
    )
    return lora_params, wrapped_names


def build_frozen_reference_decoder(cfg: Dict, device: torch.device) -> nn.Module:
    """Build a SECOND, fully frozen RAE for KL pull-back.

    Called BEFORE wrap_decoder_with_lora() so this copy has no V18 LoRA
    (only the RAE-pretrained encoder LoRA, which is loaded from
    rae_cfg.checkpoint_path inside load_rae_model). This copy is never
    updated.
    """
    from src.pet_flow.inference_pet_flow import load_rae_model  # type: ignore

    rae_frozen = load_rae_model(cfg, device)
    for p in rae_frozen.parameters():
        p.requires_grad_(False)
    rae_frozen.eval()
    n_trainable = sum(1 for _, p in rae_frozen.named_parameters() if p.requires_grad)
    print(
        f"[decoder_lora] built frozen reference RAE for KL pull-back. "
        f"trainable params = {n_trainable} (must be 0)",
        flush=True,
    )
    if n_trainable != 0:
        raise RuntimeError("frozen reference RAE has trainable params; check load_rae_model")
    return rae_frozen


def compute_kl_pullback_loss(
    rae_lora: nn.Module,
    rae_frozen: nn.Module,
    z_gt: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    """L = MSE(decode_lora(z_gt) - decode_frozen(z_gt)).

    Both decoders see the same z_gt (a GT latent). Gradient flows only
    through rae_lora (rae_frozen is in eval+no_grad).

    The crop logic mirrors PETFlowDiTFirstHop.decode_crop's center-crop step.
    """
    x_lora = rae_lora.decode(z_gt)
    if x_lora.shape[1] > 1:
        x_lora = x_lora[:, 0:1]
    h, w = x_lora.shape[-2:]
    if h > crop_size:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        x_lora = x_lora[:, :, top:top + crop_size, left:left + crop_size]
    elif h < crop_size or w < crop_size:
        raise ValueError(
            f"compute_kl_pullback_loss: decoded shape {(h, w)} < crop {crop_size}"
        )

    with torch.no_grad():
        x_frozen = rae_frozen.decode(z_gt)
        if x_frozen.shape[1] > 1:
            x_frozen = x_frozen[:, 0:1]
        if x_frozen.shape[-2:] != (crop_size, crop_size):
            top = (x_frozen.shape[-2] - crop_size) // 2
            left = (x_frozen.shape[-1] - crop_size) // 2
            x_frozen = x_frozen[:, :, top:top + crop_size, left:left + crop_size]

    return (x_lora - x_frozen).pow(2).mean()
