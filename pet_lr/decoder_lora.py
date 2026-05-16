"""V18 — RAE decoder LoRA finetune helper (SPEC / STUB FOR CODEX TO COMPLETE)

This file is a contract sketch. Codex MUST inspect the actual RAE decoder
structure on the GPU server before filling in the TODOs marked below.

Why this is a stub on the laptop side
-------------------------------------
The RAE module is loaded from `/home/qujiaxiang/project/RAE/code/RAE/` (path
configured in V7 yaml as `rae_root`). The decoder structure (block list,
attention/MLP layer naming) lives in that external repo and is NOT mirrored
in this repo. Without server access I cannot enumerate the actual `nn.Linear`
modules to wrap.

Codex Day-1 Phase A (server, ≤30 min):
1. Load V7 best.pt model. Run `print(model.rae)` and copy first 200 lines to
   review/0517/V18_decoder_lora/RAE_DECODER_STRUCTURE.txt (commit to gitee).
2. Identify the attribute path to the decoder's transformer block list
   (e.g. `model.rae.decoder.blocks` or `model.rae.dec.layers`).
   Update V18 yaml's `training.decoder_lora.target_root` accordingly.
3. For the last block (`blocks[-1]`), run `for n,m in blocks[-1].named_modules():
   print(n, type(m).__name__)` and copy that output to STRUCTURE.txt too.
4. Update `target_module_types` / `name_regex` in yaml so that the wrap pass
   below selects 4-8 Linear modules per block (q/k/v/out + mlp.fc1/fc2 is
   the safest default).

API contract for this file
--------------------------
- `wrap_decoder_with_lora(rae, cfg)` is the entry point called from
  `model_first_hop.py:PETFlowDiTFirstHop.__init__` when `training.decoder_lora.enabled=true`.
- It must:
    a. freeze ALL decoder params (rae.requires_grad_(False))
    b. enumerate target Linear modules in last_n_blocks
    c. wrap each with a LoRAAdapter
    d. set ONLY the LoRA A/B matrices to requires_grad=True
    e. return a list of LoRA params for the optimizer
- `compute_kl_pullback_loss(rae, rae_frozen, z_gt)` is called from train loop
  to compute `MSE(decode_lora(z_gt) - decode_frozen(z_gt))`.

Determinism contract
--------------------
- With `init_scale_zero=True` (default), LoRA B is zero-init so V18 step 0
  output == V7 best.pt output EXACTLY. This means warm-start can be verified
  by running 1 eval batch before training and confirming PSNR == V7 baseline.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Standard low-rank adapter: y = Wx + (B@A)@x * alpha/rank.

    Wraps an existing nn.Linear without changing its weight (frozen).
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale_zero: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(base, nn.Linear), f"LoRAAdapter only wraps nn.Linear, got {type(base)}"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f = base.in_features
        out_f = base.out_features
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        # A: in_f -> rank   (Kaiming-uniform init like standard LoRA)
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_f, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: rank -> out_f  (zero-init so B@A == 0 at step 0; preserves base output)
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.rank, dtype=base.weight.dtype))
        if not init_scale_zero:
            nn.init.normal_(self.lora_B, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path (frozen)
        out = self.base(x)
        # LoRA delta
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scale * delta

    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def _resolve_attr_path(obj: object, dotted: str) -> object:
    """Walk `a.b.c` style path; raises with helpful message on miss."""
    cur = obj
    for piece in dotted.split("."):
        if piece == "":
            continue
        if not hasattr(cur, piece):
            raise AttributeError(
                f"decoder_lora.target_root path failed at '{piece}' (full path '{dotted}'). "
                f"Codex must inspect `print(model.rae)` and update target_root in yaml."
            )
        cur = getattr(cur, piece)
    return cur


def wrap_decoder_with_lora(
    rae: nn.Module,
    cfg: Dict,
) -> Tuple[List[nn.Parameter], List[str]]:
    """Wrap last_n_blocks of `rae`'s decoder with LoRA adapters.

    Returns:
      (lora_params, wrapped_names) -- pass lora_params to optimizer as a new param group.
    """
    lora_cfg = cfg["training"].get("decoder_lora", {})
    if not bool(lora_cfg.get("enabled", False)):
        return [], []

    target_root = str(lora_cfg.get("target_root", "decoder.blocks"))
    last_n = int(lora_cfg.get("last_n_blocks", 2))
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.0))
    init_zero = bool(lora_cfg.get("init_scale_zero", True))
    target_types = tuple(lora_cfg.get("target_module_types", ["Linear"]))
    name_regex = lora_cfg.get("name_regex", "")
    name_re = re.compile(name_regex) if name_regex else None

    # First: freeze ALL decoder params (defensive; trainer should also do this)
    for p in rae.parameters():
        p.requires_grad_(False)

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

    lora_params: List[nn.Parameter] = []
    wrapped_names: List[str] = []

    for block_idx in range(len(blocks) - last_n, len(blocks)):
        block = blocks[block_idx]
        # collect candidates: walk (name, module) and pick by type + name regex
        candidates = []
        for sub_name, sub_mod in block.named_modules():
            type_name = type(sub_mod).__name__
            if type_name not in target_types:
                continue
            if name_re is not None and not name_re.search(sub_name):
                continue
            candidates.append((sub_name, sub_mod))

        if not candidates:
            print(
                f"[decoder_lora] WARN block[{block_idx}] no Linear matched "
                f"regex='{name_regex}' types={target_types}; LoRA skipped this block",
                flush=True,
            )
            continue

        for sub_name, sub_mod in candidates:
            # replace in-place: find parent module that owns sub_mod
            parent_path = sub_name.rsplit(".", 1)
            if len(parent_path) == 1:
                parent = block
                attr = parent_path[0]
            else:
                parent = block.get_submodule(parent_path[0])
                attr = parent_path[1]
            adapter = LoRAAdapter(
                base=sub_mod, rank=rank, alpha=alpha,
                dropout=dropout, init_scale_zero=init_zero,
            )
            setattr(parent, attr, adapter)
            lora_params.extend(adapter.lora_parameters())
            wrapped_names.append(f"block[{block_idx}].{sub_name}")

    if not lora_params:
        raise RuntimeError(
            "decoder_lora is enabled but ZERO Linear modules were wrapped. "
            "Codex must verify target_root / name_regex / target_module_types match the actual RAE decoder. "
            "Inspect with `for n,m in rae.named_modules(): print(n, type(m).__name__)` and update yaml."
        )

    print(
        f"[decoder_lora] wrapped {len(wrapped_names)} Linear modules across "
        f"last {last_n} blocks of {target_root}; "
        f"trainable LoRA params = {sum(p.numel() for p in lora_params)}",
        flush=True,
    )
    return lora_params, wrapped_names


def build_frozen_reference_decoder(rae_template: nn.Module, cfg: Dict, device: torch.device) -> nn.Module:
    """Build a SECOND, fully frozen copy of the RAE decoder for KL pull-back.

    Called once at boot, BEFORE LoRA wrapping. This copy is never wrapped or
    updated; it provides the reference `decode_pretrained(z_gt)` output for
    the KL loss.

    Codex implementation note:
      It is acceptable (and cheaper) to skip this if `decoder_kl_pullback.enabled=false`.
      If enabled, the simplest implementation is to re-call `build_rae(cfg, device)`
      a second time before wrapping the primary copy. This doubles RAE memory but
      RAE is small relative to backbone.
    """
    # Codex TODO: import build_rae from pet_lr.model_first_hop or pet_lr.model
    # from pet_lr.model_first_hop import build_rae as _build_rae  # circular; resolve at runtime
    raise NotImplementedError(
        "Codex must wire build_frozen_reference_decoder by calling build_rae(cfg, device) "
        "BEFORE wrap_decoder_with_lora() is invoked. Return value goes to the main model "
        "as `self.rae_frozen` and is set to .eval() + requires_grad_(False) permanently."
    )


def compute_kl_pullback_loss(
    rae_lora: nn.Module,
    rae_frozen: nn.Module,
    z_gt: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    """L = MSE(decode_lora(z_gt) - decode_frozen(z_gt)).

    Both decoders see the same z_gt (a GT latent from the validation manifold).
    Gradient flows only through rae_lora (rae_frozen is in eval+no_grad).
    """
    # decode_lora path: gradients flow through LoRA A/B
    x_lora = rae_lora.decode(z_gt)
    if x_lora.shape[1] > 1:
        x_lora = x_lora[:, 0:1]
    h, w = x_lora.shape[-2:]
    if h > crop_size:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        x_lora = x_lora[:, :, top:top + crop_size, left:left + crop_size]

    with torch.no_grad():
        x_frozen = rae_frozen.decode(z_gt)
        if x_frozen.shape[1] > 1:
            x_frozen = x_frozen[:, 0:1]
        if x_frozen.shape[-2:] != (crop_size, crop_size):
            top = (x_frozen.shape[-2] - crop_size) // 2
            left = (x_frozen.shape[-1] - crop_size) // 2
            x_frozen = x_frozen[:, :, top:top + crop_size, left:left + crop_size]

    return (x_lora - x_frozen).pow(2).mean()
