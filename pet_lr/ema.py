"""Exponential Moving Average (EMA) for model weights.

EMA maintains a shadow copy of model parameters updated as:
    ema_param = decay * ema_param + (1 - decay) * param

Adapted from RAE/src/pet_flow/utils/ema.py for PET_LatentResidual.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Usage::

        model = MyModel()
        ema = EMA(model, decay=0.9999)

        # During training
        optimizer.step()
        ema.update()

        # For evaluation / checkpoint saving
        with ema.average_parameters():
            evaluate(model)
            save_checkpoint(model, ...)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        self.shadow_params: list[torch.Tensor] = [
            p.clone().detach() for p in model.parameters() if p.requires_grad
        ]
        self._collected_params: list[torch.Tensor] | None = None

    def _get_decay(self) -> float:
        if self.step < self.update_after_step:
            return 0.0
        s = self.step - self.update_after_step
        if self.warmup_steps > 0 and s < self.warmup_steps:
            return self.decay * (s / self.warmup_steps)
        return self.decay

    @torch.no_grad()
    def update(self) -> None:
        self.step += 1
        if self.step < self.update_after_step:
            return
        if self.step % self.update_every != 0:
            return
        decay = self._get_decay()
        for shadow, param in zip(
            self.shadow_params,
            (p for p in self.model.parameters() if p.requires_grad),
        ):
            shadow.lerp_(param.data, 1.0 - decay)

    def copy_to_model(self) -> None:
        for shadow, param in zip(
            self.shadow_params,
            (p for p in self.model.parameters() if p.requires_grad),
        ):
            param.data.copy_(shadow)

    def store_model_params(self) -> None:
        self._collected_params = [
            p.clone().detach() for p in self.model.parameters() if p.requires_grad
        ]

    def restore_model_params(self) -> None:
        if self._collected_params is None:
            raise RuntimeError("No stored parameters to restore")
        for stored, param in zip(
            self._collected_params,
            (p for p in self.model.parameters() if p.requires_grad),
        ):
            param.data.copy_(stored)
        self._collected_params = None

    class _Ctx:
        def __init__(self, ema: "EMA"):
            self.ema = ema

        def __enter__(self):
            self.ema.store_model_params()
            self.ema.copy_to_model()
            return self.ema.model

        def __exit__(self, *args):
            self.ema.restore_model_params()

    def average_parameters(self) -> _Ctx:
        """Context manager: temporarily swap model weights with EMA weights."""
        return self._Ctx(self)

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.step = state_dict["step"]
        self.decay = state_dict.get("decay", self.decay)
        self.shadow_params = state_dict["shadow_params"]

    def to(self, device: torch.device) -> "EMA":
        self.shadow_params = [p.to(device) for p in self.shadow_params]
        return self
