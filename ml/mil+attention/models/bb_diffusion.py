# models/bb_diffusion.py
from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn

from .scheduler import BrownianBridgeScheduler


class BrownianBridgeDiffusion(nn.Module):
    """
    Wraps:
      - denoising network
      - bridge scheduler

    Training:
      predict epsilon from x_t conditioned on x0 and t

    Inference:
      iterative reverse bridge from x0 to y
    """
    def __init__(self, denoiser: nn.Module, scheduler: BrownianBridgeScheduler):
        super().__init__()
        self.denoiser = denoiser
        self.scheduler = scheduler

    def forward_train(self, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> Dict[str, Any]:
        x_t, noise = self.scheduler.q_sample(x0, y, t)
        eps_pred = self.denoiser(x_t, x0, t)
        y_pred = self.scheduler.predict_y_from_eps(x_t, x0, t, eps_pred)

        return {
            "x_t": x_t,
            "noise": noise,
            "eps_pred": eps_pred,
            "y_pred": y_pred,
        }

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        num_steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Reverse bridge sampling starting from x0.
        """
        B = x0.shape[0]
        device = x0.device

        T = self.scheduler.num_steps if num_steps is None else int(num_steps)
        T = min(T, self.scheduler.num_steps)

        # initialize near final bridge state
        t_init = torch.full((B,), T, device=device, dtype=torch.long)
        x_t, _ = self.scheduler.q_sample(x0, x0, t_init, noise=torch.randn_like(x0))

        for t_int in range(T, 0, -1):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            eps_pred = self.denoiser(x_t, x0, t)
            y_pred = self.scheduler.predict_y_from_eps(x_t, x0, t, eps_pred)
            x_t = self.scheduler.p_sample_step(x_t, x0, t_int, y_pred, eta=eta)

        return x_t.clamp(0.0, 1.0)