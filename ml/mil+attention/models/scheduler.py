# models/scheduler.py
from __future__ import annotations

import torch


class BrownianBridgeScheduler:
    """
    Simple Brownian-bridge-style scheduler between:
      x0 = condition/raw image
      y  = target H&E image

    Forward process:
      x_t = (1 - m_t) * x0 + m_t * y + sigma_t * eps

    where m_t grows from ~0 -> 1 and sigma_t is a schedule
    """
    def __init__(
        self,
        num_steps: int = 100,
        sigma_min: float = 1e-4,
        sigma_max: float = 0.05,
        device: str = "cpu",
    ):
        self.num_steps = int(num_steps)
        self.device = device

        t = torch.linspace(0.0, 1.0, self.num_steps + 1, device=device)

        # Bridge interpolation coefficient
        self.m = t  # linear bridge from x0 to y

        # Noise schedule: zero-ish at endpoints, max in middle
        bridge_shape = 4.0 * t * (1.0 - t)  # 0 at t=0,1 ; max at middle
        self.sigma = sigma_min + (sigma_max - sigma_min) * bridge_shape

    def to(self, device: torch.device):
        self.device = str(device)
        self.m = self.m.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Use 1..T-1 for training; endpoints are trivial
        return torch.randint(1, self.num_steps, (batch_size,), device=device)

    def q_sample(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ):
        if noise is None:
            noise = torch.randn_like(y)

        m_t = self.m[t].view(-1, 1, 1, 1)
        s_t = self.sigma[t].view(-1, 1, 1, 1)

        x_t = (1.0 - m_t) * x0 + m_t * y + s_t * noise
        return x_t, noise

    def predict_y_from_eps(
        self,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        m_t = self.m[t].view(-1, 1, 1, 1)
        s_t = self.sigma[t].view(-1, 1, 1, 1)

        denom = torch.clamp(m_t, min=1e-6)
        y_pred = (x_t - (1.0 - m_t) * x0 - s_t * eps_pred) / denom
        return y_pred.clamp(0.0, 1.0)

    def p_sample_step(
        self,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        t: int,
        y_pred: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Simple deterministic/stochastic reverse bridge step.
        """
        if t <= 0:
            return x0

        t_prev = t - 1

        m_prev = self.m[t_prev]
        s_prev = self.sigma[t_prev]

        x_prev = (1.0 - m_prev) * x0 + m_prev * y_pred

        if eta > 0 and s_prev > 0:
            x_prev = x_prev + eta * s_prev * torch.randn_like(x_prev)

        return x_prev.clamp(0.0, 1.0)