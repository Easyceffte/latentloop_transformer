from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from .config import DenseFeedbackConfig
from .layers import RMSNorm


class DenseFeedbackRouter(nn.Module):
    """Layer-14 sparse feedback router.

    The default implementation emits differentiable, prefix-causal per-token feedback for
    same-forward recomputation. A detached cached path remains available only for explicit ablations.
    """

    ROUTES = [
        "block1.diff1",
        "block1.diff2",
        "block1.loop1",
        "block1.loop2",
        "block2.diff1",
        "block2.diff2",
        "block2.loop1",
        "block2.loop2",
        "memory.graph",
        "memory.edges",
    ]

    def __init__(self, cfg: DenseFeedbackConfig, d_model: int, target_dim: int):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(d_model)
        self.routes = self.ROUTES[: cfg.n_routes]
        self.proj = nn.ModuleDict({name.replace('.', '_'): nn.Linear(d_model, target_dim, bias=False) for name in self.routes})
        self.gate = nn.ModuleDict({name.replace('.', '_'): nn.Linear(d_model + target_dim, 1) for name in self.routes})
        self.target_tokens = nn.ParameterDict({name.replace('.', '_'): nn.Parameter(torch.zeros(target_dim)) for name in self.routes})

    def warmup_scale(self, global_step: int) -> float:
        return min(1.0, float(global_step) / max(1, self.cfg.warmup_steps))

    def forward(self, h_deep: torch.Tensor, global_step: int = 0) -> Dict[str, torch.Tensor]:
        # Prefix-causal pooling: feedback injected into token i may only depend on
        # Layer-14 states at positions <= i. Full-sequence mean pooling would leak
        # future tokens into the recomputed lower layers.
        h_norm = self.norm(h_deep)
        denom = torch.arange(1, h_norm.shape[1] + 1, device=h_norm.device, dtype=h_norm.dtype)[None, :, None]
        pooled = h_norm.cumsum(dim=1) / denom
        scale = self.warmup_scale(global_step)
        outs: Dict[str, torch.Tensor] = {}
        regs = []
        gate_means = []
        signal_means = []
        for route in self.routes:
            key = route.replace('.', '_')
            p = self.proj[key](pooled)
            target = self.target_tokens[key][None, None, :].expand_as(p).to(dtype=p.dtype, device=p.device)
            g = torch.sigmoid(self.gate[key](torch.cat([pooled, target], dim=-1)))
            fb = scale * g * p
            outs[route] = fb
            regs.append((g.float() ** 2).mean())
            gate_means.append(g.float().mean())
            signal_means.append(fb.float().abs().mean())
        outs["dense_feedback_reg_loss"] = torch.stack(regs).mean() if regs else h_deep.new_zeros(())
        outs["dense_feedback_gate_mean"] = torch.stack(gate_means).mean() if gate_means else h_deep.new_zeros(())
        outs["dense_feedback_signal_mean"] = torch.stack(signal_means).mean() if signal_means else h_deep.new_zeros(())
        return outs
