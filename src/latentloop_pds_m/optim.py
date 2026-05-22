from __future__ import annotations

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_wsd_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int, stable_until_ratio: float = 0.8, min_lr_ratio: float = 0.1) -> LambdaLR:
    stable_until = int(total_steps * stable_until_ratio)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        if step < stable_until:
            return 1.0
        progress = min(1.0, float(step - stable_until) / max(1, total_steps - stable_until))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)
