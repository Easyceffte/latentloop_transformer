from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoopConfig
from .layers import RMSNorm, CrossAttention, SwiGLU, binary_entropy


class LoopRefineLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.anchor_attn = CrossAttention(dim, heads=heads, context_dim=dim, dropout=dropout)
        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 4, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, z: torch.Tensor, anchor: torch.Tensor, causal_mask: Optional[torch.Tensor]) -> torch.Tensor:
        zn = self.norm1(z)
        y, _ = self.self_attn(zn, zn, zn, attn_mask=causal_mask, need_weights=False)
        z = z + y
        z = z + self.anchor_attn(self.norm2(z), anchor, causal=True)
        f = self.ffn(self.norm3(z))
        z = z + self.gate(z) * f
        return z


class RecurrentLoopTower(nn.Module):
    def __init__(self, cfg: LoopConfig, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.in_norm = RMSNorm(d_model)
        self.down = nn.Linear(d_model, cfg.latent_dim, bias=False)
        self.z_norm = RMSNorm(cfg.latent_dim)
        self.step_embed = nn.Embedding(cfg.embedding_limit, cfg.latent_dim)
        self.total_steps_embed = nn.Embedding(cfg.embedding_limit, cfg.latent_dim)
        self.layers = nn.ModuleList([LoopRefineLayer(cfg.latent_dim, cfg.heads, dropout) for _ in range(cfg.block_layers)])
        self.exit_gate = nn.Linear(cfg.latent_dim, 1)
        nn.init.zeros_(self.exit_gate.weight)
        nn.init.zeros_(self.exit_gate.bias)
        self.feedback_bias = nn.Parameter(torch.zeros(cfg.latent_dim), requires_grad=False)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.z_norm(self.down(self.in_norm(h)))

    def add_feedback(self, feedback: torch.Tensor) -> None:
        with torch.no_grad():
            fb = feedback.detach().mean(dim=tuple(range(feedback.dim() - 1)))
            self.feedback_bias.mul_(0.9).add_(0.1 * fb.to(self.feedback_bias.dtype))

    @staticmethod
    def _align_feedback(fb: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        fb = fb.to(device=z_ref.device, dtype=z_ref.dtype)
        if fb.dim() == 2:
            fb = fb[:, None, :]
        if fb.shape[1] < z_ref.shape[1]:
            pad = fb[:, -1:, :].expand(fb.shape[0], z_ref.shape[1] - fb.shape[1], fb.shape[2])
            fb = torch.cat([fb, pad], dim=1)
        elif fb.shape[1] > z_ref.shape[1]:
            fb = fb[:, : z_ref.shape[1], :]
        return fb

    def run_steps(self, z_init: torch.Tensor, anchor: torch.Tensor, steps: int, adaptive: bool, direct_feedback: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = z_init
        route_feedback: Dict[str, torch.Tensor] = {}
        if isinstance(direct_feedback, dict):
            route_feedback = {k: self._align_feedback(v, z) for k, v in direct_feedback.items() if torch.is_tensor(v)}
        elif direct_feedback is not None:
            z = z + 0.01 * self._align_feedback(direct_feedback, z)
        l = z.shape[1]
        causal_mask = torch.ones((l, l), device=z.device, dtype=torch.bool).triu(1)
        exit_probs = []
        cumulative = z.new_zeros((z.shape[0], 1))
        executed = steps
        total_idx = min(max(steps, 0), self.cfg.embedding_limit - 1)
        for step in range(steps):
            s_idx = min(step, self.cfg.embedding_limit - 1)
            cond = self.step_embed.weight[s_idx].to(z.dtype)[None, None, :] + self.total_steps_embed.weight[total_idx].to(z.dtype)[None, None, :]
            z = z + cond + self.feedback_bias.to(device=z.device, dtype=z.dtype)[None, None, :]
            for layer_idx, layer in enumerate(self.layers):
                # Spec route mapping: loop1 -> loop block layer 1, loop2 -> layer 2.
                fb = route_feedback.get(f"loop{layer_idx + 1}")
                if fb is not None:
                    z = z + 0.01 * fb
                z = layer(z, anchor, causal_mask)
            p_exit = torch.sigmoid(self.exit_gate(z.mean(dim=1)))
            exit_probs.append(p_exit.squeeze(-1))
            if adaptive:
                cumulative = 1.0 - (1.0 - cumulative) * (1.0 - p_exit)
                if bool((cumulative > self.cfg.exit_threshold).all()) and step + 1 >= 2:
                    executed = step + 1
                    break
        if exit_probs:
            p = torch.stack(exit_probs, dim=1)
            entropy = binary_entropy(p).mean()
            p_mean = p.mean()
        else:
            entropy = z.new_zeros(())
            p_mean = z.new_zeros(())
        return z, {"loop_exit_entropy": entropy, "loop_exit_prob_mean": p_mean, "loop_steps_executed": z.new_tensor(float(executed))}

    def shortcut_consistency(self, z_init: torch.Tensor, anchor: torch.Tensor, direct_feedback: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        z2, _ = self.run_steps(z_init, anchor, 2, adaptive=False, direct_feedback=direct_feedback)
        z4, _ = self.run_steps(z_init, anchor, 4, adaptive=False, direct_feedback=direct_feedback)
        z8, _ = self.run_steps(z_init, anchor, 8, adaptive=False, direct_feedback=direct_feedback)
        # KL is a per-token distributional shortcut target. Applying PyTorch
        # reduction="batchmean" directly to [B,L,D] divides only by B and
        # therefore multiplies the loss by L. Flatten to [B*L,D] so the raw
        # metric is comparable across sequence lengths and cannot dominate the
        # LM objective merely because seq_len=512.
        z2_flat = z2.float().reshape(-1, z2.shape[-1])
        z4_flat = z4.float().reshape(-1, z4.shape[-1])
        z8_flat = z8.detach().float().reshape(-1, z8.shape[-1])
        log2 = F.log_softmax(z2_flat, dim=-1)
        log4 = F.log_softmax(z4_flat, dim=-1)
        prob8 = F.softmax(z8_flat, dim=-1)
        kl2 = F.kl_div(log2, prob8, reduction="batchmean")
        kl4 = F.kl_div(log4, prob8, reduction="batchmean")
        kl = kl2 + kl4
        z8_det = z8.detach()
        cos2 = F.cosine_similarity(z2.mean(dim=1).float(), z8_det.mean(dim=1).float(), dim=-1).mean()
        cos4 = F.cosine_similarity(z4.mean(dim=1).float(), z8_det.mean(dim=1).float(), dim=-1).mean()
        mse2 = F.mse_loss(z2.float(), z8_det.float())
        mse4 = F.mse_loss(z4.float(), z8_det.float())
        metrics = {
            "shortcut_kl_normalized": kl,
            "shortcut_2v8_cosine": cos2,
            "shortcut_4v8_cosine": cos4,
            "shortcut_2v8_mse": mse2,
            "shortcut_4v8_mse": mse4,
        }
        return kl, metrics, z8

    def forward(self, h: torch.Tensor, z_seed: torch.Tensor, z_anchor: torch.Tensor, *, compute_shortcut: bool = True, direct_feedback: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        z0_loop = self.project(h)
        # Use diffusion seed as proposal and own projection as stabilizing anchor.
        z_init = 0.5 * z_seed + 0.5 * z0_loop
        if self.training:
            choices = self.cfg.train_steps_choices
            steps = int(choices[torch.randint(0, len(choices), ()).item()])
            z_loop, aux = self.run_steps(z_init, z_anchor, steps, adaptive=False, direct_feedback=direct_feedback)
            if compute_shortcut:
                sc_loss, sc_metrics, _ = self.shortcut_consistency(z_init, z_anchor, direct_feedback=direct_feedback)
            else:
                sc_loss = z_loop.new_zeros(())
                sc_metrics = {
                    "shortcut_kl_normalized": sc_loss,
                    "shortcut_2v8_cosine": sc_loss,
                    "shortcut_4v8_cosine": sc_loss,
                    "shortcut_2v8_mse": sc_loss,
                    "shortcut_4v8_mse": sc_loss,
                }
        else:
            z_loop, aux = self.run_steps(z_init, z_anchor, self.cfg.infer_steps, adaptive=True, direct_feedback=direct_feedback)
            sc_loss = z_loop.new_zeros(())
            sc_metrics = {
                "shortcut_kl_normalized": sc_loss,
                "shortcut_2v8_cosine": sc_loss,
                "shortcut_4v8_cosine": sc_loss,
                "shortcut_2v8_mse": sc_loss,
                "shortcut_4v8_mse": sc_loss,
            }
        return {
            "z_loop": z_loop,
            "z0_loop": z0_loop,
            "shortcut_consistency_loss": sc_loss,
            **sc_metrics,
            **aux,
        }
