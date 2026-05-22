from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiffusionConfig
from .layers import RMSNorm, CrossAttention, SinusoidalTimestepEmbedding, SwiGLU


class LatentDenoiserLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim=dim, heads=heads, context_dim=dim, dropout=dropout)
        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, causal_mask: Optional[torch.Tensor]) -> torch.Tensor:
        y, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=causal_mask, need_weights=False)
        x = x + y
        x = x + self.cross_attn(self.norm2(x), condition, causal=True)
        x = x + self.ffn(self.norm3(x))
        return x


class LatentDenoiser(nn.Module):
    """Transformer denoiser over latent token sequences, predicting v for DDPM/DDIM."""

    def __init__(self, latent_dim: int, denoise_dim: int, layers: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, denoise_dim, bias=False)
        self.cond_proj = nn.Linear(latent_dim, denoise_dim, bias=False)
        self.time = SinusoidalTimestepEmbedding(denoise_dim)
        self.blocks = nn.ModuleList([
            LatentDenoiserLayer(denoise_dim, heads, denoise_dim * 4, dropout) for _ in range(layers)
        ])
        self.norm = RMSNorm(denoise_dim)
        self.out_proj = nn.Linear(denoise_dim, latent_dim, bias=False)
        self.feedback_bias = nn.Parameter(torch.zeros(denoise_dim), requires_grad=False)

    def add_feedback(self, feedback: torch.Tensor) -> None:
        # feedback: [B, L, denoise_dim] or [B, denoise_dim]. Use detached EMA-like cached bias for next forward.
        with torch.no_grad():
            fb = feedback.detach().mean(dim=tuple(range(feedback.dim() - 1)))
            if fb.numel() != self.feedback_bias.numel():
                if fb.numel() < self.feedback_bias.numel():
                    fb = F.pad(fb, (0, self.feedback_bias.numel() - fb.numel()))
                else:
                    fb = fb[: self.feedback_bias.numel()]
            self.feedback_bias.mul_(0.9).add_(0.1 * fb.to(self.feedback_bias.dtype))

    def _align_feedback(self, fb: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        fb = fb.to(device=z_ref.device, dtype=z_ref.dtype)
        if fb.dim() == 2:
            fb = fb[:, None, :]
        if fb.shape[1] < z_ref.shape[1]:
            # Token feedback is causal per position. Synthetic slots sit after tokens and
            # are never visible to token queries under the causal denoiser mask, so the
            # last prefix state is safe for slot positions.
            pad = fb[:, -1:, :].expand(fb.shape[0], z_ref.shape[1] - fb.shape[1], fb.shape[2])
            fb = torch.cat([fb, pad], dim=1)
        elif fb.shape[1] > z_ref.shape[1]:
            fb = fb[:, : z_ref.shape[1], :]
        return fb

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, direct_feedback: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        b, l, _ = z_t.shape
        route_feedback: Dict[str, torch.Tensor] = {}
        if isinstance(direct_feedback, dict):
            route_feedback = {k: self._align_feedback(v, z_t) for k, v in direct_feedback.items() if torch.is_tensor(v)}
        elif direct_feedback is not None:
            z_t = z_t + 0.01 * self._align_feedback(direct_feedback, z_t)
        x = self.in_proj(z_t)
        cond = self.cond_proj(condition)
        t_emb = self.time(t).to(dtype=x.dtype)[:, None, :]
        x = x + t_emb + self.feedback_bias.to(device=x.device, dtype=x.dtype)[None, None, :]
        causal_mask = torch.ones((l, l), device=x.device, dtype=torch.bool).triu(1)
        for idx, block in enumerate(self.blocks):
            # Spec route mapping: diff1 -> denoiser layer 1, diff2 -> layer 2.
            fb = route_feedback.get(f"diff{idx + 1}")
            if fb is not None:
                x = x + 0.01 * self.in_proj(fb).to(dtype=x.dtype)
            x = block(x, cond, causal_mask)
        return self.out_proj(self.norm(x))


class DDIMScheduler(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        # cosine-ish beta schedule, clipped for numerical stability.
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-5, 0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.timesteps = timesteps

    def coeffs(self, t: torch.Tensor, ndim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self.alpha_bars[t].clamp(1e-8, 1.0)
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return ab.sqrt().view(shape), (1.0 - ab).sqrt().view(shape)


class DiffusionTower(nn.Module):
    def __init__(self, cfg: DiffusionConfig, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.in_norm = RMSNorm(d_model)
        self.down = nn.Linear(d_model, cfg.latent_dim, bias=False)
        self.z_norm = RMSNorm(cfg.latent_dim)
        self.denoiser = LatentDenoiser(cfg.latent_dim, cfg.denoise_dim, cfg.denoiser_layers, cfg.denoiser_heads, dropout)
        self.scheduler = DDIMScheduler(cfg.ddpm_timesteps)
        self.slot_queries = nn.Parameter(torch.randn(cfg.idea_slots, cfg.latent_dim) * 0.02)
        self.slot_attn = CrossAttention(cfg.latent_dim, heads=max(1, min(8, cfg.latent_dim // 8)), context_dim=cfg.latent_dim, dropout=dropout)
        self.slot_context_out = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.z_norm(self.down(self.in_norm(h)))

    def make_slots(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return diagnostic global slots and a causal slot context for token path.

        A single full-sequence slot vector would leak future tokens if fed back into every
        token position during causal LM training. This method therefore builds prefix-only
        slot states with cumulative softmax statistics: token i only receives slot summaries
        computed from z[:, :i+1]. The final prefix slots are returned for diversity monitoring.
        """
        b, l, d = z.shape
        q = self.slot_queries.to(dtype=z.dtype, device=z.device)  # [S,D]
        # Prefix softmax for each slot over token positions. Use prefix-cummax only:
        # a full-sequence max is numerically tempting but creates a hidden future-token
        # dependency once finite precision/clamping enters the path.
        scores = torch.einsum("bld,sd->bls", z, q) / (d ** 0.5)
        prefix_max = scores.cummax(dim=1).values.detach()
        weights = (scores - prefix_max).exp().clamp_max(1e4)
        denom = weights.cumsum(dim=1).clamp_min(1e-6)  # [B,L,S]
        numer = (weights[..., None] * z[:, :, None, :]).cumsum(dim=1)  # [B,L,S,D]
        prefix_slots = numer / denom[..., None]
        token_to_slot = torch.einsum("bld,blsd->bls", z, prefix_slots) / (d ** 0.5)
        slot_mix = F.softmax(token_to_slot.float(), dim=-1).to(z.dtype)
        causal_context = torch.einsum("bls,blsd->bld", slot_mix, prefix_slots)
        global_slots = prefix_slots[:, -1] + q[None]
        return global_slots, self.slot_context_out(causal_context)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = torch.randn_like(z0) if noise is None else noise
        ca, sa = self.scheduler.coeffs(t, z0.dim())
        return ca.to(z0.dtype) * z0 + sa.to(z0.dtype) * noise

    def v_target(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ca, sa = self.scheduler.coeffs(t, z0.dim())
        return ca.to(z0.dtype) * noise - sa.to(z0.dtype) * z0

    def predict_x0_from_v(self, z_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ca, sa = self.scheduler.coeffs(t, z_t.dim())
        ca = ca.to(z_t.dtype)
        sa = sa.to(z_t.dtype)
        return ca * z_t - sa * v

    def ddim_step(self, z_t: torch.Tensor, t: int, t_next: int, condition: torch.Tensor) -> torch.Tensor:
        b = z_t.shape[0]
        tt = torch.full((b,), t, device=z_t.device, dtype=torch.long)
        v = self.denoiser(z_t, tt, condition)
        x0 = self.predict_x0_from_v(z_t, tt, v)
        ab_t = self.scheduler.alpha_bars[t].to(device=z_t.device, dtype=z_t.dtype).clamp(1e-8, 1.0)
        ab_next = self.scheduler.alpha_bars[t_next].to(device=z_t.device, dtype=z_t.dtype).clamp(1e-8, 1.0) if t_next >= 0 else z_t.new_tensor(1.0)
        eps = (z_t - ab_t.sqrt() * x0) / (1.0 - ab_t).sqrt().clamp_min(1e-6)
        if t_next < 0:
            return x0
        return ab_next.sqrt() * x0 + (1.0 - ab_next).sqrt() * eps

    def reverse_sample(self, z0_anchor: torch.Tensor, condition: torch.Tensor, t_start: int, steps: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        steps = steps or self.cfg.ddim_steps
        b = z0_anchor.shape[0]
        noise = torch.randn_like(z0_anchor)
        t0 = min(max(int(t_start), 1), self.cfg.ddpm_timesteps - 1)
        z = self.q_sample(z0_anchor, torch.full((b,), t0, device=z0_anchor.device, dtype=torch.long), noise)
        schedule = torch.linspace(t0, 0, steps + 1, device=z.device).long().tolist()
        prev_delta = None
        prev_speed = None
        exit_step = steps
        for i in range(steps):
            t = int(schedule[i])
            t_next = int(schedule[i + 1]) - 1
            z_next = self.ddim_step(z, t, t_next, condition)
            delta = (z_next - z).abs().mean()
            if prev_delta is not None:
                accel = (delta - prev_delta).abs() / z0_anchor.detach().abs().mean().clamp_min(1e-4)
                prev_speed = accel
                if accel.item() < self.cfg.adaptive_exit_threshold and i >= 2:
                    z = z_next
                    exit_step = i + 1
                    break
            prev_delta = delta
            z = z_next
        return z, {"diff_exit_step": z.new_tensor(float(exit_step)), "diff_acceleration": (prev_speed if prev_speed is not None else z.new_tensor(0.0))}

    def branch_merge(self, branches: torch.Tensor, anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # branches: [B, K, L, D]. Merge causally per position: token i must not
        # receive a branch weight computed from positions > i.
        b, k, l, d = branches.shape
        pos = torch.arange(1, l + 1, device=branches.device, dtype=branches.dtype)[None, None, :, None]
        anchor_n = F.normalize(anchor[:, None], dim=-1)
        branch_n = F.normalize(branches, dim=-1)
        local_similarity = (branch_n * anchor_n).sum(-1)  # [B,K,L]
        similarity = local_similarity.cumsum(dim=-1) / torch.arange(1, l + 1, device=branches.device, dtype=branches.dtype)[None, None, :]
        if k > 1:
            prefix_mean = branches.cumsum(dim=2) / pos
            flat = F.normalize(prefix_mean, dim=-1)  # [B,K,L,D]
            sim_mat = torch.einsum("bkld,bmld->bklm", flat, flat)
            diversity = 1.0 - (sim_mat.sum(-1) - 1.0) / max(1, k - 1)  # [B,K,L]
        else:
            diversity = torch.zeros_like(similarity)
        score = self.cfg.branch_similarity_weight * similarity + self.cfg.branch_diversity_weight * diversity
        weights = F.softmax(score, dim=1)  # [B,K,L]
        merged = (branches * weights[..., None]).sum(dim=1)
        return merged, weights

    def forward(self, h: torch.Tensor, *, training_mode: bool, force_full_reverse: bool = False, direct_feedback: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        z_token = self.project(h)
        z_slots, slot_context = self.make_slots(z_token)
        z_token = z_token + slot_context
        z0 = torch.cat([z_token, z_slots], dim=1)
        condition = z0
        b = z0.shape[0]

        if self.training and training_mode:
            choices = torch.tensor(self.cfg.train_t_start_choices, device=h.device, dtype=torch.long)
            idx = torch.randint(0, choices.numel(), (b,), device=h.device)
            t = choices[idx].clamp(max=self.cfg.ddpm_timesteps - 1)
            noise = torch.randn_like(z0)
            z_t = self.q_sample(z0, t, noise)
            v_pred = self.denoiser(z_t, t, condition, direct_feedback=direct_feedback)
            target = self.v_target(z0, t, noise)
            ddpm_loss = F.mse_loss(v_pred.float(), target.float())
            if force_full_reverse or self.cfg.train_lm_path == "full_reverse":
                # Full reverse per sample t_start is costly; use max t for a deterministic shared path in training.
                z_diff, aux = self.reverse_sample(z0, condition, int(t.max().item()), self.cfg.ddim_steps)
            else:
                z_diff = self.predict_x0_from_v(z_t, t, v_pred)
                aux = {"diff_exit_step": z0.new_tensor(1.0), "diff_acceleration": z0.new_tensor(0.0)}
            branch_weights = z0.new_ones((b, 1))
        else:
            branch_list = []
            exits = []
            accels = []
            for t_start in self.cfg.infer_t_start:
                out, aux = self.reverse_sample(z0, condition, t_start, self.cfg.ddim_steps)
                branch_list.append(out)
                exits.append(aux["diff_exit_step"])
                accels.append(aux["diff_acceleration"])
            branches = torch.stack(branch_list, dim=1)
            z_diff, branch_weights = self.branch_merge(branches, z0)
            ddpm_loss = z0.new_zeros(())
            aux = {"diff_exit_step": torch.stack(exits).float().mean(), "diff_acceleration": torch.stack(accels).float().mean()}

        return {
            "z_diff": z_diff[:, : h.shape[1]],
            "z_diff_full": z_diff,
            "z0": z0[:, : h.shape[1]],
            "z0_full": z0,
            "z_slots": z0[:, h.shape[1] :],
            "ddpm_loss": ddpm_loss,
            "branch_weights": branch_weights,
            **aux,
        }
