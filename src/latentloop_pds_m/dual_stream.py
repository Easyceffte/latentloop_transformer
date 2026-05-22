from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LatentLoopConfig
from .diffusion import DiffusionTower
from .loop import RecurrentLoopTower
from .layers import RMSNorm, binary_entropy


class InteractionDepthController(nn.Module):
    def __init__(self, latent_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.score = nn.Sequential(
            RMSNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.score(delta.mean(dim=1)).squeeze(-1)


class BidirectionalInteraction(nn.Module):
    def __init__(self, latent_dim: int, heads: int, cfg):
        super().__init__()
        self.cfg = cfg
        # Use causal self/cross interaction in latent space to preserve LM prefix validity.
        self.loop_from_diff = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.diff_from_loop = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.loop_gate = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
        self.diff_gate = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
        self.idc = InteractionDepthController(latent_dim, cfg.adaptive_exit_threshold)
        self.fuse = nn.Linear(latent_dim * 2, latent_dim, bias=False)
        self.stability = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1), nn.Sigmoid())

    def forward(self, z_diff: torch.Tensor, z_loop: torch.Tensor) -> Dict[str, torch.Tensor]:
        max_rounds = self.cfg.train_rounds if self.training else self.cfg.infer_max_rounds
        min_rounds = self.cfg.train_rounds if self.training else self.cfg.infer_min_rounds
        l = z_diff.shape[1]
        causal_block = torch.ones((l, l), device=z_diff.device, dtype=torch.bool).triu(1)  # MHA: True = blocked
        probs = []
        prev_delta = None
        prev_speed = None
        rounds_executed = max_rounds
        for r in range(max_rounds):
            lfd, _ = self.loop_from_diff(z_loop, z_diff, z_diff, attn_mask=causal_block, need_weights=False)
            gl = self.loop_gate(torch.cat([z_loop, lfd], dim=-1))
            z_loop = z_loop + gl * lfd
            dfl, _ = self.diff_from_loop(z_diff, z_loop, z_loop, attn_mask=causal_block, need_weights=False)
            gd = self.diff_gate(torch.cat([z_diff, dfl], dim=-1))
            z_diff = z_diff + gd * dfl
            delta = (z_diff - z_loop).abs()
            p_continue = self.idc(delta)
            probs.append(p_continue)
            speed = delta.mean()
            if prev_delta is not None:
                accel = (speed - prev_delta).abs() / (z_diff.detach().abs().mean().clamp_min(1e-4))
                prev_speed = accel
                if (not self.training) and r + 1 >= min_rounds and accel.item() < self.cfg.adaptive_exit_threshold:
                    rounds_executed = r + 1
                    break
            prev_delta = speed
        z_cat = torch.cat([z_diff, z_loop], dim=-1)
        z_fused = self.fuse(z_cat)
        stability = self.stability(torch.cat([z_diff.mean(dim=1), z_loop.mean(dim=1)], dim=-1)).mean()
        p = torch.stack(probs, dim=1) if probs else z_diff.new_zeros((z_diff.shape[0], 1))
        return {
            "z_diff": z_diff,
            "z_loop": z_loop,
            "z_fused": z_fused,
            "interaction_entropy": binary_entropy(p).mean(),
            "interaction_continue_prob": p.mean(),
            "interaction_rounds": z_fused.new_tensor(float(rounds_executed)),
            "interaction_acceleration": prev_speed if prev_speed is not None else z_fused.new_zeros(()),
            "stability_score": stability,
        }


class FlowController(nn.Module):
    def __init__(self, latent_dim: int, alpha_inf: float, alpha_write: float, alpha_read: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.alpha_inf = alpha_inf
        self.alpha_write = alpha_write
        self.alpha_read = alpha_read

    def forward(self, z_fused: torch.Tensor, m_overview: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Per-token gates avoid future-token leakage from sequence mean pooling.
        if m_overview.dim() == 2:
            m_overview = m_overview[:, None, :].expand_as(z_fused)
        logits = self.net(torch.cat([z_fused, m_overview], dim=-1))
        gates = F.softmax(logits, dim=-1)
        return {
            "gates": gates,
            "g_inf": gates[..., 0],
            "g_write": gates[..., 1],
            "g_read": gates[..., 2],
            "infer_scale": self.alpha_inf * gates[..., 0],
            "write_scale": self.alpha_write * gates[..., 1],
            "read_scale": self.alpha_read * gates[..., 2],
        }


class DualStreamBlock(nn.Module):
    def __init__(self, cfg: LatentLoopConfig, d_model: int, is_primary: bool):
        super().__init__()
        self.cfg = cfg
        self.is_primary = is_primary
        self.diffusion = DiffusionTower(cfg.diffusion, d_model, cfg.transformer.dropout)
        self.loop = RecurrentLoopTower(cfg.loop, d_model, cfg.transformer.dropout)
        self.interaction = BidirectionalInteraction(cfg.diffusion.latent_dim, cfg.loop.heads, cfg.interaction)
        self.controller = FlowController(cfg.diffusion.latent_dim, cfg.controller.alpha_inf, cfg.controller.alpha_write, cfg.controller.alpha_read)
        self.out_norm = RMSNorm(cfg.diffusion.latent_dim)
        self.up = nn.Linear(cfg.diffusion.latent_dim, d_model, bias=False)
        self.gate = nn.Sequential(nn.Linear(d_model + cfg.diffusion.latent_dim, d_model), nn.Sigmoid())

    def add_feedback(self, feedback: torch.Tensor, route: str) -> None:
        # Legacy cached route. Differentiable current-forward feedback is passed via forward(direct_feedback=...).
        if route.startswith("diff"):
            self.diffusion.denoiser.add_feedback(feedback)
        elif route.startswith("loop"):
            self.loop.add_feedback(feedback)

    @staticmethod
    def _feedback_map(direct_feedback: Optional[Dict[str, torch.Tensor]], prefix: str) -> Optional[Dict[str, torch.Tensor]]:
        if not direct_feedback:
            return None
        out: Dict[str, torch.Tensor] = {}
        for k, v in direct_feedback.items():
            if k.startswith(prefix) and torch.is_tensor(v):
                # block1.diff1 -> diff1, block2.loop2 -> loop2
                out[k.split('.')[-1]] = v
        return out or None

    @staticmethod
    def _feedback_exact(direct_feedback: Optional[Dict[str, torch.Tensor]], key: str) -> Optional[torch.Tensor]:
        if not direct_feedback:
            return None
        v = direct_feedback.get(key)
        return v if torch.is_tensor(v) else None

    def forward(self, h: torch.Tensor, memory=None, global_step: int = 0, direct_feedback: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        block_prefix = "block1" if self.is_primary else "block2"
        diff_fb = self._feedback_map(direct_feedback, f"{block_prefix}.diff")
        loop_fb = self._feedback_map(direct_feedback, f"{block_prefix}.loop")
        diff = self.diffusion(h, training_mode=True, direct_feedback=diff_fb)
        loop = self.loop(h, diff["z_diff"], diff["z0"], compute_shortcut=True, direct_feedback=loop_fb)
        inter = self.interaction(diff["z_diff"], loop["z_loop"])
        z_fused = inter["z_fused"]
        # Preserve the pre-memory latent state for Phase 1.5B memory-objective
        # training. The normal z_fused returned below may include memory read
        # residual on the primary dual-stream block; z_memory_query is the exact
        # token-local latent passed into memory.retrieve().
        z_memory_query = z_fused
        mem_losses: Dict[str, torch.Tensor] = {}
        if memory is not None and self.is_primary:
            node_fb = self._feedback_exact(direct_feedback, "memory.graph")
            edge_fb = self._feedback_exact(direct_feedback, "memory.edges")
            overview_pack = memory.overview(z_fused, direct_node_feedback=node_fb, direct_edge_feedback=edge_fb)
            m_overview = overview_pack["overview"]
        else:
            m_overview = z_fused.new_zeros(z_fused.shape)
        ctrl = self.controller(z_fused, m_overview)
        if memory is not None and self.is_primary:
            retr = memory.retrieve(z_fused, global_step=global_step)
            # Positionwise memory read. Do not cross-attend over retrieved states, because that leaks future tokens.
            read = retr["retrieved"]
            z_fused = z_fused + ctrl["read_scale"][..., None].to(z_fused.dtype) * read
            mem_losses.update(memory.write_losses_and_maybe_update(z_fused, retr["retrieved"], ctrl["write_scale"], global_step))
            mem_losses["top10_concentration"] = retr["top10_concentration"]
            mem_losses["graph_sparsity_loss"] = memory.graph.sparsity_loss()
            mem_losses["graph_edge_ratio"] = memory.graph.significant_edge_ratio()
            mem_losses["write_gate_var"] = ctrl["g_write"].float().var(unbiased=False)
        else:
            for k in ["surprise", "memory_surprise_loss", "memory_stability_loss", "write_skip_rate", "top10_concentration", "graph_sparsity_loss", "graph_edge_ratio", "write_gate_var"]:
                mem_losses[k] = z_fused.new_zeros(())
        z_out = ctrl["infer_scale"][..., None].to(z_fused.dtype) * self.out_norm(z_fused)
        delta_h = self.up(z_out)
        gate = self.gate(torch.cat([h, z_fused], dim=-1))
        h_out = h + gate * delta_h
        aux = {
            **{f"diff_{k}": v for k, v in diff.items() if torch.is_tensor(v) and k not in {"z_diff", "z0", "z_diff_full", "z0_full"}},
            **{f"loop_{k}": v for k, v in loop.items() if torch.is_tensor(v) and k != "z_loop"},
            **{f"inter_{k}": v for k, v in inter.items() if torch.is_tensor(v) and not k.startswith("z_")},
            **{f"ctrl_{k}": v for k, v in ctrl.items() if torch.is_tensor(v)},
            **mem_losses,
            "z_anchor": diff["z0"],
            "z_memory_query": z_memory_query,
            "z_fused": z_fused,
            "h_out": h_out,
        }
        return aux
