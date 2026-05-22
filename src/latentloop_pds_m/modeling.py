from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import LatentLoopConfig
from .layers import TransformerBlock, RMSNorm, pairwise_cosine_offdiag_loss
from .dual_stream import DualStreamBlock
from .memory import LatentDenseMemoryGraph
from .dense_feedback import DenseFeedbackRouter


class LatentLoopTransformerPDSM(nn.Module):
    """From-scratch LatentLoop-Transformer-PDS+M.

    Implements the 18-layer autoregressive backbone with dual-stream blocks at layers 5/11,
    first-block-only LDMG interaction, and Layer-14 dense feedback. Qwen wrapper mode is
    intentionally not implemented in this code path.
    """

    def __init__(self, cfg: LatentLoopConfig):
        super().__init__()
        if cfg.training_mode != "from_scratch":
            raise ValueError("This implementation intentionally supports only training_mode='from_scratch'.")
        self.cfg = cfg
        tcfg = cfg.transformer
        self.embed_tokens = nn.Embedding(tcfg.vocab_size, tcfg.d_model)
        self.layers = nn.ModuleList()
        self.dual_indices = set(tcfg.dual_stream_layers)
        for i in range(tcfg.n_layers):
            if i in self.dual_indices:
                self.layers.append(DualStreamBlock(cfg, tcfg.d_model, is_primary=(i == tcfg.dual_stream_layers[0])))
            else:
                self.layers.append(TransformerBlock(tcfg.d_model, tcfg.n_q_heads, tcfg.n_kv_heads, tcfg.ffn_dim, tcfg.max_seq_len, tcfg.rope_theta, tcfg.dropout))
        self.final_norm = RMSNorm(tcfg.d_model)
        self.lm_head = nn.Linear(tcfg.d_model, tcfg.vocab_size, bias=False)
        if tcfg.tie_lm_head:
            self.lm_head.weight = self.embed_tokens.weight
        self.memory = LatentDenseMemoryGraph(cfg.memory)
        self.feedback = DenseFeedbackRouter(cfg.dense_feedback, tcfg.d_model, cfg.diffusion.latent_dim) if cfg.dense_feedback.enabled else None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        memory = sum(p.numel() for p in self.memory.parameters())
        return {"total": total, "trainable": trainable, "memory": memory}

    @staticmethod
    def _append_aux(aux: Dict[str, List[torch.Tensor]], key: str, value: torch.Tensor) -> None:
        aux.setdefault(key, []).append(value if value.dim() else value[None])

    def _dispatch_feedback_cached(self, fb: Dict[str, torch.Tensor]) -> None:
        # Explicit legacy path only when cached_next_forward is enabled. It is detached by design.
        if not fb:
            return
        dual_layers = list(self.cfg.transformer.dual_stream_layers)
        for route, value in fb.items():
            if not isinstance(value, torch.Tensor) or route.startswith("dense_"):
                continue
            if route.startswith("block1") and len(dual_layers) >= 1:
                block = self.layers[dual_layers[0]]
                if isinstance(block, DualStreamBlock):
                    block.add_feedback(value, route.split(".", 1)[1])
            elif route.startswith("block2") and len(dual_layers) >= 2:
                block = self.layers[dual_layers[1]]
                if isinstance(block, DualStreamBlock):
                    block.add_feedback(value, route.split(".", 1)[1])
            elif route == "memory.graph":
                self.memory.graph.apply_feedback(value[:, None, :], strength=0.001)
            elif route == "memory.edges":
                with torch.no_grad():
                    delta = value.detach().mean().clamp(-0.01, 0.01)
                    self.memory.graph.edge_feedback_bias.mul_(0.9).add_(delta.to(self.memory.graph.edge_feedback_bias.dtype))

    def _run_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        global_step: int,
        *,
        direct_feedback: Optional[Dict[str, torch.Tensor]] = None,
        collect_feedback: bool = True,
        use_cached_feedback: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
        h = self.embed_tokens(input_ids)
        aux: Dict[str, List[torch.Tensor]] = {}
        feedback_routes: Dict[str, torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DualStreamBlock):
                out = layer(h, memory=self.memory if layer.is_primary else None, global_step=global_step, direct_feedback=direct_feedback)
                h = out["h_out"]
                for k, v in out.items():
                    if torch.is_tensor(v) and k not in {"h_out", "z_fused", "z_anchor", "z_memory_query"}:
                        self._append_aux(aux, k, v)
                    if k in {"z_fused", "z_anchor", "z_memory_query"} and torch.is_tensor(v):
                        aux.setdefault(k, []).append(v)
            else:
                if self.training and self.cfg.transformer.gradient_checkpointing:
                    h = checkpoint(lambda x, mod=layer: mod(x, attention_mask=attention_mask), h, use_reentrant=False)
                else:
                    h = layer(h, attention_mask=attention_mask)
            if self.feedback is not None and collect_feedback and i == self.cfg.transformer.dense_feedback_layer:
                fb = self.feedback(h, global_step)
                feedback_routes = {k: v for k, v in fb.items() if torch.is_tensor(v) and not k.startswith("dense_")}
                if use_cached_feedback and self.cfg.dense_feedback.cached_next_forward:
                    self._dispatch_feedback_cached(fb)
                for k, v in fb.items():
                    if torch.is_tensor(v) and k.startswith("dense_"):
                        self._append_aux(aux, k, v)
        logits = self.lm_head(self.final_norm(h))
        return logits, aux, feedback_routes

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, global_step: int = 0) -> Dict[str, torch.Tensor]:
        if self.feedback is not None and self.cfg.dense_feedback.recompute_current_forward:
            # Pass 1 computes Layer-14 feedback. Pass 2 recomputes the current forward with differentiable feedback.
            _, probe_aux, fb = self._run_layers(input_ids, attention_mask, global_step, collect_feedback=True, use_cached_feedback=False)
            logits, aux, _ = self._run_layers(input_ids, attention_mask, global_step, direct_feedback=fb, collect_feedback=False)
            # Keep feedback regularization/diagnostics from the probe pass in the optimized loss.
            for k, vals in probe_aux.items():
                if k.startswith("dense_"):
                    aux.setdefault(k, []).extend(vals)
        else:
            logits, aux, fb = self._run_layers(input_ids, attention_mask, global_step, collect_feedback=True, use_cached_feedback=True)
        losses = self.compute_losses(logits, labels, aux, global_step)
        metrics = self.aggregate_metrics(aux)
        return {"logits": logits, **losses, "aux": metrics}

    def aggregate_metrics(self, aux: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        for k, vals in aux.items():
            if k in {"z_fused", "z_anchor", "z_memory_query"}:
                continue
            flat = []
            for v in vals:
                if torch.is_tensor(v):
                    flat.append(v.float().mean())
            if flat:
                metrics[k] = torch.stack(flat).mean()
        gates = [v for k, vals in aux.items() if k == "ctrl_gates" for v in vals]
        if gates:
            g = torch.cat([x.reshape(-1, 3).float() for x in gates], dim=0)
            metrics["gate_inf_mean"] = g[:, 0].mean()
            metrics["gate_write_mean"] = g[:, 1].mean()
            metrics["gate_read_mean"] = g[:, 2].mean()
            metrics["gate_write_var"] = g[:, 1].var(unbiased=False)
        return metrics

    def compute_losses(self, logits: torch.Tensor, labels: Optional[torch.Tensor], aux: Dict[str, List[torch.Tensor]], global_step: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg.losses
        total = logits.new_zeros(())
        loss_dict: Dict[str, torch.Tensor] = {}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            chunks = max(1, self.cfg.optim.chunked_ce_size)
            ce_num = shift_logits.new_zeros((), dtype=torch.float32)
            ce_den = shift_logits.new_zeros((), dtype=torch.float32)
            for start in range(0, shift_logits.shape[1], chunks):
                end = min(shift_logits.shape[1], start + chunks)
                labels_chunk = shift_labels[:, start:end].reshape(-1)
                valid = (labels_chunk != -100).sum().to(dtype=torch.float32, device=shift_logits.device)
                if valid.item() == 0:
                    continue
                ce = F.cross_entropy(shift_logits[:, start:end].reshape(-1, shift_logits.size(-1)).float(), labels_chunk, ignore_index=-100, reduction="sum")
                ce_num = ce_num + ce
                ce_den = ce_den + valid
            lm_loss = ce_num / ce_den.clamp_min(1.0)
        else:
            lm_loss = logits.new_zeros(())
        loss_dict["lm_loss"] = lm_loss
        total = total + cfg.lm_loss * lm_loss

        def mean_aux(name: str) -> torch.Tensor:
            vals = aux.get(name, [])
            if not vals:
                return logits.new_zeros(())
            return torch.stack([v.float().mean() for v in vals]).mean().to(logits.device)

        ddpm_loss = mean_aux("diff_ddpm_loss")
        anchor = self._anchor_loss(aux, logits)
        slot_div = self._slot_loss(aux, logits)
        # Minimize negative entropy = maximize entropy under positive weight.
        exit_entropy_loss = -(mean_aux("loop_loop_exit_entropy") + mean_aux("inter_interaction_entropy"))
        shortcut = mean_aux("loop_shortcut_consistency_loss")
        dense_reg = mean_aux("dense_feedback_reg_loss")
        controller_balance = self._controller_balance(aux, logits)
        mem_surprise = mean_aux("memory_surprise_loss")
        graph_sparse = mean_aux("graph_sparsity_loss")
        mem_stability = mean_aux("memory_stability_loss")

        anchor_frac = min(1.0, float(global_step) / max(1, cfg.anchor_warmup_steps))
        anchor_w = cfg.anchor_loss_initial + (cfg.anchor_loss - cfg.anchor_loss_initial) * anchor_frac
        ctrl_w = cfg.controller_balance_loss * max(0.0, 1.0 - float(global_step) / max(1, cfg.controller_balance_warmup_steps))
        weighted = {
            "ddpm_loss": cfg.ddpm_loss * ddpm_loss,
            "anchor_loss": anchor_w * anchor,
            "slot_diversity_loss": cfg.slot_diversity_loss * slot_div,
            "exit_entropy_loss": cfg.exit_entropy_loss * exit_entropy_loss,
            "shortcut_consistency_loss": cfg.shortcut_consistency_loss * shortcut,
            "dense_feedback_reg_loss": cfg.dense_feedback_reg_loss * dense_reg,
            "controller_balance_loss": ctrl_w * controller_balance,
            "memory_surprise_loss": cfg.memory_surprise_loss * mem_surprise,
            "graph_sparsity_loss": cfg.graph_sparsity_loss * graph_sparse,
            "memory_stability_loss": cfg.memory_stability_loss * mem_stability,
        }
        raw = {
            "ddpm_loss_raw": ddpm_loss,
            "anchor_loss_raw": anchor,
            "slot_diversity_loss_raw": slot_div,
            "exit_entropy_raw": -exit_entropy_loss,
            "shortcut_consistency_loss_raw": shortcut,
            "dense_feedback_reg_loss_raw": dense_reg,
            "controller_balance_loss_raw": controller_balance,
            "memory_surprise_loss_raw": mem_surprise,
            "graph_sparsity_loss_raw": graph_sparse,
            "memory_stability_loss_raw": mem_stability,
        }
        for name, value in weighted.items():
            total = total + value
            loss_dict[name] = value
        loss_dict.update(raw)
        loss_dict["total_loss_without_shortcut"] = total - weighted["shortcut_consistency_loss"]
        loss_dict["loss"] = total
        return loss_dict

    def _anchor_loss(self, aux: Dict[str, List[torch.Tensor]], ref: torch.Tensor) -> torch.Tensor:
        zf = aux.get("z_fused", [])
        za = aux.get("z_anchor", [])
        if not zf or not za:
            return ref.new_zeros(())
        vals = []
        for f, a in zip(zf, za):
            vals.append(1.0 - F.cosine_similarity(f.mean(dim=1).float(), a.mean(dim=1).float(), dim=-1).mean())
        return torch.stack(vals).mean().to(ref.device)

    def _slot_loss(self, aux: Dict[str, List[torch.Tensor]], ref: torch.Tensor) -> torch.Tensor:
        slots = aux.get("diff_z_slots", [])
        if not slots:
            return ref.new_zeros(())
        return torch.stack([pairwise_cosine_offdiag_loss(s.float()) for s in slots]).mean().to(ref.device)

    def _controller_balance(self, aux: Dict[str, List[torch.Tensor]], ref: torch.Tensor) -> torch.Tensor:
        gates = aux.get("ctrl_gates", [])
        if not gates:
            return ref.new_zeros(())
        g = torch.cat([x.reshape(-1, 3).float() for x in gates], dim=0)
        return ((g - 1.0 / 3.0) ** 2).mean().to(ref.device)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 1.0, top_k: int = 50, global_step: Optional[int] = None) -> torch.Tensor:
        self.eval()
        ids = input_ids
        if global_step is None:
            # Inference should use the trained-time schedule state, not step 0; otherwise
            # dense feedback and write-threshold annealing remain in cold-start mode.
            global_step = max(self.cfg.dense_feedback.warmup_steps, self.cfg.losses.anchor_warmup_steps, self.cfg.memory.write_threshold_decay_steps)
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.cfg.transformer.max_seq_len :]
            out = self(ids_cond, labels=None, global_step=int(global_step))
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                logits = mask.scatter(1, idx, vals)
            probs = F.softmax(logits.float(), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
