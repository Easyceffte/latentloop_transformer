from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig
from .layers import RMSNorm, SwiGLU, entropy_from_probs
from .kernels import chunked_exact_topk_retrieval


class MemoryDistiller(nn.Module):
    """Distill raw latent state into a long-term-memory friendly insight vector.

    The distiller is intentionally small for 4070-local runs. It implements the
    ReMe-inspired idea that online/long-term writes should not store raw z_fused
    directly; they should store a filtered semantic delta conditioned on what the
    memory already retrieved.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            RMSNorm(dim),
        )

    def forward(self, z_query: torch.Tensor, z_fused: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        if z_query.shape != z_fused.shape or z_query.shape != retrieved.shape:
            raise ValueError(f"distiller inputs must share shape, got {z_query.shape}, {z_fused.shape}, {retrieved.shape}")
        return self.net(torch.cat([z_query, z_fused, retrieved], dim=-1))


class SimplifiedGraphOverview(nn.Module):
    def __init__(self, n_nodes: int, dim: int):
        super().__init__()
        self.nodes = nn.Parameter(torch.randn(n_nodes, dim) * 0.02)
        # Store logits, not probabilities. Zero init means dense uniform graph before sparsity pressure.
        self.edge_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.query = nn.Linear(dim, dim, bias=False)
        self.feedback_proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer("activation_count", torch.zeros(n_nodes), persistent=True)
        self.register_buffer("feedback_bias", torch.zeros(dim), persistent=True)
        self.register_buffer("edge_feedback_bias", torch.zeros(()), persistent=True)

    def forward(self, query_vec: torch.Tensor, direct_node_feedback: Optional[torch.Tensor] = None, direct_edge_feedback: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # query_vec: [B,D] or [B,L,D]. Output overview keeps the leading shape.
        original_shape = query_vec.shape[:-1]
        q_flat = query_vec.reshape(-1, query_vec.shape[-1])
        q = F.normalize(self.query(q_flat), dim=-1)

        nodes_raw = self.nodes + self.feedback_bias.to(device=self.nodes.device, dtype=self.nodes.dtype)[None, :]
        nodes = F.normalize(nodes_raw, dim=-1)
        src_logits = torch.matmul(q, nodes.t())
        src_probs = F.softmax(src_logits, dim=-1)

        edge_logits = self.edge_logits + self.edge_feedback_bias.to(device=self.edge_logits.device, dtype=self.edge_logits.dtype)
        trans = F.softmax(edge_logits, dim=-1)
        tgt_probs = torch.matmul(src_probs, trans)
        overview_flat = torch.matmul(tgt_probs, self.nodes)
        # Differentiable current-forward memory feedback must be token-local. Do
        # not fold it into global nodes/edge logits, which would leak future token
        # summaries into every position. Cached graph mutation remains a separate
        # explicit path.
        if direct_node_feedback is not None:
            fb = direct_node_feedback.reshape(-1, direct_node_feedback.shape[-1]).to(overview_flat.dtype)
            if fb.shape[0] != overview_flat.shape[0]:
                fb = fb[:1].expand_as(overview_flat) if fb.shape[0] == 1 else fb[: overview_flat.shape[0]]
            overview_flat = overview_flat + 0.001 * self.feedback_proj(fb)
        if direct_edge_feedback is not None:
            efb = direct_edge_feedback.reshape(-1, direct_edge_feedback.shape[-1]).to(overview_flat.dtype)
            if efb.shape[0] != overview_flat.shape[0]:
                efb = efb[:1].expand_as(overview_flat) if efb.shape[0] == 1 else efb[: overview_flat.shape[0]]
            overview_flat = overview_flat + 0.001 * self.feedback_proj(efb)
        overview = overview_flat.reshape(*original_shape, self.nodes.shape[-1])
        if self.training:
            with torch.no_grad():
                self.activation_count.add_(src_probs.detach().sum(dim=0).to(self.activation_count.device))
        return {
            "overview": overview,
            "src_probs": src_probs.reshape(*original_shape, -1),
            "tgt_probs": tgt_probs.reshape(*original_shape, -1),
            "edge_probs": trans,
        }

    def apply_feedback(self, h_deep: torch.Tensor, strength: float) -> torch.Tensor:
        # Legacy cached feedback path, disabled by default. Kept for explicit ablations only.
        fb = self.feedback_proj(h_deep.mean(dim=tuple(range(h_deep.dim() - 1)))).detach()
        with torch.no_grad():
            self.feedback_bias.mul_(0.9).add_(strength * fb.to(self.feedback_bias.dtype))
        return fb

    def sparsity_loss(self) -> torch.Tensor:
        # A real sparsity objective: low entropy rows + small non-top transition mass.
        probs = F.softmax(self.edge_logits, dim=-1)
        entropy = entropy_from_probs(probs).mean() / max(1.0, torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device)))
        top = probs.max(dim=-1).values.mean()
        return entropy + (1.0 - top)

    def significant_edge_ratio(self, threshold: float = 0.1) -> torch.Tensor:
        return (F.softmax(self.edge_logits, dim=-1) > threshold).float().mean()


class LatentDenseMemoryGraph(nn.Module):
    """Continuous latent memory matrix + graph overview + sparse top-k retrieval.

    For n_slots above exact_threshold, retrieval uses per-query LSH bucket tables and never
    materializes a [B*L, N] candidate mask. LSH rebuild is full-bank and no_grad; normal
    training steps score only per-query candidate rows.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.key_proj = nn.Linear(cfg.d_mem, cfg.query_dim, bias=False)
        self.query_proj = nn.Linear(cfg.d_mem, cfg.query_dim, bias=False)
        self.values = nn.Parameter(torch.randn(cfg.n_slots, cfg.d_mem) * 0.02)
        self.ive = nn.Sequential(RMSNorm(cfg.d_mem), SwiGLU(cfg.d_mem, cfg.d_mem * 4), RMSNorm(cfg.d_mem))
        self.reconstruct = nn.Sequential(RMSNorm(cfg.d_mem), nn.Linear(cfg.d_mem, cfg.d_mem), nn.GELU(), nn.Linear(cfg.d_mem, cfg.d_mem))
        self.distiller = MemoryDistiller(cfg.d_mem) if getattr(cfg, "use_memory_distiller", True) else None
        self.graph = SimplifiedGraphOverview(cfg.graph_nodes, cfg.d_mem)
        self.register_buffer("retrieval_count", torch.zeros(cfg.n_slots), persistent=True)
        self.register_buffer("last_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("lsh_planes", torch.randn(cfg.lsh_tables, cfg.query_dim, cfg.lsh_bits), persistent=False)
        self.register_buffer("lsh_codes", torch.empty(0, dtype=torch.long), persistent=False)
        self._bucket_maps: Optional[List[Dict[int, torch.Tensor]]] = None
        self._lsh_built = False
        self._last_lsh_rebuild_step = -1

    def current_write_threshold(self, global_step: int) -> float:
        drops = global_step // max(1, self.cfg.write_threshold_decay_steps)
        return max(self.cfg.write_threshold_target, self.cfg.write_threshold_initial - drops * self.cfg.write_threshold_decay_amount)

    def overview(self, query_vec: torch.Tensor, direct_node_feedback: Optional[torch.Tensor] = None, direct_edge_feedback: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self.graph(query_vec, direct_node_feedback=direct_node_feedback, direct_edge_feedback=direct_edge_feedback)

    def _keys_for_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.key_proj(values)

    def _all_keys_no_grad(self, chunk: int = 65536) -> torch.Tensor:
        keys = []
        with torch.no_grad():
            for start in range(0, self.cfg.n_slots, chunk):
                end = min(self.cfg.n_slots, start + chunk)
                keys.append(self.key_proj(self.values[start:end]).detach().float().cpu())
        return torch.cat(keys, dim=0)

    def _rebuild_lsh(self) -> None:
        keys = F.normalize(self._all_keys_no_grad(), dim=-1)
        planes_cpu = self.lsh_planes.detach().float().cpu()
        codes = []
        maps: List[Dict[int, torch.Tensor]] = []
        powers = (2 ** torch.arange(self.cfg.lsh_bits, dtype=torch.long))
        for table in range(self.cfg.lsh_tables):
            bits = (keys @ planes_cpu[table] > 0).long()
            code = (bits * powers).sum(dim=-1).long()
            codes.append(code)
            buckets: Dict[int, List[int]] = defaultdict(list)
            for idx, c in enumerate(code.tolist()):
                buckets[int(c)].append(idx)
            maps.append({k: torch.tensor(v, dtype=torch.long) for k, v in buckets.items()})
        self.lsh_codes = torch.stack(codes, dim=0).to(self.values.device)
        self._bucket_maps = maps
        self._lsh_built = True

    def _candidate_matrix_lsh(self, q: torch.Tensor, max_candidates: int) -> torch.Tensor:
        # q: [Nq,Q]. Returns [Nq,C] candidate ids. No dense [Nq,N] allocation.
        if (not self._lsh_built) or self._bucket_maps is None:
            self._rebuild_lsh()
        qn = F.normalize(q.detach().float(), dim=-1)
        planes = self.lsh_planes.detach().float().to(qn.device)
        powers = (2 ** torch.arange(self.cfg.lsh_bits, device=qn.device, dtype=torch.long))
        rows = []
        need = max(max_candidates, self.cfg.top_k)
        for row in range(qn.shape[0]):
            chunks = []
            for table in range(self.cfg.lsh_tables):
                bits = (qn[row] @ planes[table] > 0).long()
                code = int((bits * powers).sum().item())
                cand_cpu = self._bucket_maps[table].get(code) if self._bucket_maps is not None else None
                if cand_cpu is not None and cand_cpu.numel() > 0:
                    chunks.append(cand_cpu.to(q.device))
            if chunks:
                cand = torch.unique(torch.cat(chunks))
            else:
                cand = torch.empty(0, device=q.device, dtype=torch.long)
            if cand.numel() < need:
                filler = torch.randint(0, self.cfg.n_slots, (need - cand.numel(),), device=q.device)
                cand = torch.cat([cand, filler])
            if cand.numel() > max_candidates:
                perm = torch.randperm(cand.numel(), device=q.device)[:max_candidates]
                cand = cand[perm]
            rows.append(cand[:max_candidates])
        return torch.stack(rows, dim=0)

    def retrieve(self, z_query: torch.Tensor, top_k: Optional[int] = None, global_step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        top_k = top_k or self.cfg.top_k
        b, l, d = z_query.shape
        q = self.query_proj(z_query).reshape(b * l, self.cfg.query_dim)
        q = F.normalize(q, dim=-1)
        if self.cfg.use_lsh and self.cfg.n_slots > self.cfg.exact_threshold:
            if global_step is not None and self._lsh_built and self.cfg.rebuild_lsh_every > 0:
                if self._last_lsh_rebuild_step < 0 or (global_step - self._last_lsh_rebuild_step) >= self.cfg.rebuild_lsh_every:
                    self._rebuild_lsh()
                    self._last_lsh_rebuild_step = int(global_step)
            max_c = max(top_k, min(self.cfg.lsh_max_candidates, self.cfg.n_slots))
            idx_chunks = []
            val_chunks = []
            chunk = max(1, int(getattr(self.cfg, "lsh_query_chunk", 32)))
            for start in range(0, q.shape[0], chunk):
                q_chunk = q[start : start + chunk]
                cand = self._candidate_matrix_lsh(q_chunk, max_c)  # [M,C]
                cand_values = self.values[cand]
                cand_keys = F.normalize(self._keys_for_values(cand_values), dim=-1)
                scores = torch.einsum("nq,ncq->nc", q_chunk, cand_keys) / (self.cfg.query_dim ** 0.5)
                kk = min(top_k, scores.shape[-1])
                vals_c, local_idx = scores.topk(kk, dim=-1)
                idx_c = cand.gather(1, local_idx)
                idx_chunks.append(idx_c)
                val_chunks.append(vals_c)
            idx = torch.cat(idx_chunks, dim=0)
            vals = torch.cat(val_chunks, dim=0)
            weights = F.softmax(vals, dim=-1)
            gathered = self.values[idx]
            retrieved = torch.einsum("nk,nkd->nd", weights.to(gathered.dtype), gathered).view(b, l, d)
            retrieved = self.ive(retrieved)
        else:
            backend = getattr(self.cfg, "retrieval_backend", "auto")
            use_chunked = backend in {"auto", "exact_chunked"}
            if use_chunked:
                exact = chunked_exact_topk_retrieval(
                    values=self.values,
                    query_proj=self.query_proj,
                    key_proj=self.key_proj,
                    z_query=z_query,
                    top_k=top_k,
                    query_dim=self.cfg.query_dim,
                    query_chunk=int(getattr(self.cfg, "exact_query_chunk", 128)),
                    slot_chunk=int(getattr(self.cfg, "exact_slot_chunk", 4096)),
                )
                idx = exact["indices"].reshape(b * l, -1)
                vals = exact["scores"].reshape(b * l, -1)
                weights = exact["weights"].reshape(b * l, -1)
                retrieved = exact["retrieved_raw"]
                retrieved = self.ive(retrieved)
            else:
                keys_n = F.normalize(self._keys_for_values(self.values), dim=-1)
                scores = q @ keys_n.t() / (self.cfg.query_dim ** 0.5)
                kk = min(top_k, scores.shape[-1])
                vals, idx = scores.topk(kk, dim=-1)
                weights = F.softmax(vals, dim=-1)
                gathered = self.values[idx]  # [B*L,K,D]
                retrieved = torch.einsum("nk,nkd->nd", weights.to(gathered.dtype), gathered).view(b, l, d)
                retrieved = self.ive(retrieved)
        if self.training:
            with torch.no_grad():
                flat = idx.detach().reshape(-1)
                self.retrieval_count.index_add_(0, flat.to(self.retrieval_count.device), torch.ones_like(flat, dtype=self.retrieval_count.dtype, device=self.retrieval_count.device))
                self.last_indices = idx.detach().reshape(-1).unique().to(self.last_indices.device)
        concentration = weights[:, : min(10, weights.shape[-1])].sum(-1).mean()
        return {"retrieved": retrieved, "indices": idx.view(b, l, -1), "weights": weights.view(b, l, -1), "top10_concentration": concentration}

    def write_losses_and_maybe_update(self, z_fused: torch.Tensor, retrieved: torch.Tensor, g_write: torch.Tensor, global_step: int) -> Dict[str, torch.Tensor]:
        recon = self.reconstruct(retrieved)
        surprise_per_token = (recon.float() - z_fused.float()).pow(2).mean(dim=-1)
        surprise = surprise_per_token.mean()
        theta = z_fused.new_tensor(self.current_write_threshold(global_step))
        write_mask = surprise_per_token > theta
        write_mask_f = write_mask[:, :, None].to(z_fused.dtype)
        skip_rate = 1.0 - write_mask.float().mean()
        # g_write: [B] or [B,L]
        if g_write.dim() == 1:
            gw = g_write[:, None, None].to(z_fused.dtype)
        else:
            gw = g_write[:, :, None].to(z_fused.dtype)
        # The surprise threshold is not merely diagnostic: it gates both the
        # stability penalty and optional online memory movement.
        delta = write_mask_f * gw * surprise_per_token[:, :, None].to(z_fused.dtype) * (z_fused - retrieved)
        stability = delta.float().pow(2).mean()
        surprise_loss = F.relu(theta - surprise)
        if self.cfg.online_update and self.training and self.last_indices.numel() > 0 and write_mask.any():
            with torch.no_grad():
                if self.distiller is not None:
                    insight = self.distiller(z_fused.detach(), z_fused.detach(), retrieved.detach())
                    active = insight.detach()[write_mask]
                else:
                    active = z_fused.detach()[write_mask]
                target = active.mean(dim=0) if active.numel() else z_fused.detach().mean(dim=(0, 1))
                ids = self.last_indices[: min(self.last_indices.numel(), 4096)].to(self.values.device)
                strength = float((g_write.detach()[write_mask] if g_write.dim() > 1 else g_write.detach()).float().mean().clamp(0, 1))
                self.values[ids].lerp_(target[None].to(self.values.dtype), self.cfg.memory_lr * strength)
        return {"surprise": surprise, "memory_surprise_loss": surprise_loss, "memory_stability_loss": stability, "write_skip_rate": skip_rate}

    def hebbian_update(self, indices: torch.Tensor) -> None:
        return None

    def apply_forgetting(self) -> None:
        with torch.no_grad():
            # Decay graph confidence toward an uninformative prior and prune tiny
            # probabilities. Multiplying all probabilities by gamma before a
            # later softmax is ineffective; operating in probability space and
            # renormalizing gives observable forgetting semantics.
            probs = F.softmax(self.graph.edge_logits, dim=-1)
            uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
            probs = self.cfg.edge_decay_gamma * probs + (1.0 - self.cfg.edge_decay_gamma) * uniform
            probs = torch.where(probs < self.cfg.edge_prune_threshold, torch.zeros_like(probs), probs)
            row_sum = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sum > 0, probs / row_sum.clamp_min(1e-8), uniform)
            self.graph.edge_logits.copy_(torch.log(probs.clamp_min(1e-8)))

    def reusable_slot_mask(self) -> torch.Tensor:
        avg = self.retrieval_count.float().mean().clamp_min(1.0)
        return self.retrieval_count < (avg * self.cfg.low_freq_reuse_ratio)
