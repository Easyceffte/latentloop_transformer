from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _topk_merge(
    best_scores: Optional[torch.Tensor],
    best_indices: Optional[torch.Tensor],
    scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a new candidate score matrix into a running top-k.

    This is the fallback kernel used for local 4070 runs. It avoids materializing
    a full [num_queries, n_slots] tensor when n_slots grows beyond the current
    16K local setting. For 16K it is still fast enough; for 64K it keeps memory
    bounded by query_chunk * slot_chunk.
    """
    local_k = min(k, scores.shape[-1])
    s, j = scores.topk(local_k, dim=-1)
    idx = indices[j]
    if best_scores is None or best_indices is None:
        return s, idx
    merged_s = torch.cat([best_scores, s], dim=-1)
    merged_i = torch.cat([best_indices, idx], dim=-1)
    kk = min(k, merged_s.shape[-1])
    out_s, order = merged_s.topk(kk, dim=-1)
    out_i = merged_i.gather(1, order)
    return out_s, out_i


def chunked_exact_topk_retrieval(
    *,
    values: torch.Tensor,
    query_proj,
    key_proj,
    z_query: torch.Tensor,
    top_k: int,
    query_dim: int,
    query_chunk: int = 128,
    slot_chunk: int = 4096,
    scale: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Exact cosine top-k retrieval with bounded score memory.

    Args:
        values: [N, D] memory value table.
        query_proj/key_proj: projection modules. Gradients flow into both.
        z_query: [B, L, D] latent queries.
        top_k: number of slots to retrieve.
        query_dim: projected query/key dimension.
        query_chunk: number of token queries per score block.
        slot_chunk: number of memory rows per score block.
        scale: optional score scale; defaults to sqrt(query_dim).

    Returns:
        retrieved_raw: weighted sum before IVE, [B, L, D].
        indices: [B, L, K]
        weights: [B, L, K]
        scores: [B, L, K]
    """
    if values.dim() != 2:
        raise ValueError(f"values must be [N,D], got {tuple(values.shape)}")
    if z_query.dim() != 3:
        raise ValueError(f"z_query must be [B,L,D], got {tuple(z_query.shape)}")
    b, l, d = z_query.shape
    n_slots = values.shape[0]
    if n_slots <= 0:
        raise ValueError("memory values table is empty")
    kk = min(int(top_k), int(n_slots))
    query_chunk = max(1, int(query_chunk))
    slot_chunk = max(1, int(slot_chunk))
    score_scale = float(scale) if scale is not None else float(query_dim) ** 0.5

    q_all = query_proj(z_query.reshape(b * l, d))
    q_all = F.normalize(q_all, dim=-1)
    out_scores = []
    out_indices = []
    # Precompute keys once. 16K/64K local V3 intentionally keeps exact retrieval
    # to make query/key gradients observable; this tensor is modest on 4070.
    keys = F.normalize(key_proj(values), dim=-1)
    all_slot_indices = torch.arange(n_slots, device=values.device, dtype=torch.long)
    for q_start in range(0, q_all.shape[0], query_chunk):
        q = q_all[q_start : q_start + query_chunk]
        best_s = None
        best_i = None
        for s_start in range(0, n_slots, slot_chunk):
            s_end = min(n_slots, s_start + slot_chunk)
            scores = (q @ keys[s_start:s_end].t()) / score_scale
            idx_range = all_slot_indices[s_start:s_end]
            best_s, best_i = _topk_merge(best_s, best_i, scores, idx_range, kk)
        out_scores.append(best_s)
        out_indices.append(best_i)
    vals = torch.cat(out_scores, dim=0)
    idx = torch.cat(out_indices, dim=0)
    weights = F.softmax(vals, dim=-1)
    gathered = values[idx]
    retrieved_raw = torch.einsum("nk,nkd->nd", weights.to(gathered.dtype), gathered).view(b, l, d)
    return {
        "retrieved_raw": retrieved_raw,
        "indices": idx.view(b, l, kk),
        "weights": weights.view(b, l, kk),
        "scores": vals.view(b, l, kk),
    }
