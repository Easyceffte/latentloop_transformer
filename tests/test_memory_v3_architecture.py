from __future__ import annotations

import math
from pathlib import Path

import torch

from latentloop_pds_m.config import MemoryConfig
from latentloop_pds_m.memory import LatentDenseMemoryGraph
from latentloop_pds_m.kernels import chunked_exact_topk_retrieval
from scripts.train_memory_v3 import adjacent_contrastive_loss, graph_coactivation_loss, retrieval_metrics


def test_chunked_exact_retrieval_matches_full_topk():
    torch.manual_seed(0)
    cfg = MemoryConfig(n_slots=128, d_mem=32, query_dim=16, top_k=8, use_lsh=False, exact_query_chunk=5, exact_slot_chunk=31)
    mem = LatentDenseMemoryGraph(cfg)
    z = torch.randn(2, 7, 32)
    out = chunked_exact_topk_retrieval(
        values=mem.values,
        query_proj=mem.query_proj,
        key_proj=mem.key_proj,
        z_query=z,
        top_k=8,
        query_dim=16,
        query_chunk=5,
        slot_chunk=31,
    )
    q = torch.nn.functional.normalize(mem.query_proj(z.reshape(-1, 32)), dim=-1)
    k = torch.nn.functional.normalize(mem.key_proj(mem.values), dim=-1)
    vals, idx = (q @ k.t() / math.sqrt(16)).topk(8, dim=-1)
    assert torch.equal(out["indices"].reshape(-1, 8), idx)
    assert torch.allclose(out["scores"].reshape(-1, 8), vals, atol=1e-6)


def test_adjacent_contrastive_has_gradients():
    torch.manual_seed(1)
    cfg = MemoryConfig(n_slots=256, d_mem=32, query_dim=16, top_k=16, use_lsh=False)
    mem = LatentDenseMemoryGraph(cfg)
    zq = torch.randn(4, 6, 32)
    zt = zq + 0.05 * torch.randn_like(zq)
    loss = adjacent_contrastive_loss(mem, zq, zt, tau=0.1, num_negatives=7)
    loss.backward()
    assert torch.isfinite(loss)
    assert mem.query_proj.weight.grad is not None
    assert mem.key_proj.weight.grad is not None
    assert float(mem.query_proj.weight.grad.abs().sum()) > 0
    assert float(mem.key_proj.weight.grad.abs().sum()) > 0


def test_graph_coactivation_has_edge_gradient():
    torch.manual_seed(2)
    cfg = MemoryConfig(n_slots=128, d_mem=32, query_dim=16, top_k=8, graph_nodes=8, use_lsh=False)
    mem = LatentDenseMemoryGraph(cfg)
    zq = torch.randn(2, 5, 32)
    loss = graph_coactivation_loss(mem, zq)
    loss.backward()
    assert torch.isfinite(loss)
    assert mem.graph.edge_logits.grad is not None
    assert float(mem.graph.edge_logits.grad.abs().sum()) > 0


def test_retrieval_baseline_uses_topk_not_nslots():
    weights = torch.full((2, 3, 64), 1.0 / 64.0)
    m = retrieval_metrics(weights)
    assert abs(m["retrieval_uniform_top10_baseline"] - 10 / 64) < 1e-9
    assert abs(m["retrieval_top10_mass"] - 10 / 64) < 1e-6
