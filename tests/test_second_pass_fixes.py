from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from latentloop_pds_m.config import MemoryConfig, DenseFeedbackConfig
from latentloop_pds_m.memory import LatentDenseMemoryGraph
from latentloop_pds_m.dense_feedback import DenseFeedbackRouter
from scripts.data.common import make_mock_documents, selected_text_from_row


def test_openmath_problem_solution_formatter_keeps_both_fields():
    text, field = selected_text_from_row(
        {"problem": "What is 2+2?", "generated_solution": "Compute 2+2=4."},
        {"formatter": "problem_solution", "text_field_priority": ["generated_solution", "problem"]},
    )
    assert field == "problem+generated_solution"
    assert "Problem:" in text and "What is 2+2?" in text
    assert "Solution:" in text and "Compute 2+2=4." in text


def test_offline_mock_is_stable_across_processes(tmp_path: Path):
    code = """
from scripts.data.common import make_mock_documents, sha1_text
row = next(make_mock_documents('fineweb_edu', 'general', 1, seed=123))
print(sha1_text(row['text']))
"""
    vals = []
    for _ in range(3):
        out = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True).strip()
        vals.append(out)
    assert len(set(vals)) == 1


def test_memory_write_mask_gates_stability_loss():
    cfg = MemoryConfig(n_slots=16, d_mem=8, query_dim=4, top_k=2, graph_nodes=4, use_lsh=False)
    mem = LatentDenseMemoryGraph(cfg).train()
    z = torch.zeros(1, 3, 8)
    retrieved = torch.zeros_like(z)
    g = torch.ones(1, 3)
    # At step 0 theta=0.3 and surprise is 0, so write_mask is false and stability is exactly 0.
    out = mem.write_losses_and_maybe_update(z, retrieved, g, global_step=0)
    assert float(out["write_skip_rate"]) == 1.0
    assert float(out["memory_stability_loss"].detach()) == 0.0


def test_apply_forgetting_changes_edge_distribution_when_decay_visible():
    cfg = MemoryConfig(n_slots=16, d_mem=8, query_dim=4, top_k=2, graph_nodes=4, use_lsh=False)
    cfg.edge_decay_gamma = 0.5
    cfg.edge_prune_threshold = 0.0
    mem = LatentDenseMemoryGraph(cfg)
    with torch.no_grad():
        mem.graph.edge_logits.zero_()
        mem.graph.edge_logits[:, 0] = 5.0
    before = torch.softmax(mem.graph.edge_logits, dim=-1).clone()
    mem.apply_forgetting()
    after = torch.softmax(mem.graph.edge_logits, dim=-1)
    assert not torch.allclose(before, after)
    assert after[:, 0].mean() < before[:, 0].mean()


def test_dense_feedback_gate_metric_is_actual_gate_range():
    router = DenseFeedbackRouter(DenseFeedbackConfig(warmup_steps=1), d_model=16, target_dim=8)
    h = torch.randn(2, 5, 16)
    out = router(h, global_step=1)
    assert "dense_feedback_gate_mean" in out
    assert "dense_feedback_signal_mean" in out
    assert 0.0 <= float(out["dense_feedback_gate_mean"].detach()) <= 1.0
