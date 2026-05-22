from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_memory_features import make_features
from train_memory_v2 import MemoryFeatureDataset, _retrieval_metrics, _load_memory_from_checkpoint
from latentloop_pds_m.config import MemoryConfig
from latentloop_pds_m.memory import LatentDenseMemoryGraph


def test_synthetic_feature_schema(tmp_path: Path):
    obj = make_features(num_samples=5, num_spans=4, d_mem=16, seq_len=32, span_size=8, seed=7)
    path = tmp_path / "features.pt"
    torch.save(obj, path)
    ds = MemoryFeatureDataset(path)
    assert len(ds) == 5
    item = ds[0]
    assert item["z_query"].shape == (4, 16)
    assert item["z_target"].shape == (4, 16)
    assert torch.isfinite(item["z_query"]).all()
    assert torch.isfinite(item["z_target"]).all()


def test_retrieval_top10_baseline_uses_topk_not_nslots():
    weights = torch.full((2, 3, 32), 1.0 / 32.0)
    metrics = _retrieval_metrics(weights)
    assert abs(metrics["retrieval_uniform_top10_baseline"] - 10 / 32) < 1e-8
    assert abs(metrics["retrieval_top10_mass"] - 10 / 32) < 1e-6


def test_checkpoint_memory_values_crop(tmp_path: Path):
    src = LatentDenseMemoryGraph(MemoryConfig(n_slots=8, d_mem=4, query_dim=2, top_k=2, graph_nodes=4, use_lsh=False))
    dst = LatentDenseMemoryGraph(MemoryConfig(n_slots=4, d_mem=4, query_dim=2, top_k=2, graph_nodes=4, use_lsh=False))
    with torch.no_grad():
        src.values.copy_(torch.arange(8 * 4).reshape(8, 4).float())
    ckpt = {"model": {f"memory.{k}": v for k, v in src.state_dict().items()}}
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(ckpt, ckpt_path)
    info = _load_memory_from_checkpoint(dst, ckpt_path)
    assert info["loaded"] is True
    assert torch.equal(dst.values, src.values[:4])


def test_train_memory_v2_two_steps_subprocess(tmp_path: Path):
    feature_path = tmp_path / "features.pt"
    obj = make_features(num_samples=16, num_spans=4, d_mem=16, seq_len=32, span_size=8, seed=11)
    torch.save(obj, feature_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
training_mode: from_scratch
transformer: {vocab_size: 128, max_seq_len: 32, n_layers: 2, d_model: 32, n_q_heads: 4, n_kv_heads: 2, ffn_dim: 64, dual_stream_layers: [0], dense_feedback_layer: 1, tie_lm_head: true, gradient_checkpointing: false}
diffusion: {latent_dim: 16, denoise_dim: 32, denoiser_layers: 1, denoiser_heads: 2, ddpm_timesteps: 10, idea_slots: 4}
loop: {latent_dim: 16, block_layers: 1, heads: 2}
interaction: {train_rounds: 1, infer_min_rounds: 1, infer_max_rounds: 1}
controller: {alpha_inf: 0.8, alpha_write: 0.5, alpha_read: 0.6}
memory: {n_slots: 128, d_mem: 16, query_dim: 8, top_k: 8, graph_nodes: 4, use_lsh: false, exact_threshold: 1000000, online_update: false}
dense_feedback: {enabled: false}
losses: {}
optim: {lr: 0.001, min_lr: 0.0001, beta1: 0.9, beta2: 0.98, weight_decay: 0.01, total_steps: 2, grad_clip: 1.0, micro_batch_size: 4, grad_accum: 1, bf16: false}
memory_objective: {contrastive_weight: 0.1, graph_weight: 0.01, norm_weight: 0.001, contrastive_tau: 0.1}
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    report = tmp_path / "report.md"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_memory_v2.py"),
        "--config", str(cfg_path),
        "--features", str(feature_path),
        "--max_steps", "2",
        "--output_dir", str(out_dir),
        "--report", str(report),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "memory_v2_last.pt").exists()
    assert report.exists()
    rows = [json.loads(x) for x in (out_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[-1]["memory.values_grad_norm"] > 0
    assert rows[-1]["memory.query_proj_grad_norm"] > 0
    assert rows[-1]["memory.key_proj_grad_norm"] > 0
