from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from latentloop_pds_m.data import make_synthetic_jsonl, JsonlTokenDataset
from latentloop_pds_m.optim import build_wsd_scheduler


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            return "cpu"
        return "cuda"
    return "cpu"


def dtype_for(cfg: LatentLoopConfig, device: str, requested: str = "auto") -> torch.dtype:
    if requested == "fp32":
        return torch.float32
    if requested == "bf16":
        return torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda" and cfg.optim.bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_cfg(path: str | Path) -> LatentLoopConfig:
    return LatentLoopConfig.from_file(path)


def build_model(cfg: LatentLoopConfig, device: str, dtype: torch.dtype | None = None) -> LatentLoopTransformerPDSM:
    if dtype is None:
        dtype = dtype_for(cfg, device)
    model = LatentLoopTransformerPDSM(cfg)
    if dtype != torch.float32:
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device)
    return model


def make_batch(cfg: LatentLoopConfig, seq: int, device: str, batch_size: int = 1) -> Dict[str, torch.Tensor]:
    seq = min(seq, cfg.transformer.max_seq_len)
    ids = torch.randint(3, cfg.transformer.vocab_size, (batch_size, seq), device=device)
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_float(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def tensor_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().item())


def grad_norm_of(obj: Any) -> float:
    vals = []
    if isinstance(obj, torch.nn.Parameter):
        if obj.grad is not None:
            vals.append(obj.grad.detach().float().norm())
    elif isinstance(obj, torch.nn.Module):
        for p in obj.parameters():
            if p.grad is not None:
                vals.append(p.grad.detach().float().norm())
    elif isinstance(obj, Iterable):
        for p in obj:
            if isinstance(p, torch.nn.Parameter) and p.grad is not None:
                vals.append(p.grad.detach().float().norm())
    if not vals:
        return 0.0
    return float(torch.stack(vals).sum().detach().cpu())


def module_groups(model: LatentLoopTransformerPDSM) -> Dict[str, Any]:
    duals = [m for m in model.layers if hasattr(m, "diffusion")]
    return {
        "embedding": model.embed_tokens,
        "lm_head": model.lm_head,
        "diffusion": [p for d in duals for p in d.diffusion.parameters()],
        "loop": [p for d in duals for p in d.loop.parameters()],
        "slot_queries": [d.diffusion.slot_queries for d in duals],
        "controller": [p for d in duals for p in d.controller.parameters()],
        "memory_values": model.memory.values,
        "memory_graph_edges": model.memory.graph.edge_logits,
        "memory_graph": model.memory.graph,
        "memory_ive": model.memory.ive,
        "dense_feedback": model.feedback if model.feedback is not None else [],
    }


def summarize_grads(model: LatentLoopTransformerPDSM) -> Dict[str, float]:
    return {name: grad_norm_of(obj) for name, obj in module_groups(model).items()}


def tiny_data_file(cfg: LatentLoopConfig, out_dir: Path, n: int = 16, seq_len: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "audit_synthetic.jsonl"
    make_synthetic_jsonl(path, n=n, vocab_size=cfg.transformer.vocab_size, seq_len=seq_len or cfg.transformer.max_seq_len)
    return path
