from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_features(num_samples: int, num_spans: int, d_mem: int, seq_len: int, span_size: int, seed: int = 1234):
    if num_spans * span_size > seq_len:
        raise ValueError(f"num_spans * span_size must be <= seq_len, got {num_spans}*{span_size}>{seq_len}")
    gen = torch.Generator().manual_seed(seed)
    z_query = torch.randn(num_samples, num_spans, d_mem, generator=gen)
    # Fixed low-rank-ish transform gives a learnable relation while avoiding a trivial identity-only target.
    w = torch.randn(d_mem, d_mem, generator=gen) / (d_mem ** 0.5)
    z_fused = torch.tanh(z_query @ w) + 0.05 * torch.randn(num_samples, num_spans, d_mem, generator=gen)
    input_ids = torch.randint(3, 32000, (num_samples, seq_len), generator=gen, dtype=torch.long)
    span_start = torch.arange(num_spans, dtype=torch.long) * span_size
    span_end = span_start + span_size
    return {
        "format": "latentloop_memory_features_v2",
        "version": 2,
        "random_init": True,
        "num_samples": int(num_samples),
        "num_spans": int(num_spans),
        "d_mem": int(d_mem),
        "seq_len": int(seq_len),
        "span_size": int(span_size),
        "source": ["synthetic"] * num_samples,
        "domain": ["synthetic"] * num_samples,
        "sample_id": torch.arange(num_samples, dtype=torch.long),
        "input_ids": input_ids,
        "span_start": span_start,
        "span_end": span_end,
        "z_query_l5": z_query.to(torch.float32),
        "z_fused_l5": z_fused.to(torch.float32),
    }


def summarize(obj):
    out = {}
    for k in ["z_query_l5", "z_fused_l5"]:
        x = obj[k].float()
        out[k] = {
            "shape": list(x.shape),
            "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)),
            "norm_mean": float(x.norm(dim=-1).mean()),
            "nan_count": int(torch.isnan(x).sum()),
            "inf_count": int(torch.isinf(x).sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic Phase 1.5B memory feature cache for local smoke tests.")
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--num_spans", type=int, default=16)
    ap.add_argument("--d_mem", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--span_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    obj = make_features(args.num_samples, args.num_spans, args.d_mem, args.seq_len, args.span_size, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out)
    report = {
        "decision": "PASS",
        "path": str(out),
        "file_size_bytes": out.stat().st_size,
        **summarize(obj),
    }
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
