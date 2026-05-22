from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from latentloop_pds_m.data import JsonlTokenDataset


def _safe_torch_load(path: str | Path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def _load_model_state(model: torch.nn.Module, checkpoint: str | Path, *, strict: bool = False) -> Dict[str, object]:
    ckpt = _safe_torch_load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def _span_pool(x: torch.Tensor, span_size: int, max_spans: int | None = None) -> torch.Tensor:
    # x: [B,L,D]
    b, l, d = x.shape
    n = l // span_size
    if max_spans is not None:
        n = min(n, max_spans)
    if n <= 0:
        raise ValueError(f"span_size={span_size} exceeds sequence length={l}")
    x = x[:, : n * span_size, :].reshape(b, n, span_size, d)
    return x.mean(dim=2)


def _validate_features(out: Dict[str, object]) -> Dict[str, object]:
    report = {}
    for key in ["z_query_l5", "z_fused_l5"]:
        x = out[key].float()
        report[key] = {
            "shape": list(x.shape),
            "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)),
            "norm_mean": float(x.norm(dim=-1).mean()),
            "nan_count": int(torch.isnan(x).sum()),
            "inf_count": int(torch.isinf(x).sum()),
        }
    report["decision"] = "PASS" if all(report[k]["nan_count"] == 0 and report[k]["inf_count"] == 0 and report[k]["norm_mean"] > 0 for k in ["z_query_l5", "z_fused_l5"]) else "FAIL"
    return report


def main():
    ap = argparse.ArgumentParser(description="Extract span-level z_query/z_fused features for Phase 1.5B memory-objective training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--allow_random_init", action="store_true", help="Allow extraction from a randomly initialized model when no checkpoint is supplied.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--span_size", type=int, default=16)
    ap.add_argument("--max_spans", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="docs/PHASE1_5B_FEATURE_EXTRACTION_REPORT.md")
    args = ap.parse_args()

    if not args.checkpoint and not args.allow_random_init:
        raise ValueError("--checkpoint is required unless --allow_random_init is set")

    cfg = LatentLoopConfig.from_file(args.config)
    cfg.transformer.max_seq_len = args.seq_len
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda" and args.dtype == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda" and args.dtype == "fp16":
        dtype = torch.float16

    model = LatentLoopTransformerPDSM(cfg).to(device=device, dtype=dtype if dtype != torch.float32 else None)
    load_info = {"checkpoint": args.checkpoint, "random_init": not bool(args.checkpoint)}
    if args.checkpoint:
        load_info.update(_load_model_state(model, args.checkpoint, strict=False))
    model.eval()

    ds = JsonlTokenDataset(args.data, seq_len=args.seq_len)
    n = min(args.max_samples, len(ds))
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n]
    dl = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False, drop_last=False)

    zq_l5: List[torch.Tensor] = []
    zf_l5: List[torch.Tensor] = []
    zq_l11: List[torch.Tensor] = []
    zf_l11: List[torch.Tensor] = []
    input_ids: List[torch.Tensor] = []
    sources: List[str] = []
    domains: List[str] = []
    max_spans = args.max_spans if args.max_spans > 0 else None

    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(dl, desc="extract_memory_features")):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, aux, _ = model._run_layers(batch["input_ids"], batch.get("attention_mask"), global_step=0, collect_feedback=False)
            mem_qs = aux.get("z_memory_query", [])
            fused = aux.get("z_fused", [])
            if len(mem_qs) < 1 or len(fused) < 1:
                raise RuntimeError("Model did not expose z_memory_query/z_fused. Apply the Phase 1.5B dual_stream/modeling patch first.")
            zq_l5.append(_span_pool(mem_qs[0].detach().float().cpu(), args.span_size, max_spans))
            zf_l5.append(_span_pool(fused[0].detach().float().cpu(), args.span_size, max_spans))
            if len(mem_qs) > 1 and len(fused) > 1:
                zq_l11.append(_span_pool(mem_qs[1].detach().float().cpu(), args.span_size, max_spans))
                zf_l11.append(_span_pool(fused[1].detach().float().cpu(), args.span_size, max_spans))
            input_ids.append(batch["input_ids"].detach().cpu())
            # Preserve available metadata from the underlying JSON records.
            for original_idx in indices[batch_i * args.batch_size : batch_i * args.batch_size + batch["input_ids"].shape[0]]:
                rec = ds.records[original_idx]
                sources.append(str(rec.get("source", rec.get("dataset", "unknown"))))
                domains.append(str(rec.get("domain", rec.get("category", "unknown"))))

    num_spans = zq_l5[0].shape[1]
    span_start = torch.arange(num_spans, dtype=torch.long) * args.span_size
    span_end = span_start + args.span_size
    out_obj: Dict[str, object] = {
        "format": "latentloop_memory_features_v2",
        "version": 2,
        "random_init": load_info["random_init"],
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seq_len": args.seq_len,
        "span_size": args.span_size,
        "num_samples": int(sum(x.shape[0] for x in input_ids)),
        "num_spans": int(num_spans),
        "source": sources,
        "domain": domains,
        "sample_id": torch.tensor(indices[: len(sources)], dtype=torch.long),
        "input_ids": torch.cat(input_ids, dim=0),
        "span_start": span_start,
        "span_end": span_end,
        "z_query_l5": torch.cat(zq_l5, dim=0),
        "z_fused_l5": torch.cat(zf_l5, dim=0),
    }
    if zq_l11 and zf_l11:
        out_obj["z_query_l11"] = torch.cat(zq_l11, dim=0)
        out_obj["z_fused_l11"] = torch.cat(zf_l11, dim=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, out)
    validation = _validate_features(out_obj)
    report = {
        "title": "Phase 1.5B Feature Extraction Report",
        "load_info": load_info,
        "feature_path": str(out),
        "file_size_bytes": out.stat().st_size,
        "num_samples": out_obj["num_samples"],
        "seq_len": args.seq_len,
        "span_size": args.span_size,
        "num_spans": num_spans,
        "validation": validation,
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("# Phase 1.5B Feature Extraction Report\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
