from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from latentloop_pds_m.data import JsonlTokenDataset
from latentloop_pds_m.optim import build_wsd_scheduler


def _scan_vocab(data_path: Path, vocab_size: int, max_lines: int = 128) -> Dict[str, int]:
    lo, hi, neg, lines = None, None, 0, 0
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows = json.loads(line).get("input_ids", [])
            if rows:
                lo = min(rows) if lo is None else min(lo, min(rows))
                hi = max(rows) if hi is None else max(hi, max(rows))
                neg += sum(1 for x in rows if x < 0)
            lines += 1
            if lines >= max_lines:
                break
    if lines == 0:
        raise ValueError(f"empty training data: {data_path}")
    if neg or (lo is not None and lo < 0) or (hi is not None and hi >= vocab_size):
        raise ValueError(f"vocab mismatch: min={lo} max={hi} vocab_size={vocab_size} negative={neg}")
    return {"lines": lines, "min_token_id": int(lo or 0), "max_token_id": int(hi or 0), "negative_count": int(neg)}


def _load_phase1(model: LatentLoopTransformerPDSM, checkpoint: str) -> Dict[str, object]:
    if not checkpoint:
        return {"loaded": False, "reason": "no phase1 checkpoint"}
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"loaded": True, "missing": list(missing), "unexpected": list(unexpected), "checkpoint": checkpoint}


def _load_memory(model: LatentLoopTransformerPDSM, memory_checkpoint: str) -> Dict[str, object]:
    if not memory_checkpoint:
        return {"loaded": False, "reason": "no memory checkpoint"}
    ckpt = torch.load(memory_checkpoint, map_location="cpu")
    mem = ckpt.get("memory", None) if isinstance(ckpt, dict) else None
    if mem is None:
        raise ValueError(f"memory checkpoint has no 'memory' key: {memory_checkpoint}")
    current = model.memory.state_dict()
    new = dict(current)
    loaded, skipped = [], []
    for k, dst in current.items():
        if k not in mem:
            skipped.append({"key": k, "reason": "missing"})
            continue
        src = mem[k]
        if tuple(src.shape) == tuple(dst.shape):
            new[k] = src
            loaded.append(k)
        else:
            skipped.append({"key": k, "reason": f"shape mismatch {tuple(src.shape)}->{tuple(dst.shape)}"})
    model.memory.load_state_dict(new, strict=False)
    return {"loaded": True, "loaded_keys": loaded, "skipped": skipped, "checkpoint": memory_checkpoint}


def _set_trainable(model: LatentLoopTransformerPDSM, mode: str) -> Dict[str, object]:
    if mode == "memory_only":
        prefixes = ["memory."]
    elif mode == "memory_controller_feedback":
        prefixes = ["memory.", "feedback."]
        # Controller lives inside dual stream blocks. Keep all controller params, not full attention/FFN.
        controller_substring = ".controller."
    else:
        prefixes = [""]
        controller_substring = ""
    trainable, frozen = [], []
    for name, p in model.named_parameters():
        if mode == "memory_controller_feedback":
            ok = any(name.startswith(x) for x in prefixes) or controller_substring in name or name.endswith(".gate.0.weight") or name.endswith(".gate.0.bias")
        else:
            ok = any(name.startswith(x) for x in prefixes)
        p.requires_grad_(ok)
        (trainable if ok else frozen).append(name)
    return {
        "mode": mode,
        "trainable_param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "frozen_param_count": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
        "trainable_preview": trainable[:200],
        "frozen_count": len(frozen),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint Memory-LM short finetune for local V3-16K.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--phase1_checkpoint", required=True)
    ap.add_argument("--memory_checkpoint", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=256)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--freeze_mode", choices=["memory_only", "memory_controller_feedback", "none"], default="memory_controller_feedback")
    ap.add_argument("--report", default="docs/LOCAL_V3_JOINT_MEMORY_LM_REPORT.md")
    args = ap.parse_args()

    cfg = LatentLoopConfig.from_file(args.config)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    vocab_stats = _scan_vocab(data_path, cfg.transformer.vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (cfg.optim.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = LatentLoopTransformerPDSM(cfg).to(device=device, dtype=dtype if dtype != torch.float32 else None)
    phase1_info = _load_phase1(model, args.phase1_checkpoint)
    memory_info = _load_memory(model, args.memory_checkpoint) if args.memory_checkpoint else {"loaded": False, "reason": "no memory checkpoint"}
    train_info = _set_trainable(model, args.freeze_mode)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters selected")
    opt = torch.optim.AdamW(trainable, lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2), weight_decay=cfg.optim.weight_decay)
    sched = build_wsd_scheduler(opt, cfg.optim.warmup_steps, cfg.optim.total_steps, cfg.optim.stable_until_ratio, cfg.optim.min_lr / cfg.optim.lr)
    ds = JsonlTokenDataset(data_path, seq_len=cfg.transformer.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.optim.micro_batch_size, shuffle=True, drop_last=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = (out_dir / "metrics.jsonl").open("w", encoding="utf-8")
    rows = []
    step = 0
    opt_step = 0
    model.train()
    pbar = tqdm(total=args.max_steps, desc="joint_memory_lm")
    while step < args.max_steps:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, global_step=opt_step)
            (out["loss"] / cfg.optim.grad_accum).backward()
            boundary = ((step + 1) % cfg.optim.grad_accum == 0) or ((step + 1) >= args.max_steps)
            if boundary:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.optim.grad_clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); opt_step += 1
            row = {k: float(v.detach().float().cpu()) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1}
            row.update({f"aux_{k}": float(v.detach().float().cpu()) for k, v in out.get("aux", {}).items() if torch.is_tensor(v) and v.numel() == 1})
            row["micro_step"] = step
            row["optimizer_step"] = opt_step
            rows.append(row)
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n"); metrics_f.flush()
            if step % 10 == 0:
                pbar.set_postfix(loss=row.get("loss"), lm=row.get("lm_loss"))
            step += 1; pbar.update(1)
            if step >= args.max_steps:
                break
    metrics_f.close(); pbar.close()
    ckpt_path = out_dir / "joint_memory_lm_last.pt"
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "micro_step": step, "optimizer_step": opt_step, "phase1_info": phase1_info, "memory_info": memory_info, "train_info": train_info}, ckpt_path)
    def summ(key: str):
        vals = [r[key] for r in rows if key in r]
        return None if not vals else {"first": vals[0], "last": vals[-1], "min": min(vals), "max": max(vals), "mean": sum(vals)/len(vals)}
    report = {"decision": "PASS", "checkpoint": str(ckpt_path), "vocab_stats": vocab_stats, "phase1_info": phase1_info, "memory_info": memory_info, "train_info": train_info, "loss": summ("loss"), "lm_loss": summ("lm_loss"), "rows": len(rows)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("# Local V3 Joint Memory-LM Report\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
