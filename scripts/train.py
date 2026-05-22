from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from latentloop_pds_m.data import JsonlTokenDataset, make_synthetic_jsonl
from latentloop_pds_m.optim import build_wsd_scheduler


def _rng_state():
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(model, opt, sched, cfg, micro_step: int, optimizer_step: int, resumable_optimizer_boundary: bool = True):
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "config": cfg.to_dict(),
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "step": micro_step,
        "rng_state": _rng_state(),
        "resumable_optimizer_boundary": bool(resumable_optimizer_boundary),
    }


def _validate_processed_data_path(data_arg: str, processed_root: Path | None = None) -> Path:
    if not data_arg:
        raise ValueError("--data is required unless --synthetic is set")
    data_path = Path(data_arg)
    if not data_path.exists():
        raise FileNotFoundError(f"training data does not exist: {data_path}")
    if data_path.suffix != ".jsonl":
        raise ValueError(f"training data must be a processed .jsonl file: {data_path}")
    processed_root = (processed_root or (ROOT / "data" / "processed")).resolve()
    resolved = data_path.resolve()
    try:
        resolved.relative_to(processed_root)
    except ValueError as exc:
        raise ValueError(f"training data must be under {processed_root}: {resolved}") from exc
    return data_path


def _scan_token_id_range(data_path: Path, max_lines: int = 256) -> dict:
    """Scan a small prefix of a processed JSONL file before CUDA init.

    This catches data/config tokenizer mismatches on CPU instead of failing later
    as a CUDA embedding index error.
    """
    min_id = None
    max_id = None
    invalid_negative_count = 0
    line_count = 0
    token_count = 0
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            rec = json.loads(line)
            ids = rec.get("input_ids")
            if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                raise ValueError(f"processed data line {line_count} has non-list[int] input_ids")
            if ids:
                lo = min(ids)
                hi = max(ids)
                min_id = lo if min_id is None else min(min_id, lo)
                max_id = hi if max_id is None else max(max_id, hi)
                invalid_negative_count += sum(1 for x in ids if x < 0)
                token_count += len(ids)
            if line_count >= max_lines:
                break
    if line_count == 0:
        raise ValueError(f"processed data is empty: {data_path}")
    return {
        "line_count_scanned": line_count,
        "token_count_scanned": token_count,
        "min_token_id": int(min_id if min_id is not None else 0),
        "max_token_id": int(max_id if max_id is not None else 0),
        "invalid_negative_count": int(invalid_negative_count),
    }


def _validate_vocab_compatibility(data_path: Path, vocab_size: int, max_lines: int = 256) -> dict:
    stats = _scan_token_id_range(data_path, max_lines=max_lines)
    if stats["invalid_negative_count"] or stats["min_token_id"] < 0 or stats["max_token_id"] >= int(vocab_size):
        raise ValueError(
            "Data/config vocab mismatch: "
            f"min input_id={stats['min_token_id']}, max input_id={stats['max_token_id']}, "
            f"config vocab_size={vocab_size}. Use a config with matching vocab_size or regenerate data."
        )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "smoke_tiny.yaml"))
    ap.add_argument("--data", default="")
    ap.add_argument("--output_dir", default=str(ROOT / "outputs" / "smoke"))
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = LatentLoopConfig.from_file(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        data_path = Path(args.data) if args.data else out_dir / "synthetic.jsonl"
        make_synthetic_jsonl(data_path, n=256, vocab_size=cfg.transformer.vocab_size, seq_len=cfg.transformer.max_seq_len)
    else:
        data_path = _validate_processed_data_path(args.data)
        vocab_stats = _validate_vocab_compatibility(data_path, cfg.transformer.vocab_size)
        print(f"data vocab preflight: {vocab_stats}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (cfg.optim.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    # Keep fp32 parameters by default? The current project target prioritizes fitting on 12GB,
    # so bf16 parameters are used when explicitly enabled by config and supported by CUDA.
    # For maximum optimizer stability, set optim.bf16=false in the config for fp32 parameters.
    model = LatentLoopTransformerPDSM(cfg).to(device=device, dtype=dtype if dtype != torch.float32 else None)
    ds = JsonlTokenDataset(data_path, seq_len=cfg.transformer.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.optim.micro_batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2), weight_decay=cfg.optim.weight_decay)
    sched = build_wsd_scheduler(opt, cfg.optim.warmup_steps, cfg.optim.total_steps, cfg.optim.stable_until_ratio, cfg.optim.min_lr / cfg.optim.lr)
    step = 0
    opt_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        _set_rng_state(ckpt.get("rng_state"))
        if ckpt.get("resumable_optimizer_boundary", True) is False:
            raise ValueError(f"checkpoint is not resumable because it was saved mid-gradient-accumulation: {args.resume}")
        step = int(ckpt.get("micro_step", ckpt.get("step", 0)))
        opt_step = int(ckpt.get("optimizer_step", 0))
        print(f"resumed {args.resume} at micro_step={step} optimizer_step={opt_step}; data sampler position is not restored")
    model.train()
    pbar = tqdm(total=args.max_steps, initial=step, desc="train")
    metrics_path = out_dir / "metrics.jsonl"
    append_metrics = bool(args.resume and metrics_path.exists())
    log_f = metrics_path.open("a" if append_metrics else "w", encoding="utf-8")
    if args.resume:
        resume_rec = {
            "event": "resume",
            "resume_checkpoint": args.resume,
            "loaded_micro_step": step,
            "loaded_optimizer_step": opt_step,
            "lr": opt.param_groups[0]["lr"],
            "data_order_resume_semantics": "optimizer_scheduler_rng_restored_sampler_position_not_restored",
        }
        log_f.write(json.dumps(resume_rec, ensure_ascii=False) + "\n"); log_f.flush()
    while step < args.max_steps:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Spec schedules are defined over optimizer steps, not microsteps.
            out = model(**batch, global_step=opt_step)
            is_last_microstep = (step + 1) >= args.max_steps
            accum_boundary = ((step + 1) % cfg.optim.grad_accum == 0)
            accum_count = ((step % cfg.optim.grad_accum) + 1)
            (out["loss"] / cfg.optim.grad_accum).backward()
            if accum_boundary or is_last_microstep:
                if accum_count < cfg.optim.grad_accum:
                    scale = float(cfg.optim.grad_accum) / float(accum_count)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); opt_step += 1
                # Spec-mandated LDMG forgetting/edge decay cadence. This is intentionally
                # tied to optimizer steps, not microsteps, so gradient accumulation does not
                # accelerate memory decay.
                if opt_step > 0 and opt_step % 1000 == 0 and hasattr(model, "memory"):
                    model.memory.apply_forgetting()
            rec = {"micro_step": step, "optimizer_step": opt_step, "loss": float(out["loss"].detach().float()), "lr": opt.param_groups[0]["lr"]}
            for k, v in out.items():
                if k not in {"logits", "aux"} and torch.is_tensor(v):
                    rec[k] = float(v.detach().float().mean())
            for k, v in out["aux"].items():
                if torch.is_tensor(v):
                    rec[k] = float(v.detach().float().mean())
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); log_f.flush()
            if step % 10 == 0:
                pbar.set_postfix(loss=rec["loss"], lm=rec["lm_loss"], lr=rec["lr"])
            next_micro_step = step + 1
            if args.save_every and next_micro_step % args.save_every == 0:
                boundary_now = accum_boundary or is_last_microstep
                ckpt_name = f"ckpt_{next_micro_step:06d}.pt" if boundary_now else f"ckpt_{next_micro_step:06d}.snapshot.pt"
                torch.save(
                    _checkpoint_payload(model, opt, sched, cfg, next_micro_step, opt_step, resumable_optimizer_boundary=boundary_now),
                    out_dir / ckpt_name,
                )
            step = next_micro_step; pbar.update(1)
            if step >= args.max_steps:
                break
    torch.save(_checkpoint_payload(model, opt, sched, cfg, step, opt_step), out_dir / "last.pt")
    log_f.close(); pbar.close()
    print(f"saved {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
