from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from common import load_cfg, resolve_device, dtype_for, build_model, make_batch, write_json, summarize_grads


def run(config: str, device: str, seq: int, dtype: str, out: str | None = None):
    cfg = load_cfg(config)
    device = resolve_device(device)
    if device == "cpu":
        torch.set_num_threads(1)
    dt = dtype_for(cfg, device, dtype)
    report = {"config": config, "requested_device": device, "dtype": str(dt), "seq": seq}
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = build_model(cfg, device, dt).train()
    batch = make_batch(cfg, seq, device)
    if device == "cuda":
        report["after_model_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        report["after_model_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    out_d = model(**batch, global_step=max(1, cfg.dense_feedback.warmup_steps + 1))
    finite_loss = bool(torch.isfinite(out_d["loss"].detach()).all().item())
    out_d["loss"].backward()
    grad_norms = summarize_grads(model)
    report.update({
        "finite_loss": finite_loss,
        "loss": float(out_d["loss"].detach().float()),
        "grad_norm_sum": float(sum(grad_norms.values())),
        "elapsed_sec": time.time() - t0,
    })
    if device == "cuda":
        report["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        report["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    blocked = []
    if not finite_loss: blocked.append("nonfinite_loss")
    if report["grad_norm_sum"] <= 0: blocked.append("zero_grad")
    report["blocked_reasons"] = blocked
    report["decision"] = "PASS" if not blocked else "BLOCKED"
    if out: write_json(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--seq", type=int, default=64); ap.add_argument("--dtype", default="auto"); ap.add_argument("--out", default="")
    a=ap.parse_args(); run(a.config,a.device,a.seq,a.dtype,a.out or None)
