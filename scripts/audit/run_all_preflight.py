from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from latentloop_pds_m.dual_stream import DualStreamBlock
from latentloop_pds_m.optim import build_wsd_scheduler


def finite(x: torch.Tensor) -> bool:
    return torch.isfinite(x.detach().float()).all().item()


def small_batch(cfg: LatentLoopConfig, seq: int, device: str):
    seq = min(seq, cfg.transformer.max_seq_len)
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, seq), device=device)
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def traceability(model: LatentLoopTransformerPDSM, cfg: LatentLoopConfig) -> Dict[str, Any]:
    dual = [i for i, layer in enumerate(model.layers) if isinstance(layer, DualStreamBlock)]
    losses = [
        "lm_loss", "ddpm_loss", "anchor_loss", "slot_diversity_loss", "exit_entropy_loss",
        "shortcut_consistency_loss", "dense_feedback_reg_loss", "controller_balance_loss",
        "memory_surprise_loss", "graph_sparsity_loss", "memory_stability_loss",
    ]
    return {
        "training_mode_from_scratch": cfg.training_mode == "from_scratch",
        "n_layers": len(model.layers),
        "n_layers_ok": len(model.layers) == 18,
        "dual_indices": dual,
        "dual_indices_ok": dual == cfg.transformer.dual_stream_layers,
        "primary_dual_has_memory": isinstance(model.layers[cfg.transformer.dual_stream_layers[0]], DualStreamBlock) and model.layers[cfg.transformer.dual_stream_layers[0]].is_primary,
        "secondary_dual_no_memory": isinstance(model.layers[cfg.transformer.dual_stream_layers[1]], DualStreamBlock) and not model.layers[cfg.transformer.dual_stream_layers[1]].is_primary,
        "dense_feedback_enabled": model.feedback is not None,
        "dense_feedback_routes": list(model.feedback.routes) if model.feedback is not None else [],
        "dense_feedback_route_count_ok": model.feedback is not None and len(model.feedback.routes) == cfg.dense_feedback.n_routes == 10,
        "loss_weights_present": {k: hasattr(cfg.losses, k) for k in losses},
        "parameter_count": model.parameter_count(),
        "config_memory_slots": cfg.memory.n_slots,
        "config_memory_top_k": cfg.memory.top_k,
        "config_lsh_query_chunk": cfg.memory.lsh_query_chunk,
    }


def causality(cfg: LatentLoopConfig, device: str, seq: int) -> Dict[str, Any]:
    torch.manual_seed(111)
    model = LatentLoopTransformerPDSM(cfg).to(device)
    model.train()
    prefix = max(2, min(seq // 2, cfg.transformer.max_seq_len // 2))
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, min(seq, cfg.transformer.max_seq_len)), device=device)
    ids2 = ids.clone()
    ids2[:, prefix:] = torch.randint(3, cfg.transformer.vocab_size, ids2[:, prefix:].shape, device=device)
    results = {}
    for mode in ["train", "eval"]:
        getattr(model, mode)()
        torch.manual_seed(777)
        out1 = model(ids, labels=None, global_step=cfg.dense_feedback.warmup_steps + 1)["logits"].detach()
        torch.manual_seed(777)
        out2 = model(ids2, labels=None, global_step=cfg.dense_feedback.warmup_steps + 1)["logits"].detach()
        diff = (out1[:, :prefix] - out2[:, :prefix]).abs().max().item()
        results[f"{mode}_prefix_max_abs_diff"] = diff
        results[f"{mode}_pass"] = diff < 5e-4
    return results


def loss_terms_and_grad(cfg: LatentLoopConfig, device: str, seq: int) -> Dict[str, Any]:
    torch.manual_seed(222)
    model = LatentLoopTransformerPDSM(cfg).to(device)
    model.train()
    batch = small_batch(cfg, seq, device)
    out = model(**batch, global_step=cfg.dense_feedback.warmup_steps + 1)
    loss_names = [k for k in out.keys() if k.endswith("loss") or k.endswith("_raw") or k == "loss"]
    losses = {k: float(out[k].detach().float().mean()) for k in loss_names if torch.is_tensor(out[k])}
    finite_losses = {k: math.isfinite(v) for k, v in losses.items()}
    weighted_sum = out["lm_loss"].detach().float()
    for k in [
        "ddpm_loss", "anchor_loss", "slot_diversity_loss", "exit_entropy_loss", "shortcut_consistency_loss",
        "dense_feedback_reg_loss", "controller_balance_loss", "memory_surprise_loss", "graph_sparsity_loss", "memory_stability_loss",
    ]:
        weighted_sum = weighted_sum + out[k].detach().float()
    total_delta = float((weighted_sum - out["loss"].detach().float()).abs().max())
    model.zero_grad(set_to_none=True)
    out["lm_loss"].backward(retain_graph=True)
    grad_report = {}
    modules = {
        "embed_tokens": model.embed_tokens,
        "lm_head": model.lm_head,
        "memory_values": model.memory.values,
        "memory_graph_edges": model.memory.graph.edge_logits,
    }
    for name, obj in modules.items():
        if isinstance(obj, torch.nn.Parameter):
            g = obj.grad
            grad_report[name] = 0.0 if g is None else float(g.detach().float().norm())
        else:
            vals = [p.grad.detach().float().norm() for p in obj.parameters() if p.grad is not None]
            grad_report[name] = float(torch.stack(vals).sum()) if vals else 0.0
    if model.feedback is not None:
        route_grads = {}
        for route in model.feedback.routes:
            key = route.replace('.', '_')
            vals = []
            for p in list(model.feedback.proj[key].parameters()) + list(model.feedback.gate[key].parameters()):
                if p.grad is not None:
                    vals.append(p.grad.detach().float().norm())
            route_grads[route] = float(torch.stack(vals).sum()) if vals else 0.0
        grad_report["dense_feedback_routes"] = route_grads
    return {"losses": losses, "finite_losses": finite_losses, "total_delta": total_delta, "grad_report": grad_report}


def scheduler_audit(cfg: LatentLoopConfig, out_dir: Path) -> Dict[str, Any]:
    p = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.AdamW([p], lr=cfg.optim.lr)
    sched = build_wsd_scheduler(opt, cfg.optim.warmup_steps, cfg.optim.total_steps, cfg.optim.stable_until_ratio, cfg.optim.min_lr / cfg.optim.lr)
    rows = []
    sample_steps = sorted(set([0, 1, cfg.optim.warmup_steps - 1, cfg.optim.warmup_steps, int(cfg.optim.total_steps * cfg.optim.stable_until_ratio), cfg.optim.total_steps - 1]))
    for step in range(max(1, cfg.optim.total_steps)):
        if step in sample_steps:
            theta_drops = step // max(1, cfg.memory.write_threshold_decay_steps)
            theta = max(cfg.memory.write_threshold_target, cfg.memory.write_threshold_initial - theta_drops * cfg.memory.write_threshold_decay_amount)
            anchor_frac = min(1.0, float(step) / max(1, cfg.losses.anchor_warmup_steps))
            anchor_w = cfg.losses.anchor_loss_initial + (cfg.losses.anchor_loss - cfg.losses.anchor_loss_initial) * anchor_frac
            ctrl_w = cfg.losses.controller_balance_loss * max(0.0, 1.0 - float(step) / max(1, cfg.losses.controller_balance_warmup_steps))
            fb_scale = min(1.0, float(step) / max(1, cfg.dense_feedback.warmup_steps))
            rows.append({"optimizer_step": step, "lr": opt.param_groups[0]["lr"], "theta_write": theta, "anchor_weight": anchor_w, "controller_balance_weight": ctrl_w, "feedback_scale": fb_scale})
        opt.step(); sched.step()
    with (out_dir / "schedule_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    return {"sampled": rows, "warmup_steps": cfg.optim.warmup_steps, "total_steps": cfg.optim.total_steps}


def memory_audit(cfg: LatentLoopConfig, device: str) -> Dict[str, Any]:
    # Avoid full-bank LSH rebuild in CPU audit for large configs; inspect invariants and run exact small path elsewhere.
    return {
        "n_slots": cfg.memory.n_slots,
        "top_k": cfg.memory.top_k,
        "use_lsh": cfg.memory.use_lsh,
        "lsh_query_chunk": cfg.memory.lsh_query_chunk,
        "lsh_max_candidates": cfg.memory.lsh_max_candidates,
        "forbidden_dense_candidate_shape": "[B*L,N]",
        "exact_threshold": cfg.memory.exact_threshold,
        "risk_large_full_rebuild": cfg.memory.use_lsh and cfg.memory.n_slots > cfg.memory.exact_threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="reports/preflight_full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq", type=int, default=32)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = LatentLoopConfig.from_file(args.config)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
    model = LatentLoopTransformerPDSM(cfg).to(device)
    report = {
        "config": args.config,
        "device": device,
        "traceability": traceability(model, cfg),
        "causality": causality(cfg, device, args.seq),
        "loss_grad": loss_terms_and_grad(cfg, device, args.seq),
        "scheduler": scheduler_audit(cfg, out_dir),
        "memory": memory_audit(cfg, device),
    }
    # Decision gate.
    blocked = []
    if not report["traceability"].get("n_layers_ok"):
        blocked.append("n_layers")
    if not all(v for k, v in report["causality"].items() if k.endswith("_pass")):
        blocked.append("causality")
    if report["loss_grad"]["total_delta"] > 1e-4:
        blocked.append("loss_sum")
    if not all(report["loss_grad"]["finite_losses"].values()):
        blocked.append("nonfinite_loss")
    route_grads = report["loss_grad"]["grad_report"].get("dense_feedback_routes", {})
    if route_grads and sum(1 for v in route_grads.values() if v > 0) < 8:
        blocked.append("dense_feedback_task_grad")
    decision = "PASS_TO_100_STEP" if not blocked else "BLOCKED_DO_NOT_TRAIN"
    report["blocked_reasons"] = blocked
    report["decision"] = decision
    (out_dir / "preflight_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "final_decision.md").write_text(f"# {decision}\n\nBlocked reasons: {blocked}\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "blocked_reasons": blocked, "out": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
