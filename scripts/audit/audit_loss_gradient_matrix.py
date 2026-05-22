from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from common import load_cfg, resolve_device, dtype_for, build_model, make_batch, summarize_grads, write_json

LOSS_KEYS = [
    "lm_loss", "ddpm_loss", "anchor_loss", "slot_diversity_loss", "exit_entropy_loss",
    "shortcut_consistency_loss", "dense_feedback_reg_loss", "controller_balance_loss",
    "memory_surprise_loss", "graph_sparsity_loss", "memory_stability_loss",
]

EXPECTED = {
    "lm_loss": ["embedding", "lm_head", "diffusion", "loop", "controller", "dense_feedback"],
    "ddpm_loss": ["diffusion"],
    "slot_diversity_loss": ["slot_queries"],
    "graph_sparsity_loss": ["memory_graph_edges"],
    "dense_feedback_reg_loss": ["dense_feedback"],
}


def run(config: str, device: str, seq: int, out_dir: str | None = None):
    cfg = load_cfg(config); device = resolve_device(device)
    if device == "cpu": torch.set_num_threads(1)
    dt = dtype_for(cfg, device)
    torch.manual_seed(1000)
    model = build_model(cfg, device, dt).train()
    batch = make_batch(cfg, min(seq, cfg.transformer.max_seq_len), device)
    out = model(**batch, global_step=max(1, cfg.dense_feedback.warmup_steps + 1))
    rows=[]; blocked=[]
    for loss_name in LOSS_KEYS:
        if loss_name not in out or not torch.is_tensor(out[loss_name]):
            blocked.append(f"missing_{loss_name}"); continue
        model.zero_grad(set_to_none=True)
        loss = out[loss_name]
        # Reuse the same graph for this audit to avoid repeated heavyweight model builds.
        # Each backward is isolated by zero_grad; retain_graph keeps other loss probes valid.
        loss.backward(retain_graph=True)
        grads = summarize_grads(model)
        row = {"loss": loss_name, "loss_value": float(loss.detach().float()), **grads}
        rows.append(row)
        for group in EXPECTED.get(loss_name, []):
            if abs(grads.get(group, 0.0)) <= 0.0:
                blocked.append(f"{loss_name}_no_grad_{group}")
    report={"config":config,"device":device,"rows":rows,"blocked_reasons":blocked,"decision":"PASS" if not blocked else "BLOCKED"}
    if out_dir:
        od=Path(out_dir); od.mkdir(parents=True, exist_ok=True)
        if rows:
            with (od/"loss_gradient_matrix.csv").open("w", newline="", encoding="utf-8") as f:
                writer=csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
        write_json(od/"loss_gradient_matrix.json", report)
    print(json.dumps({"decision":report["decision"],"blocked_reasons":blocked}, indent=2, ensure_ascii=False)); return report

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cpu"); ap.add_argument("--seq",type=int,default=64); ap.add_argument("--out_dir",default="")
    a=ap.parse_args(); run(a.config,a.device,a.seq,a.out_dir or None)
