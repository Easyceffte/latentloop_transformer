from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latentloop_pds_m import LatentLoopConfig
from latentloop_pds_m.memory import LatentDenseMemoryGraph


@dataclass
class MetricsAccumulator:
    rows: List[Dict[str, float]]

    def add(self, row: Dict[str, float]) -> None:
        self.rows.append({k: float(v) for k, v in row.items() if isinstance(v, (float, int))})

    def summary(self, key: str) -> Dict[str, float] | None:
        vals = [r[key] for r in self.rows if key in r and math.isfinite(r[key])]
        if not vals:
            return None
        return {"first": vals[0], "last": vals[-1], "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}


class MemoryFeatureDataset(Dataset):
    def __init__(self, path: str | Path, target_field: str = "z_fused_l5", query_field: str = "z_query_l5", max_samples: int = 0):
        obj = torch.load(path, map_location="cpu")
        if obj.get("format") != "latentloop_memory_features_v2":
            raise ValueError(f"Unsupported feature format: {obj.get('format')}")
        if query_field not in obj or target_field not in obj:
            raise KeyError(f"Feature cache must contain {query_field!r} and {target_field!r}")
        self.query = obj[query_field].float()
        self.target = obj[target_field].float()
        if self.query.shape != self.target.shape:
            raise ValueError(f"query/target shape mismatch: {self.query.shape} vs {self.target.shape}")
        n = self.query.shape[0]
        if max_samples and max_samples < n:
            self.query = self.query[:max_samples]
            self.target = self.target[:max_samples]
            n = max_samples
        self.source = obj.get("source", ["unknown"] * n)
        self.domain = obj.get("domain", ["unknown"] * n)
        self.sample_id = obj.get("sample_id", torch.arange(n))[:n]
        self.span_start = obj.get("span_start", torch.arange(self.query.shape[1]))
        self.span_end = obj.get("span_end", self.span_start + 1)

    def __len__(self) -> int:
        return self.query.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "z_query": self.query[idx],
            "z_target": self.target[idx],
            "sample_id": torch.as_tensor(int(self.sample_id[idx]), dtype=torch.long),
        }


def _load_memory_from_checkpoint(memory: LatentDenseMemoryGraph, checkpoint: str | Path) -> Dict[str, object]:
    if not checkpoint:
        return {"loaded": False, "reason": "no checkpoint supplied"}
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    mem_state = memory.state_dict()
    loaded = []
    skipped = []
    new_state = dict(mem_state)
    for key in list(mem_state.keys()):
        ck = f"memory.{key}"
        if ck not in state:
            skipped.append({"key": key, "reason": "missing"})
            continue
        src = state[ck]
        dst = mem_state[key]
        if tuple(src.shape) == tuple(dst.shape):
            new_state[key] = src
            loaded.append(key)
        elif key in {"values", "retrieval_count"} and src.ndim == dst.ndim and src.shape[0] >= dst.shape[0] and src.shape[1:] == dst.shape[1:]:
            new_state[key] = src[: dst.shape[0]].clone()
            loaded.append(key + "[cropped]")
        else:
            skipped.append({"key": key, "reason": f"shape mismatch checkpoint={tuple(src.shape)} target={tuple(dst.shape)}"})
    memory.load_state_dict(new_state, strict=False)
    return {"loaded": True, "loaded_keys": loaded, "skipped": skipped}


def _set_trainable(memory: LatentDenseMemoryGraph, train_graph: bool) -> Dict[str, object]:
    allowed_prefixes = ["values", "query_proj", "key_proj", "ive", "reconstruct"]
    if train_graph:
        allowed_prefixes.append("graph")
    trainable = []
    frozen = []
    for name, p in memory.named_parameters():
        ok = any(name == prefix or name.startswith(prefix + ".") for prefix in allowed_prefixes)
        p.requires_grad_(ok)
        (trainable if ok else frozen).append(name)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "trainable_param_count": int(sum(p.numel() for p in memory.parameters() if p.requires_grad)),
        "frozen_param_count": int(sum(p.numel() for p in memory.parameters() if not p.requires_grad)),
    }


def _param_groups(memory: LatentDenseMemoryGraph, lr_values: float, lr_proj: float, lr_graph: float, weight_decay: float):
    groups = []
    value_params = []
    proj_params = []
    graph_params = []
    other_params = []
    for name, p in memory.named_parameters():
        if not p.requires_grad:
            continue
        if name == "values":
            value_params.append(p)
        elif name.startswith("graph."):
            graph_params.append(p)
        elif name.startswith("query_proj") or name.startswith("key_proj") or name.startswith("ive") or name.startswith("reconstruct"):
            proj_params.append(p)
        else:
            other_params.append(p)
    if value_params:
        groups.append({"params": value_params, "lr": lr_values, "weight_decay": 0.0, "name": "memory.values"})
    if proj_params:
        groups.append({"params": proj_params, "lr": lr_proj, "weight_decay": weight_decay, "name": "memory.projectors_and_value_processor"})
    if graph_params:
        groups.append({"params": graph_params, "lr": lr_graph, "weight_decay": weight_decay, "name": "memory.graph"})
    if other_params:
        groups.append({"params": other_params, "lr": lr_proj, "weight_decay": weight_decay, "name": "memory.other"})
    return groups


def _grad_norm(named_params: Iterable[Tuple[str, torch.nn.Parameter]], prefix: str) -> float:
    sq = 0.0
    for name, p in named_params:
        if name == prefix or name.startswith(prefix + "."):
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(sq)


def _retrieval_metrics(weights: torch.Tensor) -> Dict[str, float]:
    # weights: [B,S,K], expected sorted by top-k score.
    w = weights.detach().float()
    k = w.shape[-1]
    entropy = -(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(dim=-1).mean()
    return {
        "retrieval_top1_mass": float(w[..., :1].sum(dim=-1).mean().cpu()),
        "retrieval_top5_mass": float(w[..., : min(5, k)].sum(dim=-1).mean().cpu()),
        "retrieval_top10_mass": float(w[..., : min(10, k)].sum(dim=-1).mean().cpu()),
        "retrieval_uniform_top10_baseline": float(min(10, k) / max(1, k)),
        "retrieval_entropy": float(entropy.cpu()),
        "retrieval_entropy_div_logtopk": float((entropy / math.log(max(2, k))).cpu()),
    }


def _graph_metrics(memory: LatentDenseMemoryGraph) -> Dict[str, float]:
    with torch.no_grad():
        probs = torch.softmax(memory.graph.edge_logits.float(), dim=-1)
        entropy = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        return {
            "graph_entropy": float(entropy.cpu()),
            "graph_entropy_div_lognodes": float((entropy / math.log(probs.shape[-1])).cpu()),
            "graph_top1_prob_mean": float(probs.max(dim=-1).values.mean().cpu()),
            "graph_edge_ratio_0.02": float((probs > 0.02).float().mean().cpu()),
            "graph_edge_ratio_0.05": float((probs > 0.05).float().mean().cpu()),
            "graph_edge_ratio_0.10": float((probs > 0.10).float().mean().cpu()),
        }


def contrastive_loss(memory: LatentDenseMemoryGraph, z_query: torch.Tensor, z_target: torch.Tensor, tau: float) -> torch.Tensor:
    q = F.normalize(memory.query_proj(z_query.reshape(-1, z_query.shape[-1])).float(), dim=-1)
    k = F.normalize(memory.key_proj(z_target.detach().reshape(-1, z_target.shape[-1])).float(), dim=-1)
    logits = (q @ k.t()) / tau
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)


def graph_structure_loss(memory: LatentDenseMemoryGraph, z_query: torch.Tensor) -> torch.Tensor:
    # Weak adjacent-span graph objective: p(span_i -> span_i+1) should be larger than random far transitions.
    if z_query.shape[1] < 2:
        return z_query.new_zeros(())
    pack = memory.overview(z_query)
    src = pack["src_probs"].float()  # [B,S,G]
    edge = pack["edge_probs"].float()  # [G,G]
    a = src[:, :-1, :].reshape(-1, src.shape[-1])
    b = src[:, 1:, :].reshape(-1, src.shape[-1])
    p_pos = torch.einsum("ng,gh,nh->n", a, edge, b).clamp(1e-6, 1 - 1e-6)
    if b.shape[0] > 1:
        b_neg = b.roll(shifts=1, dims=0)
    else:
        b_neg = 1.0 - b
    p_neg = torch.einsum("ng,gh,nh->n", a, edge, b_neg).clamp(1e-6, 1 - 1e-6)
    return -(p_pos.log().mean() + (1.0 - p_neg).log().mean())


def train_step(memory: LatentDenseMemoryGraph, batch: Dict[str, torch.Tensor], cfg: Dict[str, float], device: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    zq = batch["z_query"].to(device)
    zt = batch["z_target"].to(device)
    out = memory.retrieve(zq, top_k=int(cfg["top_k"]))
    retrieved = out["retrieved"]
    pred = memory.reconstruct(retrieved)
    recon = F.mse_loss(pred.float(), zt.detach().float())
    nce = contrastive_loss(memory, zq, zt, tau=float(cfg["contrastive_tau"])) if cfg["contrastive_weight"] > 0 else zq.new_zeros(())
    gl = graph_structure_loss(memory, zq) if cfg["graph_weight"] > 0 else zq.new_zeros(())
    norm_reg = (memory.values.float().norm(dim=-1).mean() - zt.detach().float().norm(dim=-1).mean()).abs()
    loss = recon + cfg["contrastive_weight"] * nce + cfg["graph_weight"] * gl + cfg["norm_weight"] * norm_reg
    metrics = {
        "loss": float(loss.detach().float().cpu()),
        "recon_loss": float(recon.detach().float().cpu()),
        "contrastive_loss": float(nce.detach().float().cpu()),
        "graph_loss": float(gl.detach().float().cpu()),
        "norm_reg": float(norm_reg.detach().float().cpu()),
    }
    metrics.update(_retrieval_metrics(out["weights"]))
    return loss, metrics


def write_report(path: str | Path, rows: List[Dict[str, float]], extra: Dict[str, object], decision_threshold: float = 0.05) -> Dict[str, object]:
    acc = MetricsAccumulator(rows)
    keys = sorted({k for r in rows for k in r.keys() if isinstance(r.get(k), (int, float))})
    summaries = {k: acc.summary(k) for k in keys}
    first = summaries.get("recon_loss", {}).get("first") if summaries.get("recon_loss") else None
    last = summaries.get("recon_loss", {}).get("last") if summaries.get("recon_loss") else None
    drop = None
    if first is not None and first != 0 and last is not None:
        drop = (first - last) / abs(first)
    decision = "PASS" if drop is not None and drop >= decision_threshold else "REVIEW"
    report = {"decision": decision, "recon_loss_drop_fraction": drop, "summaries": summaries, **extra}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Phase 1.5B V2 Memory Objective Dry Run Report\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description="Train LDMG V2 memory objectives on cached Phase 1.5B features.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--report", default="docs/PHASE1_5B_V2_DRYRUN_REPORT.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = LatentLoopConfig.from_file(args.config)
    train_cfg = {
        "top_k": cfg.memory.top_k,
        "contrastive_weight": float(getattr(cfg.losses, "memory_contrastive_loss", 0.0)) if hasattr(cfg.losses, "memory_contrastive_loss") else 0.0,
        "graph_weight": float(getattr(cfg.losses, "memory_graph_structure_loss", 0.0)) if hasattr(cfg.losses, "memory_graph_structure_loss") else 0.0,
        "norm_weight": 0.001,
        "contrastive_tau": 0.1,
    }
    # YAML may include extra top-level memory_objective section, not represented in dataclasses.
    try:
        import yaml
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        mo = raw.get("memory_objective", {})
        train_cfg.update({
            "contrastive_weight": float(mo.get("contrastive_weight", train_cfg["contrastive_weight"])),
            "graph_weight": float(mo.get("graph_weight", train_cfg["graph_weight"])),
            "norm_weight": float(mo.get("norm_weight", train_cfg["norm_weight"])),
            "contrastive_tau": float(mo.get("contrastive_tau", train_cfg["contrastive_tau"])),
        })
    except Exception:
        pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory = LatentDenseMemoryGraph(cfg.memory).to(device)
    load_info = _load_memory_from_checkpoint(memory, args.checkpoint) if args.checkpoint else {"loaded": False, "reason": "no checkpoint supplied"}
    train_info = _set_trainable(memory, train_graph=train_cfg["graph_weight"] > 0)
    groups = _param_groups(memory, cfg.optim.lr * 0.2, cfg.optim.lr, cfg.optim.lr * 0.5, cfg.optim.weight_decay)
    opt = torch.optim.AdamW(groups, betas=(cfg.optim.beta1, cfg.optim.beta2))

    ds = MemoryFeatureDataset(args.features, max_samples=0)
    dl = DataLoader(ds, batch_size=cfg.optim.micro_batch_size, shuffle=True, drop_last=True)
    rows: List[Dict[str, float]] = []
    metrics_f = (out_dir / "metrics.jsonl").open("w", encoding="utf-8")
    step = 0
    memory.train()
    pbar = tqdm(total=args.max_steps, desc="memory_v2")
    while step < args.max_steps:
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            loss, row = train_step(memory, batch, train_cfg, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
            loss.backward()
            row["memory.values_grad_norm"] = _grad_norm(memory.named_parameters(), "values")
            row["memory.query_proj_grad_norm"] = _grad_norm(memory.named_parameters(), "query_proj")
            row["memory.key_proj_grad_norm"] = _grad_norm(memory.named_parameters(), "key_proj")
            row["memory.ive_grad_norm"] = _grad_norm(memory.named_parameters(), "ive")
            row["memory.reconstruct_grad_norm"] = _grad_norm(memory.named_parameters(), "reconstruct")
            row["memory.graph.edge_logits_grad_norm"] = _grad_norm(memory.named_parameters(), "graph.edge_logits")
            torch.nn.utils.clip_grad_norm_([p for p in memory.parameters() if p.requires_grad], cfg.optim.grad_clip)
            opt.step()
            row.update(_graph_metrics(memory))
            row["step"] = step
            row["lr_values"] = groups[0]["lr"] if groups else 0.0
            rows.append(row)
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_f.flush()
            if step % 10 == 0:
                pbar.set_postfix(loss=row["loss"], recon=row["recon_loss"])
            step += 1
            pbar.update(1)
            if step >= args.max_steps:
                break
    metrics_f.close()
    pbar.close()
    torch.save({"memory": memory.state_dict(), "config": cfg.to_dict(), "step": step, "load_info": load_info, "train_info": train_info}, out_dir / "memory_v2_last.pt")
    extra = {
        "load_info": load_info,
        "train_info": train_info,
        "train_cfg": train_cfg,
        "features": args.features,
        "checkpoint_saved": str(out_dir / "memory_v2_last.pt"),
        "optimizer_groups": [{"name": g.get("name", "group"), "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in groups],
    }
    report = write_report(args.report, rows, extra)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
