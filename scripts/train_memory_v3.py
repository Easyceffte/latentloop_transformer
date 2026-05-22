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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latentloop_pds_m import LatentLoopConfig
from latentloop_pds_m.memory import LatentDenseMemoryGraph


@dataclass
class MetricSeries:
    rows: List[Dict[str, float]]

    def summary(self, key: str) -> Dict[str, float] | None:
        vals = [float(r[key]) for r in self.rows if key in r and math.isfinite(float(r[key]))]
        if not vals:
            return None
        return {"first": vals[0], "last": vals[-1], "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}


class MemoryFeatureDataset(Dataset):
    def __init__(self, path: str | Path, query_field: str, target_field: str, max_samples: int = 0):
        obj = torch.load(path, map_location="cpu")
        if obj.get("format") != "latentloop_memory_features_v2":
            raise ValueError(f"Unsupported feature format: {obj.get('format')}")
        if query_field not in obj or target_field not in obj:
            raise KeyError(f"Feature cache must contain {query_field!r} and {target_field!r}")
        self.query = obj[query_field].float()
        self.target = obj[target_field].float()
        if self.query.shape != self.target.shape:
            raise ValueError(f"query/target shape mismatch: {tuple(self.query.shape)} vs {tuple(self.target.shape)}")
        if self.query.dim() != 3:
            raise ValueError(f"features must be [N,S,D], got {tuple(self.query.shape)}")
        n = self.query.shape[0]
        if max_samples and max_samples < n:
            self.query = self.query[:max_samples]
            self.target = self.target[:max_samples]
            n = max_samples
        self.sample_id = obj.get("sample_id", torch.arange(n))[:n]
        self.source = obj.get("source", ["unknown"] * n)
        self.domain = obj.get("domain", ["unknown"] * n)

    def __len__(self) -> int:
        return int(self.query.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "z_query": self.query[idx],
            "z_target": self.target[idx],
            "sample_id": torch.as_tensor(int(self.sample_id[idx]), dtype=torch.long),
        }


def _read_memory_objective_config(path: str | Path) -> Dict[str, float | str | int | bool]:
    raw: Dict[str, object] = {}
    try:
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        raw = {}
    mo = raw.get("memory_objective", {}) if isinstance(raw, dict) else {}
    return {
        "query_field": str(mo.get("query_field", "z_query_l5")),
        "target_field": str(mo.get("target_field", "z_fused_l5")),
        "reconstruction_weight": float(mo.get("reconstruction_weight", 1.0)),
        "contrastive_weight": float(mo.get("contrastive_weight", 0.1)),
        "graph_weight": float(mo.get("graph_weight", 0.01)),
        "distiller_weight": float(mo.get("distiller_weight", 0.05)),
        "norm_weight": float(mo.get("norm_weight", 0.001)),
        "contrastive_tau": float(mo.get("contrastive_tau", 0.1)),
        "num_negatives": int(mo.get("num_negatives", 15)),
        "use_adjacent_positive": bool(mo.get("use_adjacent_positive", True)),
        "max_samples": int(mo.get("max_samples", 0)),
    }


def _load_memory_checkpoint(memory: LatentDenseMemoryGraph, checkpoint: str | Path) -> Dict[str, object]:
    if not checkpoint:
        return {"loaded": False, "reason": "no checkpoint supplied"}
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("memory", None) if isinstance(ckpt, dict) else None
    if state is None:
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    mem_state = memory.state_dict()
    loaded: List[str] = []
    skipped: List[Dict[str, str]] = []
    new_state = dict(mem_state)
    for key, dst in mem_state.items():
        candidates = [key, f"memory.{key}"]
        src = None
        used_key = None
        for ck in candidates:
            if ck in state:
                src = state[ck]
                used_key = ck
                break
        if src is None:
            skipped.append({"key": key, "reason": "missing"})
            continue
        if tuple(src.shape) == tuple(dst.shape):
            new_state[key] = src
            loaded.append(key)
        elif key in {"values", "retrieval_count"} and src.ndim == dst.ndim and src.shape[0] >= dst.shape[0] and src.shape[1:] == dst.shape[1:]:
            new_state[key] = src[: dst.shape[0]].clone()
            loaded.append(key + "[cropped]")
        else:
            skipped.append({"key": key, "reason": f"shape mismatch {used_key}: {tuple(src.shape)} -> {tuple(dst.shape)}"})
    memory.load_state_dict(new_state, strict=False)
    return {"loaded": True, "loaded_keys": loaded, "skipped": skipped}


def _set_trainable(memory: LatentDenseMemoryGraph, train_graph: bool, train_distiller: bool) -> Dict[str, object]:
    allowed = ["values", "query_proj", "key_proj", "ive", "reconstruct"]
    if train_distiller:
        allowed.append("distiller")
    if train_graph:
        allowed.append("graph")
    trainable, frozen = [], []
    for name, p in memory.named_parameters():
        ok = any(name == prefix or name.startswith(prefix + ".") for prefix in allowed)
        p.requires_grad_(ok)
        (trainable if ok else frozen).append(name)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "trainable_param_count": int(sum(p.numel() for p in memory.parameters() if p.requires_grad)),
        "frozen_param_count": int(sum(p.numel() for p in memory.parameters() if not p.requires_grad)),
    }


def _param_groups(memory: LatentDenseMemoryGraph, base_lr: float, weight_decay: float):
    values, proj, graph, distiller, other = [], [], [], [], []
    for name, p in memory.named_parameters():
        if not p.requires_grad:
            continue
        if name == "values":
            values.append(p)
        elif name.startswith("graph."):
            graph.append(p)
        elif name.startswith("distiller."):
            distiller.append(p)
        elif name.startswith(("query_proj", "key_proj", "ive", "reconstruct")):
            proj.append(p)
        else:
            other.append(p)
    groups = []
    if values:
        groups.append({"params": values, "lr": base_lr * 0.2, "weight_decay": 0.0, "name": "memory.values"})
    if proj:
        groups.append({"params": proj, "lr": base_lr, "weight_decay": weight_decay, "name": "memory.proj_ive_reconstruct"})
    if distiller:
        groups.append({"params": distiller, "lr": base_lr, "weight_decay": weight_decay, "name": "memory.distiller"})
    if graph:
        groups.append({"params": graph, "lr": base_lr * 0.5, "weight_decay": weight_decay, "name": "memory.graph"})
    if other:
        groups.append({"params": other, "lr": base_lr, "weight_decay": weight_decay, "name": "memory.other"})
    return groups


def _grad_norm(named_params: Iterable[Tuple[str, torch.nn.Parameter]], prefix: str) -> float:
    sq = 0.0
    for name, p in named_params:
        if name == prefix or name.startswith(prefix + "."):
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum().cpu())
    return float(math.sqrt(sq))


def retrieval_metrics(weights: torch.Tensor) -> Dict[str, float]:
    w = weights.detach().float()
    k = int(w.shape[-1])
    entropy = -(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(dim=-1).mean()
    return {
        "retrieval_top1_mass": float(w[..., :1].sum(dim=-1).mean().cpu()),
        "retrieval_top5_mass": float(w[..., : min(5, k)].sum(dim=-1).mean().cpu()),
        "retrieval_top10_mass": float(w[..., : min(10, k)].sum(dim=-1).mean().cpu()),
        "retrieval_uniform_top10_baseline": float(min(10, k) / max(1, k)),
        "retrieval_entropy": float(entropy.cpu()),
        "retrieval_entropy_div_logtopk": float((entropy / math.log(max(2, k))).cpu()),
    }


def graph_metrics(memory: LatentDenseMemoryGraph) -> Dict[str, float]:
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


def adjacent_contrastive_loss(memory: LatentDenseMemoryGraph, z_query: torch.Tensor, z_target: torch.Tensor, tau: float, num_negatives: int) -> torch.Tensor:
    # Positive is the next span in the same sample. This avoids the degenerate
    # identity contrastive target that made the earlier V2 loss too easy.
    if z_query.shape[1] < 2:
        return z_query.new_zeros(())
    q = F.normalize(memory.query_proj(z_query[:, :-1, :].reshape(-1, z_query.shape[-1])).float(), dim=-1)
    pos = F.normalize(memory.key_proj(z_target[:, 1:, :].detach().reshape(-1, z_target.shape[-1])).float(), dim=-1)
    pool = F.normalize(memory.key_proj(z_target.detach().reshape(-1, z_target.shape[-1])).float(), dim=-1)
    m = q.shape[0]
    p = pool.shape[0]
    if m == 0 or p == 0:
        return z_query.new_zeros(())
    negs = []
    # Deterministic cyclic negatives are stable under tests and do not require
    # expensive metadata. They come from other spans/samples in the batch.
    for j in range(max(1, int(num_negatives))):
        shift = 2 + j
        idx = (torch.arange(m, device=z_query.device) + shift) % p
        negs.append(pool[idx])
    neg = torch.stack(negs, dim=1)  # [M,N,Q]
    pos_logit = (q * pos).sum(dim=-1, keepdim=True) / tau
    neg_logits = torch.einsum("mq,mnq->mn", q, neg) / tau
    logits = torch.cat([pos_logit, neg_logits], dim=-1)
    labels = torch.zeros(m, device=z_query.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def graph_coactivation_loss(memory: LatentDenseMemoryGraph, z_query: torch.Tensor) -> torch.Tensor:
    if z_query.shape[1] < 2:
        return z_query.new_zeros(())
    pack = memory.overview(z_query)
    src = pack["src_probs"].float()
    edge = pack["edge_probs"].float()
    a = src[:, :-1, :].reshape(-1, src.shape[-1])
    b_pos = src[:, 1:, :].reshape(-1, src.shape[-1])
    b_neg = b_pos.roll(shifts=max(1, b_pos.shape[0] // 2), dims=0)
    p_pos = torch.einsum("ng,gh,nh->n", a, edge, b_pos).clamp(1e-6, 1 - 1e-6)
    p_neg = torch.einsum("ng,gh,nh->n", a, edge, b_neg).clamp(1e-6, 1 - 1e-6)
    # BCE positive/negative plus a mild non-uniformity pressure. This is not
    # pure entropy minimization; content adjacency gives the direction.
    bce = -(p_pos.log().mean() + (1.0 - p_neg).log().mean())
    entropy = -(edge.clamp_min(1e-9) * edge.clamp_min(1e-9).log()).sum(dim=-1).mean() / math.log(edge.shape[-1])
    return bce + 0.01 * entropy


def train_step(memory: LatentDenseMemoryGraph, batch: Dict[str, torch.Tensor], cfg: Dict[str, float], device: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    zq = batch["z_query"].to(device)
    zt = batch["z_target"].to(device)
    out = memory.retrieve(zq, top_k=int(cfg["top_k"]))
    retrieved = out["retrieved"]
    pred = memory.reconstruct(retrieved)
    recon = F.mse_loss(pred.float(), zt.detach().float())
    if getattr(memory, "distiller", None) is not None and float(cfg["distiller_weight"]) > 0:
        insight = memory.distiller(zq, zt.detach(), retrieved.detach())
        distill = F.mse_loss(insight.float(), zt.detach().float())
    else:
        distill = zq.new_zeros(())
    contrast = adjacent_contrastive_loss(memory, zq, zt, float(cfg["contrastive_tau"]), int(cfg["num_negatives"])) if float(cfg["contrastive_weight"]) > 0 else zq.new_zeros(())
    graph = graph_coactivation_loss(memory, zq) if float(cfg["graph_weight"]) > 0 else zq.new_zeros(())
    norm_reg = (memory.values.float().norm(dim=-1).mean() - zt.detach().float().norm(dim=-1).mean()).abs()
    loss = (
        float(cfg["reconstruction_weight"]) * recon
        + float(cfg["contrastive_weight"]) * contrast
        + float(cfg["graph_weight"]) * graph
        + float(cfg["distiller_weight"]) * distill
        + float(cfg["norm_weight"]) * norm_reg
    )
    metrics = {
        "loss": float(loss.detach().float().cpu()),
        "recon_loss": float(recon.detach().float().cpu()),
        "contrastive_loss": float(contrast.detach().float().cpu()),
        "graph_loss": float(graph.detach().float().cpu()),
        "distiller_loss": float(distill.detach().float().cpu()),
        "norm_reg": float(norm_reg.detach().float().cpu()),
    }
    metrics.update(retrieval_metrics(out["weights"]))
    return loss, metrics


def write_report(path: str | Path, rows: List[Dict[str, float]], extra: Dict[str, object]) -> Dict[str, object]:
    series = MetricSeries(rows)
    keys = sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float))})
    summaries = {k: series.summary(k) for k in keys}
    recon = summaries.get("recon_loss") or {}
    ent = summaries.get("retrieval_entropy_div_logtopk") or {}
    top10 = summaries.get("retrieval_top10_mass") or {}
    drop = None
    if recon.get("first") not in (None, 0) and recon.get("last") is not None:
        drop = (float(recon["first"]) - float(recon["last"])) / abs(float(recon["first"]))
    decision_bits = {
        "recon_drop_ge_5pct": bool(drop is not None and drop >= 0.05),
        "retrieval_entropy_decreased": bool(ent.get("first") is not None and ent.get("last") is not None and float(ent["last"]) < float(ent["first"])),
        "retrieval_top10_over_0_18": bool(top10.get("last") is not None and float(top10["last"]) > 0.18),
    }
    decision = "PASS" if decision_bits["recon_drop_ge_5pct"] else "REVIEW"
    report = {"decision": decision, "decision_bits": decision_bits, "recon_loss_drop_fraction": drop, "summaries": summaries, **extra}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Phase 1.5B V3 Memory Objective Report\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train V3 local memory objectives: reconstruction + adjacent contrastive + graph + distiller.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--report", default="docs/PHASE1_5B_V3_MEMORY_REPORT.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = LatentLoopConfig.from_file(args.config)
    mo = _read_memory_objective_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = MemoryFeatureDataset(args.features, query_field=str(mo["query_field"]), target_field=str(mo["target_field"]), max_samples=int(mo["max_samples"]))
    if len(ds) < cfg.optim.micro_batch_size:
        raise ValueError(f"feature samples ({len(ds)}) < micro_batch_size ({cfg.optim.micro_batch_size}); DataLoader would produce zero batches")
    dl = DataLoader(ds, batch_size=cfg.optim.micro_batch_size, shuffle=True, drop_last=True)

    memory = LatentDenseMemoryGraph(cfg.memory).to(device)
    load_info = _load_memory_checkpoint(memory, args.checkpoint) if args.checkpoint else {"loaded": False, "reason": "no checkpoint supplied"}
    train_info = _set_trainable(memory, train_graph=float(mo["graph_weight"]) > 0, train_distiller=float(mo["distiller_weight"]) > 0)
    groups = _param_groups(memory, cfg.optim.lr, cfg.optim.weight_decay)
    opt = torch.optim.AdamW(groups, betas=(cfg.optim.beta1, cfg.optim.beta2))
    rows: List[Dict[str, float]] = []
    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")
    step = 0
    memory.train()
    pbar = tqdm(total=args.max_steps, desc="memory_v3")
    train_cfg = dict(mo)
    train_cfg["top_k"] = cfg.memory.top_k
    while step < args.max_steps:
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            loss, row = train_step(memory, batch, train_cfg, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite memory V3 loss at step {step}: {loss}")
            loss.backward()
            row["memory.values_grad_norm"] = _grad_norm(memory.named_parameters(), "values")
            row["memory.query_proj_grad_norm"] = _grad_norm(memory.named_parameters(), "query_proj")
            row["memory.key_proj_grad_norm"] = _grad_norm(memory.named_parameters(), "key_proj")
            row["memory.ive_grad_norm"] = _grad_norm(memory.named_parameters(), "ive")
            row["memory.reconstruct_grad_norm"] = _grad_norm(memory.named_parameters(), "reconstruct")
            row["memory.distiller_grad_norm"] = _grad_norm(memory.named_parameters(), "distiller")
            row["memory.graph.edge_logits_grad_norm"] = _grad_norm(memory.named_parameters(), "graph.edge_logits")
            torch.nn.utils.clip_grad_norm_([p for p in memory.parameters() if p.requires_grad], cfg.optim.grad_clip)
            opt.step()
            row.update(graph_metrics(memory))
            row["step"] = step
            rows.append(row)
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_f.flush()
            if step % 10 == 0:
                pbar.set_postfix(loss=row["loss"], recon=row["recon_loss"], ent=row["retrieval_entropy_div_logtopk"])
            step += 1
            pbar.update(1)
            if step >= args.max_steps:
                break
    metrics_f.close()
    pbar.close()
    ckpt_path = out_dir / "memory_v3_last.pt"
    torch.save({"memory": memory.state_dict(), "config": cfg.to_dict(), "step": step, "load_info": load_info, "train_info": train_info}, ckpt_path)
    extra = {
        "checkpoint_saved": str(ckpt_path),
        "metrics_path": str(metrics_path),
        "load_info": load_info,
        "train_info": train_info,
        "train_cfg": train_cfg,
        "optimizer_groups": [{"name": g.get("name", "group"), "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in groups],
        "feature_count": len(ds),
    }
    report = write_report(args.report, rows, extra)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
