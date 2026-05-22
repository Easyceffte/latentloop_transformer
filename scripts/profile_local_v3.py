from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from latentloop_pds_m.data import JsonlTokenDataset, make_synthetic_jsonl


def now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser(description="Profile Local V3 training hot spots on the actual GPU.")
    ap.add_argument("--config", default="configs/local_v3_16k_phase1_4070.yaml")
    ap.add_argument("--data", default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="docs/local_v3_profile_summary.json")
    args = ap.parse_args()
    cfg = LatentLoopConfig.from_file(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and cfg.optim.bf16 and torch.cuda.is_bf16_supported() else torch.float32
    model = LatentLoopTransformerPDSM(cfg).to(device=device, dtype=dtype if dtype != torch.float32 else None).train()
    data_path = Path(args.data) if args.data else ROOT / "outputs" / "profile_synth.jsonl"
    if args.synthetic or not data_path.exists():
        make_synthetic_jsonl(data_path, n=256, vocab_size=cfg.transformer.vocab_size, seq_len=cfg.transformer.max_seq_len)
    dl = DataLoader(JsonlTokenDataset(data_path, cfg.transformer.max_seq_len), batch_size=cfg.optim.micro_batch_size, shuffle=True, drop_last=True, pin_memory=(device=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, fused=(device=="cuda"))
    it = iter(dl)
    rows=[]
    for step in range(args.steps + args.warmup):
        t0=now()
        try:
            batch=next(it)
        except StopIteration:
            it=iter(dl); batch=next(it)
        t1=now()
        batch={k:v.to(device, non_blocking=True) for k,v in batch.items()}
        t2=now()
        out=model(**batch, global_step=step)
        t3=now()
        out["loss"].backward()
        t4=now()
        opt.step(); opt.zero_grad(set_to_none=True)
        t5=now()
        if step >= args.warmup:
            toks=int(batch["input_ids"].numel())
            rows.append({"step":step,"tokens":toks,"dataloader_ms":(t1-t0)*1000,"h2d_ms":(t2-t1)*1000,"forward_ms":(t3-t2)*1000,"backward_ms":(t4-t3)*1000,"optimizer_ms":(t5-t4)*1000,"step_ms":(t5-t0)*1000,"tokens_per_sec":toks/max(1e-9,t5-t0),"loss":float(out["loss"].detach().float().cpu())})
    def avg(k): return sum(r[k] for r in rows)/max(1,len(rows))
    summary={"device":device,"torch":torch.__version__,"steps":len(rows),"avg_step_ms":avg("step_ms"),"avg_tokens_per_sec":avg("tokens_per_sec"),"avg_dataloader_ms":avg("dataloader_ms"),"avg_forward_ms":avg("forward_ms"),"avg_backward_ms":avg("backward_ms"),"avg_optimizer_ms":avg("optimizer_ms"),"cuda_allocated_gb":torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else 0,"rows":rows}
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    md=Path("docs/LOCAL_V3_PROFILE_REPORT.md"); md.parent.mkdir(exist_ok=True); md.write_text("# Local V3 Profile Report\n\n```json\n"+json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2,ensure_ascii=False)+"\n```\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)[:4000])

if __name__ == "__main__": main()
