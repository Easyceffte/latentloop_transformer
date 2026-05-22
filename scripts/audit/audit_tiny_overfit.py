from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from common import load_cfg, resolve_device, dtype_for, build_model, make_batch, write_json
from latentloop_pds_m.optim import build_wsd_scheduler


def run(config: str, device: str, seq: int, steps: int, out_dir: str | None = None):
    cfg=load_cfg(config); device=resolve_device(device)
    if device=="cpu": torch.set_num_threads(1)
    dt=dtype_for(cfg,device)
    torch.manual_seed(321)
    model=build_model(cfg,device,dt).train()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.optim.lr,betas=(cfg.optim.beta1,cfg.optim.beta2),weight_decay=cfg.optim.weight_decay)
    sched=build_wsd_scheduler(opt,cfg.optim.warmup_steps,cfg.optim.total_steps,cfg.optim.stable_until_ratio,cfg.optim.min_lr/cfg.optim.lr)
    batch=make_batch(cfg,seq,device)
    rows=[]
    for s in range(steps):
        out=model(**batch,global_step=s); out["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.optim.grad_clip); opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        row={"optimizer_step":s,"loss":float(out["loss"].detach().float()),"lm_loss":float(out["lm_loss"].detach().float())}
        for k in ["ddpm_loss_raw","slot_diversity_loss_raw","shortcut_consistency_loss_raw","exit_entropy_raw"]:
            if k in out: row[k]=float(out[k].detach().float())
        for k,v in out["aux"].items():
            if torch.is_tensor(v) and k in {"gate_inf_mean","gate_write_mean","gate_read_mean","write_skip_rate","top10_concentration"}:
                row[k]=float(v.detach().float())
        rows.append(row)
    initial=rows[0]["lm_loss"]; final=rows[-1]["lm_loss"]
    finite=all(torch.isfinite(torch.tensor([r["loss"] for r in rows])).tolist())
    blocked=[]
    if not finite: blocked.append("nonfinite_loss")
    if steps>=20 and final >= initial: blocked.append("lm_loss_not_decreased")
    report={"config":config,"device":device,"seq":seq,"steps":steps,"initial_lm_loss":initial,"final_lm_loss":final,"rows_tail":rows[-5:],"blocked_reasons":blocked,"decision":"PASS" if not blocked else "BLOCKED"}
    if out_dir:
        od=Path(out_dir); od.mkdir(parents=True,exist_ok=True)
        if rows:
            with (od/"tiny_overfit.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
        write_json(od/"tiny_overfit.json",report)
    print(json.dumps(report,indent=2,ensure_ascii=False)); return report

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cpu"); ap.add_argument("--seq",type=int,default=64); ap.add_argument("--steps",type=int,default=8); ap.add_argument("--out_dir",default="")
    a=ap.parse_args(); run(a.config,a.device,a.seq,a.steps,a.out_dir or None)
