from __future__ import annotations
import argparse, json
import torch
from common import load_cfg, resolve_device, dtype_for, build_model, write_json


def run(config: str, device: str, max_new_tokens: int, out: str | None = None):
    cfg=load_cfg(config); device=resolve_device(device)
    if device=="cpu": torch.set_num_threads(1)
    dt=dtype_for(cfg,device)
    model=build_model(cfg,device,dt).eval()
    ids=torch.tensor([[3,4,5,6]],device=device,dtype=torch.long)
    with torch.no_grad():
        torch.manual_seed(77)
        out_ids=model.generate(ids,max_new_tokens=max_new_tokens,global_step=max(cfg.dense_feedback.warmup_steps+1,1))
        torch.manual_seed(88)
        logits1=model(ids,labels=None,global_step=cfg.dense_feedback.warmup_steps+1)["logits"].detach().float()
        torch.manual_seed(88)
        logits2=model(torch.tensor([[3,4,5,99]],device=device),labels=None,global_step=cfg.dense_feedback.warmup_steps+1)["logits"].detach().float()
    prefix_delta=float((logits1[:,:3]-logits2[:,:3]).abs().max())
    finite=bool(torch.isfinite(logits1).all().item()) and bool(torch.isfinite(out_ids.float()).all().item())
    blocked=[]
    if not finite: blocked.append("nonfinite_generate")
    if prefix_delta>5e-4: blocked.append("generation_prefix_leakage")
    report={"config":config,"device":device,"generated_ids":out_ids.detach().cpu().tolist(),"finite":finite,"prefix_delta":prefix_delta,"blocked_reasons":blocked,"decision":"PASS" if not blocked else "BLOCKED"}
    if out: write_json(out,report)
    print(json.dumps(report,indent=2,ensure_ascii=False)); return report

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cpu"); ap.add_argument("--max_new_tokens",type=int,default=2); ap.add_argument("--out",default="")
    a=ap.parse_args(); run(a.config,a.device,a.max_new_tokens,a.out or None)
