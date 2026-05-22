from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import torch
from common import load_cfg, resolve_device, dtype_for, build_model, make_batch, write_json
from latentloop_pds_m.optim import build_wsd_scheduler


def train_one(model,opt,sched,cfg,batch,opt_step):
    out=model(**batch,global_step=opt_step); (out["loss"]/cfg.optim.grad_accum).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip); opt.step(); opt.zero_grad(set_to_none=True); sched.step(); return out


def run(config: str, device: str, seq: int, out: str | None = None):
    cfg=load_cfg(config); device=resolve_device(device)
    if device=="cpu": torch.set_num_threads(1)
    dt=dtype_for(cfg,device)
    torch.manual_seed(123)
    model=build_model(cfg,device,dt).train(); opt=torch.optim.AdamW(model.parameters(),lr=cfg.optim.lr,betas=(cfg.optim.beta1,cfg.optim.beta2),weight_decay=cfg.optim.weight_decay); sched=build_wsd_scheduler(opt,cfg.optim.warmup_steps,cfg.optim.total_steps,cfg.optim.stable_until_ratio,cfg.optim.min_lr/cfg.optim.lr)
    batch=make_batch(cfg,seq,device)
    out1=train_one(model,opt,sched,cfg,batch,0); opt_step=1
    ckpt={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"config":cfg.to_dict(),"micro_step":1,"optimizer_step":opt_step,"rng_state":{"torch":torch.get_rng_state()}}
    if torch.cuda.is_available(): ckpt["rng_state"]["cuda"]=torch.cuda.get_rng_state_all()
    tmp=Path(tempfile.mkdtemp(prefix="ll_resume_"))/"ckpt.pt"; torch.save(ckpt,tmp)
    lr_before=opt.param_groups[0]["lr"]
    theta_before=model.memory.current_write_threshold(opt_step)
    model2=build_model(cfg,device,dt).train(); opt2=torch.optim.AdamW(model2.parameters(),lr=cfg.optim.lr,betas=(cfg.optim.beta1,cfg.optim.beta2),weight_decay=cfg.optim.weight_decay); sched2=build_wsd_scheduler(opt2,cfg.optim.warmup_steps,cfg.optim.total_steps,cfg.optim.stable_until_ratio,cfg.optim.min_lr/cfg.optim.lr)
    loaded=torch.load(tmp,map_location=device); model2.load_state_dict(loaded["model"]); opt2.load_state_dict(loaded["optimizer"]); sched2.load_state_dict(loaded["scheduler"]); loaded_step=int(loaded["optimizer_step"])
    lr_loaded=opt2.param_groups[0]["lr"]; theta_loaded=model2.memory.current_write_threshold(loaded_step)
    out2=train_one(model2,opt2,sched2,cfg,batch,loaded_step)
    report={"config":config,"device":device,"ckpt_path":str(tmp),"loaded_optimizer_step":loaded_step,"lr_before":lr_before,"lr_loaded":lr_loaded,"theta_before":theta_before,"theta_loaded":theta_loaded,"loss_before":float(out1["loss"].detach().float()),"loss_after_resume_step":float(out2["loss"].detach().float()),"has_optimizer_state":bool(loaded.get("optimizer")),"has_scheduler_state":bool(loaded.get("scheduler")),"has_rng_state":bool(loaded.get("rng_state"))}
    blocked=[]
    if loaded_step!=opt_step: blocked.append("optimizer_step_not_restored")
    if abs(lr_before-lr_loaded)>1e-12: blocked.append("lr_not_continuous")
    if abs(theta_before-theta_loaded)>1e-12: blocked.append("theta_not_continuous")
    if not report["has_optimizer_state"]: blocked.append("missing_optimizer_state")
    if not report["has_scheduler_state"]: blocked.append("missing_scheduler_state")
    if not report["has_rng_state"]: blocked.append("missing_rng_state")
    report["blocked_reasons"]=blocked; report["decision"]="PASS" if not blocked else "BLOCKED"
    if out: write_json(out,report)
    print(json.dumps(report,indent=2,ensure_ascii=False)); return report

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cpu"); ap.add_argument("--seq",type=int,default=64); ap.add_argument("--out",default="")
    a=ap.parse_args(); run(a.config,a.device,a.seq,a.out or None)
