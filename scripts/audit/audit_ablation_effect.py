from __future__ import annotations
import argparse, json
import types
import torch
from common import load_cfg, resolve_device, dtype_for, build_model, make_batch, write_json


def logits_for(model,batch,global_step,seed=999):
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(**batch,global_step=global_step)["logits"].detach().float()


def run(config: str, device: str, seq: int, out: str | None = None):
    cfg=load_cfg(config); device=resolve_device(device)
    if device=="cpu": torch.set_num_threads(1)
    dt=dtype_for(cfg,device)
    model=build_model(cfg,device,dt).eval(); batch=make_batch(cfg,seq,device); gs=max(1,cfg.dense_feedback.warmup_steps+1)
    base=logits_for(model,batch,gs)
    results={}
    # Dense feedback off.
    fb=model.feedback; model.feedback=None
    nofb=logits_for(model,batch,gs); model.feedback=fb
    results["dense_feedback_off_l2"]=float((base-nofb).pow(2).mean().sqrt())
    # Memory read off.
    orig_retrieve=model.memory.retrieve
    def zero_retrieve(z_query, top_k=None, global_step=None):
        b,l,d=z_query.shape; k=top_k or cfg.memory.top_k
        return {"retrieved":torch.zeros_like(z_query),"indices":torch.zeros(b,l,k,device=z_query.device,dtype=torch.long),"weights":torch.zeros(b,l,k,device=z_query.device,dtype=z_query.dtype),"top10_concentration":z_query.new_zeros(())}
    model.memory.retrieve=zero_retrieve
    nomem=logits_for(model,batch,gs); model.memory.retrieve=orig_retrieve
    results["memory_read_off_l2"]=float((base-nomem).pow(2).mean().sqrt())
    # Slots off.
    saved=[]
    for layer in model.layers:
        if hasattr(layer,"diffusion"):
            saved.append((layer.diffusion, layer.diffusion.make_slots))
            def make_zero_slots(self,z):
                b,l,d=z.shape; return z.new_zeros((b,self.cfg.idea_slots,d)), z.new_zeros(z.shape)
            layer.diffusion.make_slots=types.MethodType(make_zero_slots, layer.diffusion)
    noslots=logits_for(model,batch,gs)
    for obj,fn in saved: obj.make_slots=fn
    results["idea_slots_off_l2"]=float((base-noslots).pow(2).mean().sqrt())
    blocked=[]
    # Very small random-init deltas can be valid, but exact zero means a dead switch.
    for k,v in results.items():
        if v == 0.0: blocked.append(k.replace("_l2","_no_effect"))
    report={"config":config,"device":device,"seq":seq,"deltas":results,"blocked_reasons":blocked,"decision":"PASS" if not blocked else "BLOCKED"}
    if out: write_json(out,report)
    print(json.dumps(report,indent=2,ensure_ascii=False)); return report

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cpu"); ap.add_argument("--seq",type=int,default=64); ap.add_argument("--out",default="")
    a=ap.parse_args(); run(a.config,a.device,a.seq,a.out or None)
