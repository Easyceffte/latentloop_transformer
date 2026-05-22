from __future__ import annotations
import argparse, json
import torch
from common import load_cfg, resolve_device, dtype_for, build_model, make_batch, write_json


def run(config: str, device: str, seq: int, out: str | None = None):
    cfg=load_cfg(config); device=resolve_device(device)
    if device=="cpu": torch.set_num_threads(1)
    dt=dtype_for(cfg,device)
    model=build_model(cfg,device,dt).train()
    mem=model.memory
    stats={"candidate_shapes":[],"retrieve_calls":0,"exact_score_shape":None}
    orig_candidate=mem._candidate_matrix_lsh
    def wrapped_candidate(q,max_candidates):
        cand=orig_candidate(q,max_candidates)
        stats["candidate_shapes"].append(list(cand.shape)); return cand
    mem._candidate_matrix_lsh=wrapped_candidate
    batch=make_batch(cfg,seq,device)
    if device=="cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    out_d=model(**batch,global_step=max(1,cfg.dense_feedback.warmup_steps+1))
    out_d["loss"].backward()
    stats["retrieve_calls"]=len(stats["candidate_shapes"])
    largest=0
    for sh in stats["candidate_shapes"]:
        n=1
        for x in sh: n*=int(x)
        largest=max(largest,n)
    report={"config":config,"device":device,"seq":seq,"n_slots":cfg.memory.n_slots,"top_k":cfg.memory.top_k,"use_lsh":cfg.memory.use_lsh,"exact_threshold":cfg.memory.exact_threshold,"lsh_query_chunk":cfg.memory.lsh_query_chunk,"lsh_max_candidates":cfg.memory.lsh_max_candidates,"candidate_shapes":stats["candidate_shapes"][:16],"largest_candidate_numel":largest,"finite_loss":bool(torch.isfinite(out_d["loss"].detach()).all().item())}
    if device=="cuda": report.update({"peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9,"peak_reserved_gb":torch.cuda.max_memory_reserved()/1e9})
    blocked=[]
    if cfg.memory.n_slots>cfg.memory.exact_threshold and not cfg.memory.use_lsh: blocked.append("large_memory_exact_retrieval")
    if largest > cfg.memory.lsh_query_chunk * max(cfg.memory.lsh_max_candidates, cfg.memory.top_k): blocked.append("candidate_shape_exceeds_config")
    if not report["finite_loss"]: blocked.append("nonfinite_loss")
    report["blocked_reasons"]=blocked; report["decision"]="PASS" if not blocked else "BLOCKED"
    if out: write_json(out,report)
    print(json.dumps(report,indent=2,ensure_ascii=False)); return report

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",default="cpu"); ap.add_argument("--seq",type=int,default=64); ap.add_argument("--out",default="")
    a=ap.parse_args(); run(a.config,a.device,a.seq,a.out or None)
