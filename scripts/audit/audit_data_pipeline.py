from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import load_cfg, tiny_data_file, write_json
from latentloop_pds_m.data import JsonlTokenDataset


def chunked_ce(logits, labels, chunk):
    shift_logits = logits[:, :-1].contiguous(); shift_labels = labels[:, 1:].contiguous()
    num = torch.zeros((), dtype=torch.float32); den = torch.zeros((), dtype=torch.float32)
    for start in range(0, shift_logits.shape[1], chunk):
        end = min(shift_logits.shape[1], start + chunk)
        lab = shift_labels[:, start:end].reshape(-1)
        valid = (lab != -100).sum().float()
        if valid.item() == 0: continue
        num = num + F.cross_entropy(shift_logits[:, start:end].reshape(-1, shift_logits.size(-1)).float(), lab, ignore_index=-100, reduction="sum")
        den = den + valid
    return num / den.clamp_min(1.0)


def run(config: str, data: str, num_batches: int, out: str | None = None):
    cfg = load_cfg(config)
    tmp = Path(tempfile.mkdtemp(prefix="ll_data_audit_"))
    data_path = Path(data) if data else tiny_data_file(cfg, tmp, n=max(8, num_batches), seq_len=cfg.transformer.max_seq_len)
    ds = JsonlTokenDataset(data_path, seq_len=cfg.transformer.max_seq_len)
    dl = DataLoader(ds, batch_size=cfg.optim.micro_batch_size, shuffle=False)
    batches=[]; blocked=[]
    for i,b in enumerate(dl):
        if i >= num_batches: break
        x=b["input_ids"]; labels=b["labels"]; mask=b["attention_mask"]
        valid = int((labels[:,1:] != -100).sum())
        pad_bad = int(((mask == 0) & (labels != -100)).sum())
        label_nonpad_mismatch = int(((mask == 1) & (labels != x)).sum())
        torch.manual_seed(123)
        logits = torch.randn(*x.shape, cfg.transformer.vocab_size)
        full = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:,1:].reshape(-1), ignore_index=-100)
        chk = chunked_ce(logits, labels, cfg.optim.chunked_ce_size)
        ce_delta = float((full - chk).abs())
        rec={"batch":i,"shape":list(x.shape),"valid_shifted_labels":valid,"pad_bad":pad_bad,"label_nonpad_mismatch":label_nonpad_mismatch,"ce_delta":ce_delta,"first_input_ids":x[0,:min(16,x.shape[1])].tolist(),"first_labels":labels[0,:min(16,labels.shape[1])].tolist()}
        batches.append(rec)
        if valid <= 0: blocked.append(f"batch_{i}_no_valid_labels")
        if pad_bad: blocked.append(f"batch_{i}_pad_enters_loss")
        if label_nonpad_mismatch: blocked.append(f"batch_{i}_label_mismatch")
        if ce_delta > 1e-5: blocked.append(f"batch_{i}_chunked_ce_delta")
    report={"config":config,"data":str(data_path),"num_records":len(ds),"batches":batches,"blocked_reasons":blocked,"decision":"PASS" if not blocked else "BLOCKED"}
    if out: write_json(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False)); return report

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--data",default=""); ap.add_argument("--num_batches",type=int,default=4); ap.add_argument("--out",default="")
    a=ap.parse_args(); run(a.config,a.data,a.num_batches,a.out or None)
