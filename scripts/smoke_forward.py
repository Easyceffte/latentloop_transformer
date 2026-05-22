from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "smoke_tiny.yaml"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--backward", action="store_true")
    args = ap.parse_args()
    cfg = LatentLoopConfig.from_file(args.config)
    model = LatentLoopTransformerPDSM(cfg)
    input_ids = torch.randint(3, cfg.transformer.vocab_size, (args.batch, args.seq))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels, global_step=0)
    print("param_count", model.parameter_count())
    print("loss", float(out["loss"].detach()))
    for k, v in sorted(out["aux"].items()):
        if torch.is_tensor(v):
            print(k, float(v.detach().float().mean()))
    if args.backward:
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print("grad_norm", float(grad_norm))


if __name__ == "__main__":
    main()
