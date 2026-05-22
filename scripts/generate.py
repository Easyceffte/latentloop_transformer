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
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ids", default="3,4,5,6")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--global_step", type=int, default=-1)
    args = ap.parse_args()
    cfg = LatentLoopConfig.from_file(args.config)
    model = LatentLoopTransformerPDSM(cfg)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
    ids = torch.tensor([[int(x) for x in args.ids.split(",")]], dtype=torch.long)
    gs = None if args.global_step < 0 else args.global_step
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, global_step=gs)
    print(out.tolist()[0])


if __name__ == "__main__":
    main()
