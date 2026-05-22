from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from scripts.data.tokenizer_io import load_tokenizer


def encode_prompt(tok, text: str, bos_id: int = 2) -> List[int]:
    ids = list(map(int, tok.encode(text, out_type=int)))
    return [bos_id] + ids if not ids or ids[0] != bos_id else ids


def decode_ids(tok, ids: List[int]) -> str:
    # SentencePiece has decode/decode_ids; fallback tokenizer is intentionally not reversible.
    if hasattr(tok, "decode"):
        try:
            return tok.decode([int(x) for x in ids if int(x) > 3])
        except TypeError:
            return tok.decode_ids([int(x) for x in ids if int(x) > 3])
    if hasattr(tok, "decode_ids"):
        return tok.decode_ids([int(x) for x in ids if int(x) > 3])
    return " ".join(map(str, ids))


def load_model(config: str, checkpoint: str, device: str) -> LatentLoopTransformerPDSM:
    cfg = LatentLoopConfig.from_file(config)
    dtype = torch.bfloat16 if device == "cuda" and cfg.optim.bf16 and torch.cuda.is_bf16_supported() else torch.float32
    model = LatentLoopTransformerPDSM(cfg).to(device=device, dtype=dtype if dtype != torch.float32 else None)
    if checkpoint:
        obj = torch.load(checkpoint, map_location="cpu")
        state = obj.get("model", obj) if isinstance(obj, dict) else obj
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate decoded text from a Local V3 checkpoint.")
    ap.add_argument("--config", default="configs/local_v3_16k_phase1_4070.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/local_v3_32k/tokenizer.model")
    ap.add_argument("--prompt", default="你好")
    ap.add_argument("--prompts_file", default="")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="docs/LOCAL_V3_GENERATION_REPORT.md")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    model = load_model(args.config, args.checkpoint, args.device)
    prompts = [args.prompt]
    if args.prompts_file:
        prompts = [x.strip() for x in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for prompt in prompts:
        ids = torch.tensor([encode_prompt(tok, prompt)], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        out_ids = out[0].detach().cpu().tolist()
        text = decode_ids(tok, out_ids)
        rows.append({"prompt": prompt, "ids": out_ids, "decoded": text, "num_tokens": len(out_ids)})
        print(json.dumps({"prompt": prompt, "decoded": text}, ensure_ascii=False))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Local V3 Generation Report", "", f"checkpoint: `{args.checkpoint}`", f"config: `{args.config}`", ""]
    for r in rows:
        lines += [f"## Prompt", "", f"```text\n{r['prompt']}\n```", "", "## Decoded", "", f"```text\n{r['decoded']}\n```", ""]
    p.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
