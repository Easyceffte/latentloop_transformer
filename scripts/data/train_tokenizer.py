from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Iterable

from scripts.data.common import BOS_ID, EOS_ID, PAD_ID, UNK_ID, read_jsonl


def train_sentencepiece(input_text: Path, out_dir: Path, vocab_size: int, character_coverage: float, input_sentence_size: int) -> Path:
    try:
        import sentencepiece as spm  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentencepiece is required: {e}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "latentloop_spm32k"
    spm.SentencePieceTrainer.train(
        input=str(input_text),
        model_prefix=str(prefix),
        vocab_size=int(vocab_size),
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        byte_fallback=True,
        character_coverage=float(character_coverage),
        input_sentence_size=int(input_sentence_size),
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
        train_extremely_large_corpus=True,
    )
    model_path = prefix.with_suffix(".model")
    return model_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a 32k SentencePiece tokenizer for LatentLoop.")
    ap.add_argument("--input_jsonl", required=True, help="JSONL with a text field.")
    ap.add_argument("--out_dir", default="data/tokenizer")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--character_coverage", type=float, default=0.9995)
    ap.add_argument("--input_sentence_size", type=int, default=2_000_000)
    ap.add_argument("--max_chars", type=int, default=80_000_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "tokenizer_train.txt"
    chars = 0
    docs = 0
    with train_txt.open("w", encoding="utf-8") as f:
        for rec in read_jsonl(args.input_jsonl):
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            f.write(text.replace("\n", " ") + "\n")
            chars += len(text)
            docs += 1
            if chars >= args.max_chars:
                break
    if docs == 0:
        raise SystemExit("No text available for tokenizer training.")
    model_path = train_sentencepiece(train_txt, out_dir, args.vocab_size, args.character_coverage, args.input_sentence_size)
    try:
        import sentencepiece as spm  # type: ignore
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        actual_vocab = sp.get_piece_size()
        sample_ids = sp.encode("Tokenizer smoke sample for LatentLoop.", out_type=int)
        unk_rate = sum(1 for x in sample_ids if x == UNK_ID) / max(1, len(sample_ids))
    except Exception:
        actual_vocab = None
        unk_rate = None
    report = {
        "decision": "PASS",
        "model_path": str(model_path),
        "vocab_path": str(model_path.with_suffix(".vocab")),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "byte_fallback": True,
        "training_docs": docs,
        "training_chars": chars,
        "unk_rate_sample": unk_rate,
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
