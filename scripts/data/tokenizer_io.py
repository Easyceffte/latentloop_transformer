from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from scripts.data.common import BOS_ID, EOS_ID, PAD_ID, UNK_ID

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

class HashFallbackTokenizer:
    """Deterministic offline tokenizer for CI/mock audits only.

    It is not a substitute for the required SentencePiece tokenizer used for real training.
    """
    def __init__(self, vocab_size: int = 512):
        if vocab_size <= 16:
            raise ValueError("fallback vocab_size must be > 16")
        self.vocab_size = int(vocab_size)
    def encode(self, text: str, out_type=int):
        ids = []
        for tok in TOKEN_RE.findall(text or ""):
            h = 2166136261
            for ch in tok:
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
            ids.append(4 + (h % (self.vocab_size - 4)))
        return ids
    def get_piece_size(self) -> int:
        return self.vocab_size


def save_fallback_tokenizer(path: str | Path, vocab_size: int) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"type": "hash_fallback", "vocab_size": int(vocab_size), "warning": "mock/offline audit only; use SentencePiece for real training"}, indent=2), encoding="utf-8")
    return p


def load_tokenizer(model_path: str | Path):
    p = Path(model_path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("type") == "hash_fallback":
            return HashFallbackTokenizer(int(data.get("vocab_size", 512)))
        raise ValueError(f"unknown tokenizer json type in {p}")
    import sentencepiece as spm  # type: ignore
    return spm.SentencePieceProcessor(model_file=str(p))
