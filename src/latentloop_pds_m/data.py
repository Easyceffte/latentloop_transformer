from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


class JsonlTokenDataset(Dataset):
    """Simple JSONL dataset.

    Accepted records:
    - {"input_ids": [...]} for pre-tokenized causal LM data
    - {"text": "..."} with a tokenizer callable passed in
    - {"prompt": "...", "response": "..."} with tokenizer callable passed in
    """

    def __init__(self, path: str | Path, seq_len: int, tokenizer=None, add_eos: bool = True):
        self.path = Path(path)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"No records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, rec: Dict) -> List[int]:
        if "input_ids" in rec:
            ids = list(map(int, rec["input_ids"]))
        else:
            if self.tokenizer is None:
                raise ValueError("A tokenizer is required for text/prompt records.")
            if "prompt" in rec and "response" in rec:
                text = rec["prompt"] + rec.get("separator", "\n") + rec["response"]
            else:
                text = rec.get("text", "")
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if self.add_eos and eos is not None:
                ids.append(int(eos))
        if len(ids) < 2:
            ids = ids + [0, 0]
        return ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self._encode(self.records[idx])
        if len(ids) >= self.seq_len:
            start = random.randint(0, max(0, len(ids) - self.seq_len))
            ids = ids[start : start + self.seq_len]
        else:
            ids = ids + [0] * (self.seq_len - len(ids))
        x = torch.tensor(ids, dtype=torch.long)
        labels = x.clone()
        labels[x == 0] = -100
        mask = (x != 0).long()
        return {"input_ids": x, "labels": labels, "attention_mask": mask}


def make_synthetic_jsonl(path: str | Path, n: int = 256, vocab_size: int = 512, seq_len: int = 64) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            mode = i % 3
            if mode == 0:
                base = [rng.randint(3, vocab_size - 1) for _ in range(seq_len)]
            elif mode == 1:
                a = rng.randint(3, vocab_size - 30)
                base = [(a + j) % vocab_size for j in range(seq_len)]
                base = [x if x >= 3 else x + 3 for x in base]
            else:
                pattern = [rng.randint(3, vocab_size - 1) for _ in range(8)]
                base = (pattern * ((seq_len // len(pattern)) + 1))[:seq_len]
            f.write(json.dumps({"input_ids": base}, ensure_ascii=False) + "\n")
