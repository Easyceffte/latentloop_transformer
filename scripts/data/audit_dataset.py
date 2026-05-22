from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from scripts.data.common import BOS_ID, DOMAIN_RATIOS, EOS_ID, PAD_ID, read_jsonl, sha1_ids


def is_tail_padding(ids: List[int]) -> bool:
    seen_pad = False
    for x in ids:
        if x == PAD_ID:
            seen_pad = True
        elif seen_pad:
            return False
    return True


def chunked_ce_equivalence(vocab_size: int = 128, seq_len: int = 32, chunk_size: int = 7) -> float:
    torch.manual_seed(123)
    logits = torch.randn(2, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (2, seq_len))
    labels[0, -3:] = -100
    full = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100, reduction="mean")
    total = torch.tensor(0.0)
    count = torch.tensor(0.0)
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        l = logits[:, start:end, :].contiguous().view(-1, vocab_size)
        y = labels[:, start:end].contiguous().view(-1)
        valid = (y != -100).sum()
        if valid.item() > 0:
            ce_sum = F.cross_entropy(l, y, ignore_index=-100, reduction="sum")
            total = total + ce_sum
            count = count + valid
    return float((full - total / count).abs().item())


def audit_one(path: Path, seq_len: int, vocab_size: int, expected_mix: Dict[str, float] | None) -> Dict:
    blocked: List[str] = []
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    hashes = []
    n = 0
    min_valid = 10**9
    invalid_token_count = 0
    bad_len = 0
    bad_pad = 0
    missing_eos = 0
    missing_bos = 0
    valid_label_blocks = 0
    for rec in read_jsonl(path):
        n += 1
        ids = rec.get("input_ids")
        if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
            blocked.append(f"line {n}: input_ids is not list[int]")
            continue
        if len(ids) != seq_len:
            bad_len += 1
        invalid_token_count += sum(1 for x in ids if x < 0 or x >= vocab_size)
        if not is_tail_padding(ids):
            bad_pad += 1
        valid = sum(1 for x in ids if x != PAD_ID)
        min_valid = min(min_valid, valid)
        if valid > 1:
            valid_label_blocks += 1
        if ids and ids[0] != BOS_ID:
            missing_bos += 1
        if EOS_ID not in ids[:valid]:
            missing_eos += 1
        domain_counts[str(rec.get("domain", "unknown"))] += 1
        source_counts[str(rec.get("source", "unknown"))] += 1
        hashes.append(rec.get("block_hash") or sha1_ids(ids))
    if n == 0:
        blocked.append("empty dataset")
    if bad_len:
        blocked.append(f"{bad_len} blocks have length != {seq_len}")
    if invalid_token_count:
        blocked.append(f"invalid_token_count={invalid_token_count}")
    if bad_pad:
        blocked.append(f"{bad_pad} blocks have non-tail padding")
    if valid_label_blocks == 0:
        blocked.append("no blocks with valid label positions")
    min_required = min(480, seq_len)
    if n and min_valid < min_required:
        blocked.append(f"min_valid_tokens {min_valid} < required {min_required}")
    if missing_bos > max(1, int(0.01 * max(1, n))):
        blocked.append(f"too many blocks missing BOS at position 0: {missing_bos}")
    if missing_eos > max(1, int(0.10 * max(1, n))):
        blocked.append(f"too many blocks missing EOS: {missing_eos}")
    dup_ratio = 0.0 if n == 0 else 1.0 - (len(set(hashes)) / n)
    if dup_ratio > 0.01:
        blocked.append(f"duplicate block ratio {dup_ratio:.4f} > 0.01")
    mix = {k: v / max(1, n) for k, v in domain_counts.items()}
    if expected_mix and n:
        for d, ratio in expected_mix.items():
            actual = mix.get(d, 0.0)
            if abs(actual - ratio) > 0.01:
                blocked.append(f"mix ratio for {d} {actual:.4f} differs from expected {ratio:.4f}")
    ce_delta = chunked_ce_equivalence()
    if ce_delta > 1e-5:
        blocked.append(f"chunked CE equivalence delta {ce_delta} > 1e-5")
    return {
        "path": str(path),
        "num_blocks": n,
        "num_tokens": n * seq_len,
        "domain_counts": dict(domain_counts),
        "source_counts": dict(source_counts),
        "mix_ratio": mix,
        "duplicate_block_ratio": dup_ratio,
        "invalid_token_count": invalid_token_count,
        "min_valid_tokens": min_valid if n else 0,
        "missing_bos": missing_bos,
        "missing_eos": missing_eos,
        "chunked_ce_delta": ce_delta,
        "blocked_reasons": blocked,
        "decision": "PASS" if not blocked else "BLOCKED",
        "hashes": hashes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit pre-tokenized packed LatentLoop JSONL datasets.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--expected_mix", default="general=0.60,dialogue=0.20,reasoning=0.15,divergent=0.05")
    ap.add_argument("--out", default="data/reports/audit_dataset.json")
    ap.add_argument("--strict_all_splits", action="store_true", help="Apply exact mix-ratio tolerance to val/test as well as train.")
    args = ap.parse_args()
    expected = {}
    if args.expected_mix:
        for part in args.expected_mix.split(","):
            if part.strip():
                k, v = part.split("=")
                expected[k.strip()] = float(v)
    report = {"seq_len": args.seq_len, "vocab_size": args.vocab_size, "splits": {}, "blocked_reasons": []}
    train = audit_one(Path(args.train), args.seq_len, args.vocab_size, expected)
    report["splits"]["train"] = {k: v for k, v in train.items() if k != "hashes"}
    report["blocked_reasons"].extend([f"train: {x}" for x in train["blocked_reasons"]])
    split_hashes = {"train": set(train["hashes"])}
    for split_name, path in [("val", args.val), ("test", args.test)]:
        if path:
            res = audit_one(Path(path), args.seq_len, args.vocab_size, expected if args.strict_all_splits else None)
            report["splits"][split_name] = {k: v for k, v in res.items() if k != "hashes"}
            report["blocked_reasons"].extend([f"{split_name}: {x}" for x in res["blocked_reasons"]])
            split_hashes[split_name] = set(res["hashes"])
    for a in split_hashes:
        for b in split_hashes:
            if a >= b:
                continue
            overlap = len(split_hashes[a] & split_hashes[b])
            if overlap:
                report["blocked_reasons"].append(f"{a}/{b} exact block overlap: {overlap}")
    report["decision"] = "PASS" if not report["blocked_reasons"] else "BLOCKED"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["blocked_reasons"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
