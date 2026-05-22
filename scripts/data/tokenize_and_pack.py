from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from scripts.data.common import BOS_ID, EOS_ID, PAD_ID, read_jsonl, sha1_ids, write_jsonl


def load_sp(model_path: str):
    from scripts.data.tokenizer_io import load_tokenizer
    return load_tokenizer(model_path)


def pack_records(records: Iterable[Dict[str, Any]], tokenizer_model: str, seq_len: int, target_blocks: int | None = None, min_valid_tokens: int = 480) -> Iterator[Dict[str, Any]]:
    sp = load_sp(tokenizer_model)
    current: List[int] = []
    current_sources: Counter[str] = Counter()
    current_domains: Counter[str] = Counter()
    doc_count = 0
    made = 0

    def meta_source(sources: Counter[str]) -> str:
        return sources.most_common(1)[0][0] if sources else "unknown"
    def meta_domain(domains: Counter[str]) -> str:
        return domains.most_common(1)[0][0] if domains else "unknown"
    def emit(ids: List[int], sources: Counter[str], domains: Counter[str], docs: int):
        valid = len(ids)
        if valid < seq_len:
            ids = ids + [PAD_ID] * (seq_len - valid)
        return {
            "input_ids": ids,
            "source": meta_source(sources),
            "domain": meta_domain(domains),
            "valid_tokens": valid,
            "doc_count": max(1, docs),
            "block_hash": sha1_ids(ids),
        }

    def maybe_yield(row):
        nonlocal made
        made += 1
        return row
    def emit_current_if_valid():
        nonlocal current, current_sources, current_domains, doc_count
        if current and len(current) >= min_valid_tokens:
            row = maybe_yield(emit(current, current_sources, current_domains, doc_count))
        else:
            row = None
        current = []
        current_sources = Counter()
        current_domains = Counter()
        doc_count = 0
        return row

    for rec in records:
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        token_ids = list(map(int, sp.encode(text, out_type=int)))
        source = str(rec.get("source", "unknown"))
        domain = str(rec.get("domain", "unknown"))
        # Long documents are split into independent BOS...EOS chunks so every block is self-delimiting.
        if len(token_ids) + 2 > seq_len:
            if current:
                row = emit_current_if_valid()
                if row is not None:
                    yield row
                    if target_blocks is not None and made >= target_blocks:
                        return
            chunk = seq_len - 2
            for start in range(0, len(token_ids), chunk):
                seg = [BOS_ID] + token_ids[start : start + chunk] + [EOS_ID]
                if len(seg) < min_valid_tokens:
                    continue
                yield maybe_yield(emit(seg, Counter({source: 1}), Counter({domain: 1}), 1))
                if target_blocks is not None and made >= target_blocks:
                    return
            continue
        ids = [BOS_ID] + token_ids + [EOS_ID]
        if current and len(current) + len(ids) > seq_len:
            row = emit_current_if_valid()
            if row is not None:
                yield row
                if target_blocks is not None and made >= target_blocks:
                    return
        current.extend(ids)
        current_sources[source] += 1
        current_domains[domain] += 1
        doc_count += 1
    if current and (target_blocks is None or made < target_blocks):
        if len(current) >= min_valid_tokens:
            yield maybe_yield(emit(current, current_sources, current_domains, doc_count))

def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize normalized text JSONL and pack into fixed token blocks.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--target_blocks", type=int, default=None)
    args = ap.parse_args()
    blocks = pack_records(read_jsonl(args.input), args.tokenizer, args.seq_len, args.target_blocks)
    n = write_jsonl(args.output, blocks)
    print(json.dumps({"decision": "PASS", "blocks": n, "output": args.output}, indent=2))

if __name__ == "__main__":
    main()
