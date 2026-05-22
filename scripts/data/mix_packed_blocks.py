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
from typing import Dict, List

from scripts.data.common import DOMAIN_RATIOS, allocate_counts, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix packed blocks into the target domain ratio.")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more packed JSONL files.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--total_blocks", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    by_domain: Dict[str, List[dict]] = {"general": [], "reasoning": [], "divergent": []}
    for path in args.inputs:
        for rec in read_jsonl(path):
            d = rec.get("domain", "unknown")
            if d in by_domain:
                by_domain[d].append(rec)
    counts = allocate_counts(args.total_blocks, DOMAIN_RATIOS)
    out = []
    blocked = []
    for d, n in counts.items():
        rows = by_domain[d]
        if len(rows) < n:
            blocked.append(f"domain {d} has {len(rows)} blocks, need {n}")
        rng.shuffle(rows)
        out.extend(rows[:n])
    if blocked:
        print(json.dumps({"decision": "BLOCKED", "blocked_reasons": blocked}, indent=2))
        raise SystemExit(2)
    rng.shuffle(out)
    n = write_jsonl(args.output, out)
    print(json.dumps({"decision": "PASS", "blocks": n, "domain_counts": dict(Counter(r.get("domain") for r in out))}, indent=2))

if __name__ == "__main__":
    main()
