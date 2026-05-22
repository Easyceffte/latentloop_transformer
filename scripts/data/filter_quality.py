from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from collections import Counter

from scripts.data.common import passes_quality, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    counts = Counter()
    def rows():
        for rec in read_jsonl(args.input):
            ok, reason = passes_quality(rec.get("text", ""), rec.get("domain", "general"))
            counts[reason] += 1
            if ok:
                yield rec
    n = write_jsonl(args.output, rows())
    print(json.dumps({"decision": "PASS", "kept": n, "reasons": dict(counts)}, indent=2))

if __name__ == "__main__":
    main()
