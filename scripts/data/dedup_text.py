from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from scripts.data.common import read_jsonl, sha1_text, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    seen = set(); dropped = 0
    def rows():
        nonlocal dropped
        for rec in read_jsonl(args.input):
            h = sha1_text(rec.get("text", ""))
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            rec["text_hash"] = h
            yield rec
    n = write_jsonl(args.output, rows())
    print(json.dumps({"decision": "PASS", "kept": n, "dropped_duplicates": dropped}, indent=2))

if __name__ == "__main__":
    main()
