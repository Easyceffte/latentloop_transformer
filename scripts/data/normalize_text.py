from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

from scripts.data.common import normalize_text, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    def rows():
        for rec in read_jsonl(args.input):
            rec["text"] = normalize_text(rec.get("text", ""))
            yield rec
    n = write_jsonl(args.output, rows())
    print(json.dumps({"decision": "PASS", "records": n, "output": args.output}, indent=2))

if __name__ == "__main__":
    main()
