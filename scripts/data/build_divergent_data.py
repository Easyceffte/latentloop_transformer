from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from scripts.data.common import build_divergent_documents, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_docs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = write_jsonl(args.output, build_divergent_documents(args.num_docs, args.seed))
    print(json.dumps({"decision": "PASS", "records": n, "output": args.output}, indent=2))

if __name__ == "__main__":
    main()
