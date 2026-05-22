from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from scripts.data.common import load_yaml, write_jsonl
from scripts.data.run_data_pipeline import collect_source_docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample normalized/filterable raw docs from one configured source.")
    ap.add_argument("--config", default="data/sources/data_sources.yaml")
    ap.add_argument("--source", required=True)
    ap.add_argument("--num_docs", type=int, default=100)
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw_cache_dir", default="data/raw_cache/sample_raw_sources")
    ap.add_argument("--reuse_raw_cache", action="store_true")
    ap.add_argument("--force_redownload", action="store_true")
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--offline_mock", action="store_true")
    args = ap.parse_args()
    if args.reuse_raw_cache and args.force_redownload:
        raise ValueError("--reuse_raw_cache and --force_redownload are mutually exclusive")
    cfg = load_yaml(args.config)
    spec = cfg["sources"][args.source]
    raw_cache_path = Path(args.raw_cache_dir) / f"{args.source}.raw.jsonl"
    docs, cache_report = collect_source_docs(
        args.source,
        spec,
        args.num_docs,
        seed=42,
        offline_mock=args.offline_mock,
        raw_cache_path=raw_cache_path,
        reuse_raw_cache=args.reuse_raw_cache,
        force_redownload=args.force_redownload,
        hf_cache_dir=args.hf_cache_dir,
    )
    n = write_jsonl(args.output, docs)
    print(json.dumps({"decision": "PASS", "records": n, "output": args.output, "cache": cache_report}, indent=2))

if __name__ == "__main__":
    main()
