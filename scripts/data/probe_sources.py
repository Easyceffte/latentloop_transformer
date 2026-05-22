from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from scripts.data.common import load_yaml, make_mock_documents, selected_text_from_row


def probe_hf_source(name: str, spec: Dict[str, Any], max_rows: int) -> Dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        return {"status": "error", "error": f"datasets import failed: {e}"}
    try:
        hf_id = spec["hf_id"]
        subset = spec.get("subset")
        split = spec.get("split", "train")
        kwargs = {"split": split, "streaming": bool(spec.get("streaming", True))}
        ds = load_dataset(hf_id, subset, **kwargs) if subset else load_dataset(hf_id, **kwargs)
        rows = []
        selected_field = None
        for i, row in enumerate(ds):
            if i >= max_rows:
                break
            text, field = selected_text_from_row(dict(row), spec)
            if selected_field is None:
                selected_field = field
            rows.append({"fields": list(row.keys()), "selected_field": field, "text_len": len(text)})
        if not rows:
            return {"status": "error", "error": "no rows returned"}
        return {"status": "ok", "hf_id": hf_id, "subset": subset, "split": split, "selected_text_field": selected_field, "rows": rows}
    except Exception as e:
        return {"status": "error", "error": repr(e), "hf_id": spec.get("hf_id"), "subset": spec.get("subset")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe configured downloadable data sources.")
    ap.add_argument("--config", default="data/sources/data_sources.yaml")
    ap.add_argument("--out", default="data/reports/source_probe_report.json")
    ap.add_argument("--max_rows", type=int, default=3)
    ap.add_argument("--offline_mock", action="store_true", help="Use deterministic local mock rows; does not prove HF downloadability.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sources = cfg.get("sources", {})
    report: Dict[str, Any] = {"config": args.config, "offline_mock": args.offline_mock, "sources": {}, "blocked_reasons": []}
    for name, spec in sources.items():
        if args.offline_mock:
            rows = []
            for row in make_mock_documents(name, spec.get("domain", "general"), args.max_rows):
                text, field = selected_text_from_row(row, {"text_field": "text"})
                rows.append({"fields": list(row.keys()), "selected_field": field, "text_len": len(text)})
            res = {"status": "mock_ok", "rows": rows, "warning": "offline_mock does not verify remote downloadability"}
        else:
            res = probe_hf_source(name, spec, args.max_rows)
        report["sources"][name] = res
        if spec.get("required", False) and res.get("status") not in {"ok", "mock_ok"}:
            report["blocked_reasons"].append(f"required source {name} probe failed: {res.get('error')}")
    report["decision"] = "PASS" if not report["blocked_reasons"] else "BLOCKED"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["blocked_reasons"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
