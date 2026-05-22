from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = ["你好", "请用一句话介绍你自己。", "1+1=", "苹果是一种"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local_v3_16k_phase1_4070.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/local_v3_32k/tokenizer.model")
    ap.add_argument("--out", default="docs/LOCAL_V3_GENERATION_SMOKE.json")
    args = ap.parse_args()
    results = []
    prompts_file = ROOT / "outputs" / "generation_smoke_prompts.txt"
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    prompts_file.write_text("\n".join(PROMPTS), encoding="utf-8")
    report = ROOT / "docs" / "LOCAL_V3_GENERATION_REPORT.md"
    cmd = [sys.executable, "scripts/generate_local_v3.py", "--config", args.config, "--checkpoint", args.checkpoint, "--tokenizer", args.tokenizer, "--prompts_file", str(prompts_file), "--max_new_tokens", "32", "--out", str(report)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    ok = proc.returncode == 0
    decoded_ok = False
    if ok:
        decoded_ok = report.exists() and len(report.read_text(encoding="utf-8")) > 200
    out = {"decision": "PASS" if ok and decoded_ok else "FAIL", "cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:], "report": str(report)}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out["decision"] != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
