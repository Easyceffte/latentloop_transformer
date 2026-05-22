from __future__ import annotations
import argparse, gc, json, sys, torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_cuda_memory import run as run_cuda_memory
from audit_data_pipeline import run as run_data_pipeline
from audit_memory_runtime import run as run_memory_runtime
from audit_ablation_effect import run as run_ablation_effect
from audit_resume_continuity import run as run_resume_continuity
from audit_tiny_overfit import run as run_tiny_overfit
from audit_generation_path import run as run_generation_path
from audit_loss_gradient_matrix import run as run_loss_gradient_matrix


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--out", default="reports/residual_audit")
    ap.add_argument("--tiny_steps", type=int, default=2)
    ap.add_argument("--data", default="")
    ap.add_argument("--skip_loss_matrix", action="store_true", help="Use only when debugging orchestration; individual loss-gradient audit should still be run before training.")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    steps = [
        ("cuda_memory", lambda: run_cuda_memory(args.config, args.device, args.seq, "auto", str(out / "cuda_memory.json"))),
        ("data_pipeline", lambda: run_data_pipeline(args.config, args.data, 4, str(out / "data_pipeline.json"))),
        ("memory_runtime", lambda: run_memory_runtime(args.config, args.device, args.seq, str(out / "memory_runtime.json"))),
        ("ablation_effect", lambda: run_ablation_effect(args.config, args.device, min(args.seq, 128), str(out / "ablation_effect.json"))),
        ("resume_continuity", lambda: run_resume_continuity(args.config, args.device, min(args.seq, 128), str(out / "resume_continuity.json"))),
        ("tiny_overfit", lambda: run_tiny_overfit(args.config, args.device, min(args.seq, 128), args.tiny_steps, str(out))),
        ("generation_path", lambda: run_generation_path(args.config, args.device, 2, str(out / "generation_path.json"))),
    ]
    if not args.skip_loss_matrix:
        # This is the heaviest CPU audit. Run it last so earlier reports are preserved if interrupted.
        steps.append(("loss_gradient_matrix", lambda: run_loss_gradient_matrix(args.config, args.device, min(args.seq, 128), str(out))))
    results = {}; blocked = []
    for name, fn in steps:
        print(f"[residual-audit] running {name}...", flush=True)
        try:
            result = fn()
            results[name] = {"decision": result.get("decision"), "blocked_reasons": result.get("blocked_reasons", [])}
            if result.get("decision") == "BLOCKED":
                blocked.extend([f"{name}:{x}" for x in result.get("blocked_reasons", [])])
        except Exception as exc:
            results[name] = {"decision": "SCRIPT_ERROR", "error": repr(exc)}
            blocked.append(f"{name}_script_error:{exc!r}")
        _cleanup()
    if args.skip_loss_matrix:
        blocked.append("loss_gradient_matrix_skipped")
    decision = "PASS_TO_REAL_DATA_SHORT_RUN" if not blocked else ("PASS_TO_100_STEP" if all("tiny_overfit" in b or "loss_gradient_matrix_skipped" in b for b in blocked) else "BLOCKED_DO_NOT_TRAIN")
    report = {"config": args.config, "device": args.device, "seq": args.seq, "results": results, "blocked_reasons": blocked, "decision": decision}
    (out / "residual_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "residual_decision.md").write_text(f"# {decision}\n\nBlocked reasons:\n" + "\n".join(f"- {b}" for b in blocked) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "blocked_reasons": blocked, "out": str(out)}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
