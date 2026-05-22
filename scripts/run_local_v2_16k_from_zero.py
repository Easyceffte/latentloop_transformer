from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str], *, env=None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def mtokens_to_stage(mtokens: int) -> str:
    if mtokens not in {1, 3, 5}:
        raise ValueError("Only --mtokens 1, 3, or 5 are supported by the named local_v2 stages. Use --data_stage to override.")
    return f"local_v2_{mtokens}m"


def default_paths(args):
    stage = args.data_stage or mtokens_to_stage(args.mtokens)
    train_jsonl = ROOT / "data" / "processed" / f"{stage}.train.jsonl"
    val_jsonl = ROOT / "data" / "processed" / f"{stage}.val.jsonl"
    test_jsonl = ROOT / "data" / "processed" / f"{stage}.test.jsonl"
    phase1_dir = ROOT / "outputs" / f"{stage}_phase1_local_v2_16k"
    ckpt = phase1_dir / "last.pt"
    features = ROOT / "data" / "processed" / f"{stage}.memory_features.pt"
    memory_dir = ROOT / "outputs" / f"{stage}_phase1_5b_v2_16k"
    return stage, train_jsonl, val_jsonl, test_jsonl, phase1_dir, ckpt, features, memory_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the local 4070 V2/V3-16K workflow from zero through memory objectives and Phase2 probes.")
    ap.add_argument("--stage", choices=["all", "data", "audit_data", "phase1", "features", "memory", "memory_v3", "phase2_data", "phase2_audit_data", "phase2_phase1_resume", "phase2_features", "phase2_memory_v3", "phase2_joint", "synthetic_memory_smoke"], default="all")
    ap.add_argument("--mtokens", type=int, default=1, choices=[1, 3, 5], help="Approximate train-token budget for seq_len=256.")
    ap.add_argument("--data_stage", default="", help="Override stage name, e.g. local_v2_1m.")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--phase1_config", default="configs/local_v2_16k_phase1_4070.yaml")
    ap.add_argument("--phase1_steps", type=int, default=512, help="Start small. Increase to 2000/5000 after smoke passes.")
    ap.add_argument("--memory_config", default="configs/phase1_5b_v2_memory_4070_16k.yaml")
    ap.add_argument("--memory_v3_config", default="configs/phase1_5b_v3_memory_4070_16k.yaml")
    ap.add_argument("--joint_config", default="configs/local_v3_16k_joint_4070.yaml")
    ap.add_argument("--memory_steps", type=int, default=50)
    ap.add_argument("--memory_v3_steps", type=int, default=500)
    ap.add_argument("--phase2_mtokens", type=int, default=3, choices=[3, 5])
    ap.add_argument("--phase2_steps", type=int, default=512)
    ap.add_argument("--phase2_joint_steps", type=int, default=256)
    ap.add_argument("--feature_samples", type=int, default=1000)
    ap.add_argument("--span_size", type=int, default=16)
    ap.add_argument("--reuse_raw_cache", action="store_true")
    ap.add_argument("--offline_mock", action="store_true", help="No internet: use deterministic mock docs; only for pipeline smoke.")
    ap.add_argument("--hf_cache_dir", default="")
    ap.add_argument("--skip_existing_data", action="store_true")
    ap.add_argument("--skip_existing_features", action="store_true")
    args = ap.parse_args()

    stage, train_jsonl, val_jsonl, test_jsonl, phase1_dir, ckpt, features, memory_dir = default_paths(args)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    summary = {
        "stage": stage,
        "seq_len": args.seq_len,
        "train_jsonl": str(train_jsonl),
        "phase1_dir": str(phase1_dir),
        "features": str(features),
        "memory_dir": str(memory_dir),
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "LOCAL_V2_16K_RUN_PLAN_ACTIVE.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def paths_for(stage_name: str):
        train = ROOT / "data" / "processed" / f"{stage_name}.train.jsonl"
        val = ROOT / "data" / "processed" / f"{stage_name}.val.jsonl"
        test = ROOT / "data" / "processed" / f"{stage_name}.test.jsonl"
        phase = ROOT / "outputs" / f"{stage_name}_phase1_local_v2_16k"
        ck = phase / "last.pt"
        feat = ROOT / "data" / "processed" / f"{stage_name}.memory_features.pt"
        mem = ROOT / "outputs" / f"{stage_name}_phase1_5b_v3_16k"
        return train, val, test, phase, ck, feat, mem

    def run_data_stage(stage_name: str):
        train, val, test, *_ = paths_for(stage_name)
        if args.skip_existing_data and train.exists() and val.exists() and test.exists():
            print(f"Data exists for {stage_name}; skipping data pipeline.")
            return
        cmd = [sys.executable, "scripts/data/run_data_pipeline.py", "--stage", stage_name, "--sources", "data/sources/local_v2_sources.yaml", "--seq_len", str(args.seq_len), "--tokenizer_out", "data/tokenizer/local_v2_32k", "--output_dir", "data/processed", "--report_dir", "data/reports", "--raw_cache_dir", "data/raw_cache", "--intermediate_dir", "data/intermediate", "--tokenizer_vocab_size", "32000"]
        if args.reuse_raw_cache:
            cmd.append("--reuse_raw_cache")
        if args.offline_mock:
            cmd.append("--offline_mock")
        if args.hf_cache_dir:
            cmd += ["--hf_cache_dir", args.hf_cache_dir]
        run(cmd, env=env)

    def run_audit_stage(stage_name: str):
        train, val, test, *_ = paths_for(stage_name)
        run([sys.executable, "scripts/data/audit_dataset.py", "--train", str(train), "--val", str(val), "--test", str(test), "--seq_len", str(args.seq_len), "--vocab_size", "32000", "--out", f"data/reports/audit_{stage_name}_local_v2.json"], env=env)

    if args.stage in {"all", "synthetic_memory_smoke"}:
        synth = ROOT / "data" / "processed" / "memory_features_synthetic_local_v2.pt"
        run([sys.executable, "scripts/make_synthetic_memory_features.py", "--num_samples", "64", "--num_spans", "8", "--d_mem", "256", "--seq_len", "128", "--span_size", "16", "--out", str(synth), "--report", "docs/LOCAL_V2_SYNTHETIC_FEATURE_REPORT.json"], env=env)
        run([sys.executable, "scripts/train_memory_v2.py", "--config", args.memory_config, "--features", str(synth), "--max_steps", "10", "--output_dir", "outputs/local_v2_synthetic_memory_smoke", "--report", "docs/LOCAL_V2_SYNTHETIC_MEMORY_SMOKE_REPORT.md"], env=env)
        if args.stage == "synthetic_memory_smoke":
            return

    if args.stage in {"all", "data"}:
        run_data_stage(stage)
        if args.stage == "data":
            return

    if args.stage in {"all", "audit_data"}:
        run_audit_stage(stage)
        if args.stage == "audit_data":
            return

    if args.stage in {"all", "phase1"}:
        run([sys.executable, "scripts/train.py", "--config", args.phase1_config, "--data", str(train_jsonl), "--max_steps", str(args.phase1_steps), "--save_every", "0", "--output_dir", str(phase1_dir)], env=env)
        if args.stage == "phase1":
            return

    if args.stage in {"all", "features"}:
        if args.skip_existing_features and features.exists():
            print(f"Features exist for {stage}; skipping feature extraction.")
        else:
            run([sys.executable, "scripts/extract_memory_features.py", "--config", args.phase1_config, "--checkpoint", str(ckpt), "--data", str(train_jsonl), "--seq_len", str(args.seq_len), "--max_samples", str(args.feature_samples), "--span_size", str(args.span_size), "--batch_size", "1", "--out", str(features), "--report", f"docs/LOCAL_V2_16K_FEATURE_EXTRACTION_{stage}.md"], env=env)
        if args.stage == "features":
            return

    if args.stage in {"all", "memory"}:
        run([sys.executable, "scripts/train_memory_v2.py", "--config", args.memory_config, "--features", str(features), "--checkpoint", str(ckpt), "--max_steps", str(args.memory_steps), "--output_dir", str(memory_dir), "--report", f"docs/LOCAL_V2_16K_MEMORY_V2_DRYRUN_{stage}.md"], env=env)
        if args.stage == "memory":
            return

    if args.stage in {"memory_v3"}:
        v3_dir = ROOT / "outputs" / f"{stage}_phase1_5b_v3_16k"
        run([sys.executable, "scripts/train_memory_v3.py", "--config", args.memory_v3_config, "--features", str(features), "--checkpoint", str(memory_dir / "memory_v2_last.pt" if (memory_dir / "memory_v2_last.pt").exists() else ckpt), "--max_steps", str(args.memory_v3_steps), "--output_dir", str(v3_dir), "--report", f"docs/LOCAL_V3_16K_MEMORY_REPORT_{stage}.md"], env=env)
        return

    phase2_stage = mtokens_to_stage(args.phase2_mtokens)
    p2_train, p2_val, p2_test, p2_phase1_dir, p2_ckpt, p2_features, p2_memory_dir = paths_for(phase2_stage)

    if args.stage == "phase2_data":
        run_data_stage(phase2_stage)
        return
    if args.stage == "phase2_audit_data":
        run_audit_stage(phase2_stage)
        return
    if args.stage == "phase2_phase1_resume":
        run([sys.executable, "scripts/train.py", "--config", args.phase1_config, "--data", str(p2_train), "--max_steps", str(args.phase2_steps), "--save_every", "0", "--resume", str(ckpt), "--output_dir", str(p2_phase1_dir)], env=env)
        return
    if args.stage == "phase2_features":
        source_ckpt = p2_ckpt if p2_ckpt.exists() else ckpt
        run([sys.executable, "scripts/extract_memory_features.py", "--config", args.phase1_config, "--checkpoint", str(source_ckpt), "--data", str(p2_train), "--seq_len", str(args.seq_len), "--max_samples", str(args.feature_samples), "--span_size", str(args.span_size), "--batch_size", "1", "--out", str(p2_features), "--report", f"docs/LOCAL_V3_16K_PHASE2_FEATURES_{phase2_stage}.md"], env=env)
        return
    if args.stage == "phase2_memory_v3":
        prev_memory = memory_dir / "memory_v2_last.pt"
        if not prev_memory.exists():
            prev_memory = p2_ckpt if p2_ckpt.exists() else ckpt
        run([sys.executable, "scripts/train_memory_v3.py", "--config", args.memory_v3_config, "--features", str(p2_features), "--checkpoint", str(prev_memory), "--max_steps", str(args.memory_v3_steps), "--output_dir", str(p2_memory_dir), "--report", f"docs/LOCAL_V3_16K_PHASE2_MEMORY_REPORT_{phase2_stage}.md"], env=env)
        return
    if args.stage == "phase2_joint":
        phase_ckpt = p2_ckpt if p2_ckpt.exists() else ckpt
        mem_ckpt = p2_memory_dir / "memory_v3_last.pt"
        run([sys.executable, "scripts/train_joint_memory_lm.py", "--config", args.joint_config, "--data", str(p2_train), "--phase1_checkpoint", str(phase_ckpt), "--memory_checkpoint", str(mem_ckpt), "--max_steps", str(args.phase2_joint_steps), "--output_dir", str(ROOT / "outputs" / f"{phase2_stage}_joint_memory_lm"), "--report", f"docs/LOCAL_V3_16K_PHASE2_JOINT_REPORT_{phase2_stage}.md"], env=env)
        return


if __name__ == "__main__":
    main()
