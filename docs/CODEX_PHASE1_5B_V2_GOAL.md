# Codex Goal: Phase 1.5B V2 Memory Objective

/goaI

Apply the Phase 1.5B V2 memory objective implementation and validate it locally.

## Hard constraints

- Do not start Phase 2.
- Do not enable `online_update=true`.
- Do not run any 500-step training until the 50-step dry-run report is reviewed.
- Do not merge `DP_work` into `main`.
- Do not commit checkpoints, feature caches, metrics JSONL, raw data, processed data, or outputs.

## Implementation checklist

1. Ensure `DualStreamBlock` returns `z_memory_query`, the pre-memory-read latent passed to `memory.retrieve()`.
2. Ensure `LatentLoopTransformerPDSM._run_layers()` stores `z_memory_query` as an internal feature, not as a scalar logged metric.
3. Add:
   - `scripts/make_synthetic_memory_features.py`
   - `scripts/extract_memory_features.py`
   - `scripts/train_memory_v2.py`
   - `configs/phase1_5b_v2_memory_4070_16k.yaml`
   - `configs/phase1_5b_v2_memory_4070_64k.yaml`
   - `docs/PHASE1_5B_V2_MEMORY_OBJECTIVE_SPEC.md`
   - `tests/test_phase1_5b_v2_memory_objective.py`

## Required local validation

```bash
python -m compileall -q src scripts tests
PYTHONPATH=. pytest -q tests/test_phase1_5b_v2_memory_objective.py
```

Then run a no-checkpoint smoke:

```bash
python scripts/make_synthetic_memory_features.py \
  --num_samples 64 --num_spans 8 --d_mem 256 --seq_len 128 --span_size 16 \
  --out data/processed/memory_features_synthetic_smoke.pt \
  --report docs/PHASE1_5B_SYNTHETIC_FEATURE_REPORT.json

python scripts/train_memory_v2.py \
  --config configs/phase1_5b_v2_memory_4070_16k.yaml \
  --features data/processed/memory_features_synthetic_smoke.pt \
  --max_steps 10 \
  --output_dir outputs/phase1_5b_v2_synthetic_smoke \
  --report docs/PHASE1_5B_V2_SYNTHETIC_SMOKE_REPORT.md
```

## Report back

Report only:

- commit hash;
- changed files;
- compileall result;
- pytest result;
- synthetic smoke result;
- whether any forbidden file was staged.
