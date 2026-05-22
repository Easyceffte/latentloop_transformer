# V2 Implementation Audit Report

## Scope

Implemented Phase 1.5B V2 memory-objective tooling on top of the cleaned audited LatentLoop-Transformer-PDS+M tree.

This package is designed for local 4070 validation without a 4090. It uses smaller memory tables for V2 dry runs:

- 16K slots: fast local smoke / design validation
- 64K slots: larger 4070 validation

The original Phase 1 checkpoint is not required for engineering smoke tests. It is required only for the final research validation on real Phase 1 features.

## Implemented files

- `src/latentloop_pds_m/dual_stream.py`
  - exposes `z_memory_query`, the pre-memory-read latent.
- `src/latentloop_pds_m/modeling.py`
  - stores `z_memory_query` as an internal feature, not as an aggregate scalar metric.
- `scripts/make_synthetic_memory_features.py`
  - creates V2 feature caches without a checkpoint.
- `scripts/extract_memory_features.py`
  - extracts span-level `z_query_l5` and `z_fused_l5` from a checkpoint or random-init model.
- `scripts/train_memory_v2.py`
  - trains reconstruction + contrastive + weak graph objectives on cached features.
- `configs/phase1_5b_v2_memory_4070_16k.yaml`
- `configs/phase1_5b_v2_memory_4070_64k.yaml`
- `docs/PHASE1_5B_V2_MEMORY_OBJECTIVE_SPEC.md`
- `docs/CODEX_PHASE1_5B_V2_GOAL.md`
- `tests/test_phase1_5b_v2_memory_objective.py`

## Design guardrails

- No Phase 2.
- No online update.
- No large memory table requirement for local validation.
- Exact retrieval is used for 16K/64K V2 objective training to keep query/key gradients clean.
- Feature caches and checkpoints are generated locally and should not be committed.
- `memory.retrieve()` already applies IVE; V2 reconstruction does not apply IVE twice.

## Test commands run

```bash
python -m compileall -q src scripts tests
PYTHONPATH=. pytest -q tests/test_phase1_5b_v2_memory_objective.py
PYTHONPATH=. pytest -q
```

## Results

```text
phase1_5b_v2 tests: 4 passed
full test suite: 27 passed
```

## Known limitations

The cloud Phase 1 checkpoint and real mixed training data are not present in this environment, so real Phase 1 feature extraction and 50-step real reconstruction dry run were not executed here.

The synthetic smoke path validates implementation and gradients, not final memory repair quality.

## Next required validation

On the user's machine/cloud:

1. Generate synthetic features and run a 10-step smoke.
2. Extract real Phase 1 features if `ckpt_020000.pt` is available.
3. Run 50-step V2 reconstruction dry run.
4. Review `PHASE1_5B_V2_DRYRUN_REPORT.md` before any longer training.
