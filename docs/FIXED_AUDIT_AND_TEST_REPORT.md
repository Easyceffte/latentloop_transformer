# Fixed Audit and Test Report

This package applies the second-pass audit fixes on top of the uploaded `latentloop_transformer-main.zip`.

## Fix summary

### P0 fixed

1. `RecurrentLoopTower.shortcut_consistency()` now computes KL on flattened `[B*L, D]` token distributions, preventing `seq_len` amplification. It also reports normalized shortcut metrics:
   - `loop_shortcut_kl_normalized`
   - `loop_shortcut_2v8_cosine`
   - `loop_shortcut_4v8_cosine`
   - `loop_shortcut_2v8_mse`
   - `loop_shortcut_4v8_mse`
   - top-level `total_loss_without_shortcut`
2. `train.py` now performs CPU-side vocab compatibility preflight for real processed JSONL before CUDA model initialization.

### P1/P2 fixed or locked with tests

3. OpenMathReasoning supports `formatter: problem_solution`, preserving both problem and generated solution.
4. `fallbacks` in `data_sources.yaml` was renamed to `planned_fallbacks` to avoid dead active-config semantics.
5. Resume no longer overwrites `metrics.jsonl`; it appends and writes a JSONL resume marker.
6. Periodic mid-accumulation snapshots are marked non-resumable; resume rejects them.
7. Dense feedback metrics now split actual `dense_feedback_gate_mean` from `dense_feedback_signal_mean`.
8. LDMG `write_mask` now gates memory stability delta and optional online update.
9. `apply_forgetting()` now has observable probability-space forgetting semantics.
10. `sample_raw_sources.py` was updated to the current `collect_source_docs()` API.
11. Offline mock data now uses a stable SHA1-derived hash instead of process-randomized Python `hash()`.
12. Cache-hit behavior has a direct test that proves `load_hf_iter()` is not called on raw-cache hit.

## Local validation

Commands run from repository root:

```bash
python -m compileall -q src scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/smoke_forward.py --config configs/smoke_tiny.yaml --seq 16 --backward
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/train.py --config configs/smoke_tiny.yaml --synthetic --max_steps 2 --output_dir /mnt/data/ll_fix_train_smoke
PYTHONPATH=. python scripts/data/run_data_pipeline.py --stage real_smoke_1m --sources data/sources/data_sources.yaml --seq_len 512 --tokenizer_out /mnt/data/ll_data_tok --output_dir /mnt/data/ll_data_processed --report_dir /mnt/data/ll_data_reports --raw_cache_dir /mnt/data/ll_data_raw --intermediate_dir /mnt/data/ll_data_inter --offline_mock --tokenizer_vocab_size 512 --max_train_blocks_override 20
PYTHONPATH=. python scripts/data/audit_dataset.py --train /mnt/data/ll_data_processed/real_smoke_1m.train.jsonl --val /mnt/data/ll_data_processed/real_smoke_1m.val.jsonl --test /mnt/data/ll_data_processed/real_smoke_1m.test.jsonl --seq_len 512 --vocab_size 512 --out /mnt/data/ll_data_audit.json
PYTHONPATH=. python scripts/data/sample_raw_sources.py --source fineweb_edu --num_docs 2 --output /mnt/data/ll_sample_raw.jsonl --offline_mock --raw_cache_dir /mnt/data/ll_raw_cache_sample
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/audit/run_residual_audit.py --config configs/smoke_tiny.yaml --device cpu --seq 8 --out /mnt/data/ll_residual_after_fix --tiny_steps 2
```

Observed results:

- `pytest`: `23 passed`
- `smoke_forward`: finite loss, backward succeeded, `loop_shortcut_consistency_loss≈0.315` rather than sequence-length-scaled hundreds/thousands
- `synthetic train --max_steps 2`: completed and saved checkpoint
- offline data pipeline mini-run: completed and audit passed
- `sample_raw_sources.py`: completed with cache report
- residual audit: `PASS_TO_REAL_DATA_SHORT_RUN`

## Remaining non-blocking limitations

- Exact DataLoader sampler position is still not restored on resume; optimizer/scheduler/RNG are restored and this limitation is recorded in the resume marker.
- Hebbian update remains not implemented as a real co-activation graph update.
- For `warmup_probe_70m`, the current JSONL dataset loader still loads records into memory. A streaming/indexed dataset should be added before very large local runs.
- Full remote Hugging Face probe/download was not executed in this environment.
