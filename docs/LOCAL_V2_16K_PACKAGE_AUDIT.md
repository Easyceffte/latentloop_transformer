# Local V2-16K Package Audit

## Scope

This zip is a clean local 4070 route for rebuilding the project from zero after the cloud Phase 1 loss.

Included path:

1. Synthetic memory smoke, no checkpoint required.
2. Scalable data pipeline for `local_v2_1m`, `local_v2_3m`, `local_v2_5m`.
3. Local Phase 1 from-scratch 16K memory checkpoint.
4. Phase 1 feature extraction.
5. Phase 1.5B V2 memory-objective dry run.

Excluded from the zip:

- raw downloaded data
- processed JSONL blocks
- tokenizer artifacts
- feature caches
- checkpoints
- training outputs
- raw metrics

## Fixed local architecture choice

- `memory.n_slots = 16_384`
- `memory.top_k = 64`
- `memory.use_lsh = false`
- `memory.online_update = false`
- default Phase 1 `seq_len = 256`

## Added files

- `configs/local_v2_16k_phase1_4070.yaml`
- `configs/local_v2_16k_phase1_4070_safe.yaml`
- `configs/phase1_5b_v2_memory_4070_16k.yaml`
- `data/sources/local_v2_sources.yaml`
- `scripts/run_local_v2_16k_from_zero.py`
- `scripts/make_synthetic_memory_features.py`
- `scripts/extract_memory_features.py`
- `scripts/train_memory_v2.py`
- `docs/LOCAL_V2_16K_FROM_ZERO_SPEC.md`
- `docs/CODEX_LOCAL_V2_16K_FROM_ZERO_GOAL.md`
- `requirements-local-v2.txt`
- tests for the local V2 path

## Tests run in packaging environment

```bash
python -m compileall -q src scripts tests
PYTHONPATH=. pytest -q
```

Result:

```text
30 passed in 21.62s
```

Synthetic memory smoke was also run before cleaning generated artifacts:

```bash
PYTHONPATH=. python scripts/run_local_v2_16k_from_zero.py --stage synthetic_memory_smoke
```

Result summary:

- feature generation: PASS
- memory V2 10-step synthetic smoke: PASS
- recon loss drop: ~27.98%
- memory.values grad norm: nonzero
- query/key grad norm: nonzero
- checkpoint save: verified during smoke, then removed from package

## Known limitations

- The real 1M data run was not executed in the packaging environment because it requires internet access and local disk/time.
- The package does not contain actual data; it contains a reusable, cached data pipeline.
- The local Phase 1 run should start with 512 micro steps. Only after that passes should it increase to 2000/5000.
- The `safe` Phase 1 config disables Dense Feedback and should be used only if the default config OOMs.

## Stop conditions

Stop if:

- data audit is not PASS
- Phase 1 has NaN/Inf/OOM
- feature extraction reports zero norm or NaN/Inf
- memory V2 reconstruction loss does not decrease
- memory query/key grad norms remain zero

Do not enable online update or Phase 2 until the local 16K memory route passes.
