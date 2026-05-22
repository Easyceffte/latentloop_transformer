# Goal for Codex · Local V2-16K From Zero

You are operating on a local Windows/4070 setup. The old 4090 checkpoint and data are unavailable. Do not attempt to recover them. Build and run the local V2-16K route.

## Hard constraints

- Do not enable `online_update`.
- Do not start Phase 2.
- Do not use 215K memory.
- Keep `memory.n_slots=16384`.
- Keep `memory.use_lsh=false` for local V2 debugging.
- Do not commit data, features, raw cache, tokenizer artifacts, metrics, checkpoints, or outputs.
- Stop after each stage and report. Do not automatically escalate from 1M to 3M/5M.

## Initial commands

```powershell
python -m pip install -e .[data,dev]
python -m compileall -q src scripts tests
pytest -q
```

## Stage 0: synthetic memory smoke

```powershell
python scripts/run_local_v2_16k_from_zero.py --stage synthetic_memory_smoke
```

Report:
- whether it passed
- final recon_loss
- memory.values/query/key grad norms
- output path

## Stage 1: real 1M data

If internet is available:

```powershell
python scripts/run_local_v2_16k_from_zero.py --stage data --mtokens 1 --reuse_raw_cache
python scripts/run_local_v2_16k_from_zero.py --stage audit_data --mtokens 1
```

If internet is not available, run `--offline_mock` only to verify the pipeline, and clearly mark it as not real training data.

Report:
- pipeline report path
- audit decision
- train/val/test paths
- train blocks and tokens
- remote access/cache hit status

## Stage 2: Phase 1-local

Start with 512 micro steps:

```powershell
python scripts/run_local_v2_16k_from_zero.py --stage phase1 --mtokens 1 --phase1_steps 512
```

If OOM, rerun with:

```powershell
python scripts/run_local_v2_16k_from_zero.py --stage phase1 --mtokens 1 --phase1_steps 512 --phase1_config configs/local_v2_16k_phase1_4070_safe.yaml
```

Report:
- output dir
- checkpoint path
- final loss/lm_loss/ddpm/shortcut/gates
- whether checkpoint exists
- whether metrics are finite

## Stage 3: feature extraction

```powershell
python scripts/run_local_v2_16k_from_zero.py --stage features --mtokens 1 --feature_samples 1000
```

Report:
- feature path
- shape of z_query_l5 / z_fused_l5
- norm mean/std
- NaN/Inf count

## Stage 4: memory V2 dry run

```powershell
python scripts/run_local_v2_16k_from_zero.py --stage memory --mtokens 1 --memory_steps 50
```

Report:
- report path
- recon_loss first/last/drop %
- memory.values/query/key grad norm
- retrieval top1/top5/top10
- entropy_div_logtopk
- graph edge ratios
- PASS/REVIEW decision

## Final handoff

After these stages, commit only lightweight code/docs/config changes. Do not commit generated artifacts.

Provide:
- git status
- changed tracked files
- path to local reports
- exact command history
- final decision: PASS_TO_1M_LONGER_RUN, BLOCKED_FIX_CODE, or REVIEW_MEMORY_OBJECTIVE
