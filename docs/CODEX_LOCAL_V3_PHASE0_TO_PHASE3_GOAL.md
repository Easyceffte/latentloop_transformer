# Goal for Codex: Local V3-16K Phase0→Phase3

Use project venv only:

```powershell
cd C:\Users\14912\Documents\MyLLM
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

Do not use global Python. Fixed constraints:

- memory.n_slots=16384
- online_update=false
- use_lsh=false
- retrieval_backend=exact_chunked
- no mock success
- no data/checkpoint/outputs commit

## Mission

Run the complete local 4070 V3 pipeline from Phase0 to Phase3 with native Chinese dialogue data.

## Commands

### Phase0 real data

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage phase0_data --mtokens 1 --reuse_raw_cache
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage audit_data --mtokens 1
```

If HF source fails, fix retry/cache/source fallback explicitly. Do not silently use mock. If one dataset name is unavailable, replace it with a real Chinese/dialogue source and document the replacement.

### Performance profile

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage profile --mtokens 1
```

If GPU util is low, use the profile report to identify whether bottleneck is dataloader, H2D, forward, backward, optimizer, or memory retrieval.

### Phase1 base

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage phase1 --mtokens 1 --phase1_steps 2000
```

If OOM, rerun with safe config manually by passing `--config configs/local_v3_16k_phase1_4070_safe.yaml`, then report the change.

### Generation smoke

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage generate --mtokens 1
```

Prompts include `你好`, `请用一句话介绍你自己。`, `1+1=`, `苹果是一种`.

### Memory V3

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage features --mtokens 1 --feature_samples 1000
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage memory_v3 --mtokens 1 --memory_v3_steps 500
```

### Phase2 joint

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage phase2_joint --mtokens 1 --joint_steps 512
```

### Phase3 dialogue SFT

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage phase3_sft --mtokens 1 --phase3_steps 1000
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage phase3_eval --mtokens 1
```

## Scale-up

If 1M passes, repeat with 5M. If 5M passes and storage/time allow, prepare 20M. 100M is supported by the pipeline but should not be started unless data/cache/VRAM/time are confirmed.

## Final report

Generate `docs/LOCAL_V3_PHASE0_TO_PHASE3_REPORT.md` with:

- environment
- pytest result
- data source status and Chinese/dialogue mix
- data audit
- profiler summary
- Phase1 checkpoint and metrics
- generation smoke decoded outputs
- Memory V3 metrics
- Phase2 joint result
- Phase3 dialogue result
- blockers and tracebacks

Final decision must be one of:

- PHASE0_DATA_PASS
- PHASE1_PASS
- GENERATION_PASS
- MEMORY_V3_PASS
- PHASE2_PASS
- PHASE3_DIALOGUE_PASS
- PARTIAL_PASS
- FAILED
