# Codex Goal: Local V3-16K Full Progression

Use only the project venv. Do not use global Python.

```powershell
cd C:\Users\14912\Documents\MyLLM
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Run V3 memory on the completed 1M features

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_16k.py --stage memory_v3 --mtokens 1 --memory_v3_steps 500
```

Pass criteria:
- `recon_loss` does not regress badly from V2.
- `contrastive_loss` is finite and decreases.
- `retrieval_entropy_div_logtopk` decreases from ~1.0.
- `retrieval_top10_mass` moves above 10/top_k and ideally above 0.18.
- `graph.edge_logits_grad_norm` is nonzero.
- No NaN/OOM.

## Try Phase2

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_16k.py --stage phase2_data --phase2_mtokens 3 --reuse_raw_cache
.\.venv\Scripts\python.exe scripts\run_local_v3_16k.py --stage phase2_audit_data --phase2_mtokens 3
.\.venv\Scripts\python.exe scripts\run_local_v3_16k.py --stage phase2_phase1_resume --phase2_mtokens 3 --phase2_steps 512
.\.venv\Scripts\python.exe scripts\run_local_v3_16k.py --stage phase2_features --phase2_mtokens 3 --feature_samples 1000
.\.venv\Scripts\python.exe scripts\run_local_v3_16k.py --stage phase2_memory_v3 --phase2_mtokens 3 --memory_v3_steps 500
.\.venv\Scripts\python.exe scripts\run_local_v3_16k.py --stage phase2_joint --phase2_mtokens 3 --phase2_joint_steps 256
```

## Report

Generate `docs/LOCAL_V3_16K_PROGRESS_REPORT.md` with: V3 memory metrics, Phase2 stage status, checkpoint paths, blockers, and final decision. Do not commit data, raw cache, outputs, checkpoints, metrics, or feature caches.
