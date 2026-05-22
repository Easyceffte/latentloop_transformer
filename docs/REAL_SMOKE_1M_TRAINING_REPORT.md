# Real Smoke 1M Training Report

This report records the local `real_smoke_1m` training probe. It intentionally
does not include datasets, raw cache, processed JSONL, checkpoints, or outputs.

## Scope

- Stage: `real_smoke_1m`
- Config: `configs/4070_512_preflight.yaml`
- Train data: `data/processed/real_smoke_1m.train.jsonl` (local only, ignored by git)
- Output dir: `outputs/real_smoke_1m_train_20260520_170702` (local only, ignored by git)
- Max steps: `512` micro-steps
- Gradient accumulation: `64`

## Environment

- Python: `C:\Users\14912\Desktop\LLM\.venv\Scripts\python.exe`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- GPU memory reported by PyTorch: `7.9956 GiB`

Note: the Codex process that generated this report could not resolve
`nvidia-smi` from PATH after the system PATH change, and the common absolute
paths were not present in that process view. PyTorch CUDA discovery succeeded.

## Data Audit

Audit command:

```powershell
& C:\Users\14912\Desktop\LLM\.venv\Scripts\python.exe scripts\data\audit_dataset.py `
  --train data\processed\real_smoke_1m.train.jsonl `
  --val data\processed\real_smoke_1m.val.jsonl `
  --test data\processed\real_smoke_1m.test.jsonl `
  --seq_len 512 `
  --vocab_size 32000 `
  --out data\reports\audit_real_smoke_1m_train_entry.json
```

Result:

- Decision: `PASS`
- Train blocks: `2048`
- Train tokens: `1,048,576`
- Val blocks: `32`
- Test blocks: `32`
- Invalid token count: `0`
- Duplicate block ratio: `0.0`
- Train mix: general `0.85009765625`, reasoning `0.10009765625`, divergent `0.0498046875`

## Training Command

```powershell
& C:\Users\14912\Desktop\LLM\.venv\Scripts\python.exe scripts\train.py `
  --config configs\4070_512_preflight.yaml `
  --data data\processed\real_smoke_1m.train.jsonl `
  --max_steps 512 `
  --output_dir outputs\real_smoke_1m_train_20260520_170702
```

## Training Result

- Completed naturally: yes
- Metrics rows: `512`
- Micro-step range: `0` to `511`
- Optimizer step range: `0` to `8`
- Non-finite metric count: `0`
- Checkpoint written locally: `outputs/real_smoke_1m_train_20260520_170702/last.pt`

Metric summary:

| Metric | First | Last | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `loss` | `18.328821` | `21.704750` | `15.373897` | `23.523129` |
| `lm_loss` | `10.467487` | `10.497889` | `10.258551` | `10.640597` |
| `ddpm_loss_raw` | `1.291331` | `1.326826` | `1.183031` | `1.410239` |
| `exit_entropy_raw` | `1.384766` | `1.326172` | `1.165039` | `1.388672` |
| `shortcut_consistency_loss_raw` | `1437.970947` | `2103.038086` | `848.091675` | `2480.888672` |
| `gate_inf_mean` | `0.333979` | `0.334045` | `0.331568` | `0.335615` |
| `gate_write_mean` | `0.332676` | `0.333767` | `0.331295` | `0.335087` |
| `gate_read_mean` | `0.333532` | `0.332088` | `0.331013` | `0.334862` |

## Interpretation

- The full 512-token preflight config ran against real processed data.
- Loss values remained finite for all 512 micro-steps.
- `optimizer_step` reached `8`, matching `512 / grad_accum 64`.
- `ddpm_loss_raw` stayed bounded in the observed run.
- `exit_entropy_raw` did not collapse to zero.
- Inference/write/read gate means stayed close to one third and did not collapse.

## Artifact Policy

The following are local-only and intentionally ignored by git:

- `data/raw_cache/`
- `data/intermediate/`
- `data/processed/`
- `data/reports/`
- `data/tokenizer/`
- `outputs/`
- checkpoint files such as `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`

