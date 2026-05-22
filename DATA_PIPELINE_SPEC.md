# LatentLoop Data Pipeline Spec v1

This project trains on **pre-tokenized fixed-length JSONL blocks**. The model training script must not read raw text directly.

## Required output schema

```json
{"input_ids":[2,123,456,3,0],"source":"fineweb_edu","domain":"general","valid_tokens":512,"doc_count":1}
```

Only `input_ids` is consumed by the current `JsonlTokenDataset`; metadata is for audit.

## Stages

| stage | train blocks | train tokens @512 | purpose |
|---|---:|---:|---|
| `real_smoke_1m` | 2,048 | 1,048,576 | verify real data path |
| `real_short_5m` | 10,240 | 5,242,880 | short real training |
| `warmup_probe_70m` | 143,360 | 73,400,320 | covers ~2,240 optimizer steps with grad_accum=64 |

Val/test are additional small audited splits.

## Mix

Train split is enforced by block count:

- 85% general
- 10% reasoning
- 5% divergent

Val/test use the same allocation method but exact <1% tolerance is not enforced by default because small splits have integer-ratio limits.

## Default downloadable sources

Configured in `data/sources/data_sources.yaml`:

- `HuggingFaceFW/fineweb-edu`
- `HuggingFaceTB/smollm-corpus`, subset `fineweb-edu-dedup`
- `HuggingFaceTB/smollm-corpus`, subset `cosmopedia-v2`
- `nvidia/OpenMathReasoning`
- `Nan-Do/code-search-net-python`

Run remote source probe before real data generation:

```powershell
python scripts\data\probe_sources.py --config data\sources\data_sources.yaml --out data\reports\source_probe_report.json
```

For CI or offline testing only:

```powershell
python scripts\data\probe_sources.py --config data\sources\data_sources.yaml --offline_mock --out data\reports\source_probe_report_mock.json
```

`--offline_mock` does not prove remote downloadability.

## Tokenizer

Real training requires a SentencePiece BPE tokenizer:

- vocab_size = 32000
- pad_id = 0
- unk_id = 1
- bos_id = 2
- eos_id = 3
- byte_fallback = true

The offline mock path may use a JSON hash tokenizer only when `--offline_mock` is set and SentencePiece is unavailable. That tokenizer is explicitly marked `mock_only` and must not be used for real training.

## Generate real smoke data

```powershell
python scripts\data\run_data_pipeline.py `
  --stage real_smoke_1m `
  --sources data\sources\data_sources.yaml `
  --seq_len 512 `
  --tokenizer_out data\tokenizer `
  --output_dir data\processed `
  --report_dir data\reports `
  --seed 42
```

Install data dependencies first:

```powershell
pip install -e .[data]
```

## Audit existing processed data

```powershell
python scripts\data\audit_dataset.py `
  --train data\processed\real_smoke_1m.train.jsonl `
  --val data\processed\real_smoke_1m.val.jsonl `
  --test data\processed\real_smoke_1m.test.jsonl `
  --seq_len 512 `
  --vocab_size 32000 `
  --out data\reports\audit_real_smoke_1m.json
```

## Blocking conditions

- source probe fails for a required source
- any token id is outside `[0, vocab_size)`
- block length is not exactly `seq_len`
- padding is not tail-only
- train mix ratio differs by more than 1%
- duplicate block ratio exceeds 1%
- train/val/test exact block overlap exists
- chunked CE equivalence check exceeds `1e-5`
- `audit_dataset.py` returns `BLOCKED`
