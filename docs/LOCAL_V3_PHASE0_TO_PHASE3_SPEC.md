# Local V3-16K Phase0→Phase3 Spec

## Current decision

This package is the 4070-local continuation of LatentLoop-Transformer-PDS+M. It keeps the V3 memory design but changes the training objective from a pure mechanism smoke into a native-Chinese-capable end-to-end local pipeline.

The target is not a minimal patch. The target is a full local engineering path:

1. Phase0 data: real, cached, auditable, Chinese-capable 1M/5M/20M/100M token budgets.
2. Phase1 base: train from scratch on Local V3-16K.
3. Phase1.5 Memory V3: reconstruction + contrastive retrieval + graph co-activation + distiller.
4. Phase2 joint: reconnect trained memory to LM.
5. Phase3 dialogue SFT: train on dialogue/instruction blocks so `你好` can decode as natural text.
6. Generation: decoded text smoke, not only token ids.
7. Performance: profiler + retrieval backend + dataloader/optimizer/attention checks.

## Data mix

The default V3 mix is:

```text
general:   60%
dialogue:  20%
reasoning: 15%
divergent:  5%
```

The data sources are in:

```text
data/sources/local_v3_zh_dialogue_sources.yaml
```

Chinese capability is not an afterthought. The config includes Chinese web/general data, Chinese instruction/dialogue data, Chinese math/reasoning data, plus English high-quality general/reasoning data for breadth.

Supported budgets at seq_len=256:

```text
local_v3_1m    = 4096 train blocks
local_v3_5m    = 20480 train blocks
local_v3_20m   = 81920 train blocks
local_v3_100m  = 409600 train blocks
```

## Model and memory constraints

For 4070 local training:

```text
memory.n_slots = 16384
top_k = 64
retrieval_backend = exact_chunked
online_update = false
use_lsh = false
```

The 16K memory table is deliberate. The V1 215K/1M memory route made memory too sparse and difficult to train locally. V3 proves retrieval, graph, and memory-conditioned LM before scaling capacity.

## Performance policy

Do not assume the bottleneck. Run:

```powershell
.\.venv\Scripts\python.exe scripts\profile_local_v3.py --config configs/local_v3_16k_phase1_4070_safe.yaml --data data/processed/local_v3_1m.train.jsonl --steps 20 --warmup 3
```

Report:

- dataloader time
- host-to-device time
- forward time
- backward time
- optimizer time
- tokens/sec
- max allocated VRAM

The likely V1 failure mode was high VRAM plus low GPU utilization from small micro-batch, double/dual paths, checkpoint recomputation, small kernels, and CPU/data stalls. Kernel work must be driven by profiler evidence.

## Generation policy

Generation must produce decoded text, not only token ids.

Use:

```powershell
.\.venv\Scripts\python.exe scripts\eval_generation_smoke.py --config configs/local_v3_16k_phase1_4070.yaml --checkpoint outputs/local_v3_1m_phase1_v3_16k/last.pt --tokenizer data/tokenizer/local_v3_32k/tokenizer.model
```

The first acceptance is not intelligence. The first acceptance is:

- no empty output
- no decode crash
- no all-pad/all-unk output
- no NaN logits
- Chinese prompt produces decoded Chinese-capable text

## Phase gates

### Phase0 data

Run real data. Mock is forbidden for success claims.

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage phase0_data --mtokens 1 --reuse_raw_cache
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage audit_data --mtokens 1
```

### Phase1 base

```powershell
.\.venv\Scripts\python.exe scripts\run_local_v3_phase0_to_phase3.py --stage phase1 --mtokens 1 --phase1_steps 2000
```

### Phase1.5 memory

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

## Real success metrics

Do not overclaim. On 4070, success is defined by mechanism and generation gates:

- data audit PASS
- Phase1 loss/lm_loss finite and decreasing
- generation smoke decodes text
- memory V3 contrastive loss decreases
- retrieval entropy starts moving below uniform
- graph edge logits receive gradient
- joint LM does not explode
- phase3 generation improves short Chinese prompts

## Forbidden shortcuts

- Do not call mock data a success.
- Do not return to 215K memory before 16K passes retrieval/joint gates.
- Do not enable online_update before distiller/retrieval/joint are verified.
- Do not commit data/raw_cache, data/processed, outputs, checkpoints, metrics, feature caches.
