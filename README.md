# LatentLoop-Transformer-PDS+M

From-scratch implementation of **LatentLoop-Transformer-PDS+M**: an 18-layer decoder-only language model with parallel latent diffusion/recurrent reasoning blocks and an implicit latent dense memory graph.

This repository intentionally implements the from-scratch route only. It does not include a Qwen/千问 frozen wrapper.

## Implemented architecture

- 18-layer causal Transformer backbone.
- Standard layers: GQA causal attention + RoPE + SwiGLU FFN.
- Dual-stream blocks at layers 5 and 11.
- Diffusion tower: latent projection, Transformer denoiser, DDPM q-sampling, v-prediction loss, DDIM reverse sampling, multi-branch diversity-aware merge.
- Recurrent loop tower: shared loop block, step embedding + total-step embedding, exit gate, shortcut-consistency loss.
- Bidirectional interaction: diffusion-to-loop and loop-to-diff cross-attention, gated residual update, IDC entropy diagnostics, stability fusion.
- Flow controller: Softmax gates for inference/write/read with independent alpha scaling.
- LDMG: learnable latent memory matrix, graph overview with directed transition matrix, sparse exact/LSH retrieval, IVE value transform, surprise/write/stability losses, retrieval diagnostics.
- Dense Feedback layer: 10 routed feedback paths from layer 14, warmup gate, differentiable current-forward recomputation by default.
- Full 11-term loss structure, WSD scheduler, chunked CE, JSONL dataset loader, smoke/train/generate/audit scripts.

## Quick smoke test

```bash
cd latentloop_pds_m
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/smoke_forward.py --config configs/smoke_tiny.yaml --seq 16 --backward
```

## Tiny synthetic training

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train.py \
  --config configs/smoke_tiny.yaml \
  --synthetic \
  --max_steps 20 \
  --output_dir outputs/smoke
```

## Configurations

Two full-size configurations are included because the written spec contains an internal parameter-count tension: an explicit `1_000_000 × 256` memory matrix alone is 256M parameters, so it cannot coexist with a ~265M total-parameter model unless the memory is compressed or factorized.

- `configs/from_scratch_full_1m.yaml`: literal 1M-slot LDMG, ~467.0M total parameters with explicit dense memory.
- `configs/from_scratch_265m_approx.yaml`: same architecture but ~215k explicit memory slots, ~266.1M total parameters.
- `configs/from_scratch_230m.yaml`: compatibility alias for the ~266M explicit-memory configuration.

For a 4070-class GPU, start with `configs/from_scratch_265m_approx.yaml` or reduce `memory.n_slots`, use `seq_len=512`, `micro_batch_size=1`, `grad_accum=64`, and keep chunked CE enabled.

## Current limitations

- Full training was not run in this environment. The included validation is CPU tiny-config forward/backward and 2-step synthetic training.
- LSH retrieval no longer builds `[B*L,N]` masks, but production-scale 1M+ slots should still prefer a FAISS GPU/custom CUDA index for speed.
- Dense Feedback defaults to differentiable current-forward recomputation, which is correct but roughly doubles parts of the forward cost. Cached next-forward feedback remains available only as an ablation.
- Diffusion Forcing sampler is represented by standard autoregressive `generate()` plus recurrent latent refinement; a true parallel token sampler should be added after baseline training is stable.

## Local V2-16K from-zero route

If the old cloud Phase 1 checkpoint/data are unavailable, use the local 4070 route:

```bash
python -m pip install -e .[data,dev]
python -m compileall -q src scripts tests
pytest -q

# No checkpoint required: verify V2 memory objective implementation.
python scripts/run_local_v2_16k_from_zero.py --stage synthetic_memory_smoke

# Real 1M-token data pipeline, cached under data/raw_cache.
python scripts/run_local_v2_16k_from_zero.py --stage data --mtokens 1 --reuse_raw_cache
python scripts/run_local_v2_16k_from_zero.py --stage audit_data --mtokens 1

# Short Phase 1-local run, then feature extraction and memory objective dry run.
python scripts/run_local_v2_16k_from_zero.py --stage phase1 --mtokens 1 --phase1_steps 512
python scripts/run_local_v2_16k_from_zero.py --stage features --mtokens 1 --feature_samples 1000
python scripts/run_local_v2_16k_from_zero.py --stage memory --mtokens 1 --memory_steps 50
```

See `docs/LOCAL_V2_16K_FROM_ZERO_SPEC.md` and `docs/CODEX_LOCAL_V2_16K_FROM_ZERO_GOAL.md`.
