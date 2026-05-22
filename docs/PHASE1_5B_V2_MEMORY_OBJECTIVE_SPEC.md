# Phase 1.5B V2 Memory Objective Spec

## Current decision

Phase 1 produced a usable `phase1_base_no_online_memory` checkpoint, but it did not train LDMG as a functional long-term memory system. The observed failure mode is not language-model collapse; it is memory-signal starvation: memory read has near-zero effect on logits, `memory.values` gradients stay around noise level, query/key gradients are smaller again, retrieval remains close to uniform, and graph edges do not specialize.

V2 therefore changes the memory training path before attempting more LM training. It does not start Phase 2 and it does not enable online update. It trains the memory subsystem with local objectives that do not require gradients to traverse the full 18-layer Transformer stack.

The design follows three conservative lessons from memory-augmented Transformer work:

1. Long-term memory should have a direct local target, not only a final-token LM loss.
2. Retrieval needs a contrastive signal; top-k similarity without a semantic target can remain uniform.
3. Graph memory needs content-linked positive/negative edges; entropy pressure alone is not meaningful.

## Scope implemented in this package

Implemented now:

- expose `z_memory_query` before memory read residual is added;
- extract span-level `z_query_l5` and `z_fused_l5` feature caches;
- generate synthetic feature caches for local testing without a Phase 1 checkpoint;
- train V2 memory objectives on cached features;
- support smaller 4070-friendly memory tables: 16K and 64K slots;
- write dry-run reports and metrics;
- test schema, retrieval baselines, cropped checkpoint loading, and one-step gradient flow.

Not enabled now:

- Phase 2;
- `online_update=true`;
- full Layer 5/11 unfreezing;
- automatic 500-step extension;
- memory-distiller write path.

## V2 memory objective

The training step consumes cached span-level features:

```text
z_query_l5: [B, S, 256]
z_fused_l5: [B, S, 256]
```

The model trains only LDMG-related parameters:

```text
memory.values
memory.query_proj
memory.key_proj
memory.ive
memory.reconstruct
memory.graph.* when graph_weight > 0
```

### Loss

```text
loss = 1.0 * reconstruction_loss
     + contrastive_weight * contrastive_retrieval_loss
     + graph_weight * graph_structure_loss
     + norm_weight * memory_norm_regularizer
```

Default V2 dry-run weights:

```text
contrastive_weight = 0.1
graph_weight = 0.01
norm_weight = 0.001
contrastive_tau = 0.1
```

### Reconstruction

```python
retrieved = memory.retrieve(z_query_l5)["retrieved"]
z_pred = memory.reconstruct(retrieved)
loss_recon = mse(z_pred, stopgrad(z_fused_l5))
```

Important implementation note: current `memory.retrieve()` already applies `memory.ive()` internally, so V2 must not apply IVE a second time.

### Contrastive retrieval

V2 uses an in-batch span objective:

```python
q = normalize(memory.query_proj(z_query))
k = normalize(memory.key_proj(stopgrad(z_fused)))
loss_nce = CE(q @ k.T / tau, arange(num_spans_in_batch))
```

This is intentionally simple. It verifies that query/key can stop behaving like an almost-uniform retriever before adding harder document-level positives/negatives.

### Graph structure

V2 applies a weak adjacent-span graph objective:

```text
span_i -> span_{i+1} should be more probable than random shifted transitions.
```

This gives `edge_logits` a supervised gradient. It does not rely on entropy minimization alone.

## 4070 memory sizes

Two configs are included:

```text
configs/phase1_5b_v2_memory_4070_16k.yaml
configs/phase1_5b_v2_memory_4070_64k.yaml
```

Use 16K for local smoke and fast iteration. Use 64K if 16K passes and memory is still stable. Both use exact retrieval (`use_lsh=false`) to keep query/key gradients clean during local objective training.

## Local no-checkpoint smoke

Generate synthetic features:

```bash
python scripts/make_synthetic_memory_features.py \
  --num_samples 128 --num_spans 16 --d_mem 256 --seq_len 256 --span_size 16 \
  --out data/processed/memory_features_synthetic.pt \
  --report docs/PHASE1_5B_SYNTHETIC_FEATURE_REPORT.json
```

Run V2 memory objective:

```bash
python scripts/train_memory_v2.py \
  --config configs/phase1_5b_v2_memory_4070_16k.yaml \
  --features data/processed/memory_features_synthetic.pt \
  --max_steps 50 \
  --output_dir outputs/phase1_5b_v2_synthetic_dryrun \
  --report docs/PHASE1_5B_V2_SYNTHETIC_DRYRUN_REPORT.md
```

This validates code, gradients, metrics, checkpoint saving, and the smaller-memory path. It does not validate whether Phase 1 memory is repaired.

## Real Phase 1 validation

After local smoke passes, extract real features from the Phase 1 checkpoint:

```bash
python scripts/extract_memory_features.py \
  --config configs/rtx4090_mixed.yaml \
  --checkpoint /root/ll_project/outputs/phase1_final/ckpt_020000.pt \
  --data /root/ll_project/data/processed/mixed_train.jsonl \
  --seq_len 256 --max_samples 2000 --span_size 16 --batch_size 1 \
  --out data/processed/memory_features_phase1_5b.pt \
  --report docs/PHASE1_5B_FEATURE_EXTRACTION_REPORT.md
```

Then run the 16K or 64K V2 memory objective. If a full 215K checkpoint is supplied but the V2 config uses fewer slots, the loader crops `memory.values` and `retrieval_count` rows safely.

## Dry-run pass criteria

For a 50-step dry run:

```text
reconstruction loss should fall by at least 5%, or report REVIEW;
memory.values grad_norm should exceed 1e-4 on real Phase 1 features;
query_proj/key_proj grad_norm should exceed 1e-5 on real Phase 1 features;
retrieval_entropy_div_logtopk should start below or move below 1.0;
graph.edge_logits grad_norm should become non-zero when graph_weight > 0;
no NaN/Inf/OOM;
checkpoint saved.
```

For synthetic features, the numerical thresholds are advisory rather than decisive. Synthetic smoke proves implementation, not research success.

## Do not proceed automatically

Do not run 500 steps or Phase 2 until the 50-step report is reviewed. If reconstruction fails on synthetic features, fix implementation. If reconstruction passes synthetic but fails Phase 1 features, adjust target choice (`z_query_l5` vs `z_fused_l5`) or memory structure.
