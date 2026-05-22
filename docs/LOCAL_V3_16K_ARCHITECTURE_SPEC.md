# Local V3-16K LatentLoop Memory Architecture Spec

## Current diagnosis

The local 1M V2-16K run proved the data/Phase1/feature/memory pipeline is executable on the 4070. The decisive signal is that the memory reconstruction objective trains strongly while retrieval entropy remains near `log(top_k)`. Therefore the next bottleneck is no longer raw memory gradient flow; it is selective retrieval and graph organization.

## V3 changes

V3 keeps the successful 16K explicit memory table and replaces the weak LM-only memory training with four coupled local objectives:

1. **Reconstruction**: `memory.retrieve(z_query_l5) -> reconstruct -> z_fused_l5`.
2. **Adjacent-span contrastive retrieval**: positives are same-sample next spans; negatives are other batch spans. This removes the V2 identity-positive shortcut.
3. **Graph co-activation**: adjacent span graph routing is trained against shuffled far negatives. This gives edge logits a content objective, not only entropy pressure.
4. **MemoryDistiller**: `concat(z_query, z_fused, retrieved) -> z_insight`, preparing future online writes to store distilled insights rather than raw `z_fused`.

## Kernel path

V3 adds `src/latentloop_pds_m/kernels/memory_retrieval.py` and routes exact retrieval through a chunked top-k kernel. It avoids a persistent `[num_queries, n_slots]` allocation and uses bounded `query_chunk x n_slots` score blocks. For 16K local runs this keeps gradients exact and observable; for 64K it prevents peak memory spikes. LSH remains available for larger tables but is not used in the 4070 V3 proof.

## 4070 configuration

- `memory.n_slots = 16384`
- `memory.top_k = 64`
- `memory.use_lsh = false`
- `memory.retrieval_backend = exact_chunked`
- `memory.exact_query_chunk = 128`
- `memory.exact_slot_chunk = 4096`
- `online_update = false`

## Phase2 policy

Phase2 is now an engineering path rather than a missing stage. It should run as:

1. `phase2_data` and `phase2_audit_data` for 3M or 5M.
2. `phase2_phase1_resume` from the 1M Phase1 checkpoint.
3. `phase2_features` from the resumed checkpoint.
4. `phase2_memory_v3` from the V2/V3 memory checkpoint.
5. `phase2_joint` to reconnect memory to LM loss with most backbone weights frozen.

## Stop rules

Do not open `online_update` until retrieval entropy falls below 0.98 of log-top-k and memory-on/off logits RMS delta increases versus the 1M baseline. Do not return to 215K slots until 16K and 64K both show selective retrieval.
