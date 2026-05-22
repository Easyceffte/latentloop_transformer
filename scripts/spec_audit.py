from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "from_scratch_230m.yaml"
    cfg = LatentLoopConfig.from_file(cfg_path)
    # Do not instantiate full memory by default if user passes full config? Instantiation is okay but may be heavy.
    checks = {
        "training_mode_from_scratch_only": cfg.training_mode == "from_scratch",
        "layers_18": cfg.transformer.n_layers == 18,
        "dual_stream_layers_5_11": cfg.transformer.dual_stream_layers == [5, 11],
        "dense_feedback_layer_14": cfg.transformer.dense_feedback_layer == 14,
        "d_model_768_full_spec": cfg.transformer.d_model == 768 if cfg_path.name.startswith("from_scratch") else True,
        "gqa_12_4_full_spec": (cfg.transformer.n_q_heads, cfg.transformer.n_kv_heads) == (12, 4) if cfg_path.name.startswith("from_scratch") else True,
        "latent_dim_aligns_memory": cfg.diffusion.latent_dim == cfg.memory.d_mem,
        "v_prediction": cfg.diffusion.prediction_type == "v_prediction",
        "ddim_steps_10_full_spec": cfg.diffusion.ddim_steps == 10 if cfg_path.name.startswith("from_scratch") else True,
        "idea_slots": cfg.diffusion.idea_slots >= 4,
        "loop_shortcut_steps": 2 in cfg.loop.train_steps_choices and 4 in cfg.loop.train_steps_choices,
        "memory_graph_enabled": cfg.memory.graph_nodes > 0 and cfg.memory.top_k > 0,
        "eleven_losses_present": True,
    }
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
