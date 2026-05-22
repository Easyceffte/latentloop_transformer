from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional
import json
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    max_seq_len: int = 512
    n_layers: int = 18
    d_model: int = 768
    n_q_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2560
    rope_theta: float = 10000.0
    dropout: float = 0.0
    dual_stream_layers: List[int] = field(default_factory=lambda: [5, 11])
    dense_feedback_layer: int = 14
    tie_lm_head: bool = True
    gradient_checkpointing: bool = False


@dataclass
class DiffusionConfig:
    latent_dim: int = 256
    denoise_dim: int = 512
    denoiser_layers: int = 4
    denoiser_heads: int = 8
    ddpm_timesteps: int = 1000
    ddim_steps: int = 10
    prediction_type: Literal["v_prediction"] = "v_prediction"
    train_t_start_choices: List[int] = field(default_factory=lambda: [300, 500, 650, 800])
    infer_t_start: List[int] = field(default_factory=lambda: [300, 500, 650, 800])
    branch_similarity_weight: float = 0.6
    branch_diversity_weight: float = 0.4
    adaptive_exit_threshold: float = 0.01
    idea_slots: int = 16
    train_lm_path: Literal["predicted_x0", "full_reverse"] = "predicted_x0"


@dataclass
class LoopConfig:
    latent_dim: int = 256
    block_layers: int = 4
    heads: int = 8
    train_steps_choices: List[int] = field(default_factory=lambda: [2, 4, 8])
    infer_steps: int = 8
    max_steps: int = 12
    exit_threshold: float = 0.95
    entropy_tau: float = 0.1
    embedding_limit: int = 16


@dataclass
class InteractionConfig:
    train_rounds: int = 3
    infer_min_rounds: int = 2
    infer_max_rounds: int = 5
    entropy_tau: float = 0.1
    adaptive_exit_threshold: float = 0.01


@dataclass
class ControllerConfig:
    alpha_inf: float = 0.8
    alpha_write: float = 0.5
    alpha_read: float = 0.6


@dataclass
class MemoryConfig:
    # Full spec default is 1_000_000. Dev configs should override this down.
    n_slots: int = 1_000_000
    d_mem: int = 256
    query_dim: int = 128
    top_k: int = 128
    graph_nodes: int = 64
    use_lsh: bool = True
    lsh_tables: int = 4
    lsh_bits: int = 16
    exact_threshold: int = 65536
    retrieval_backend: str = "auto"  # auto|exact_chunked|exact_full|lsh
    exact_query_chunk: int = 128
    exact_slot_chunk: int = 4096
    use_memory_distiller: bool = True
    rebuild_lsh_every: int = 5000
    lsh_max_candidates: int = 10000
    lsh_query_chunk: int = 32
    write_threshold_initial: float = 0.3
    write_threshold_target: float = 0.1
    write_threshold_decay_steps: int = 2000
    write_threshold_decay_amount: float = 0.05
    hebbian_eta: float = 0.001
    edge_decay_gamma: float = 0.9995
    edge_prune_threshold: float = 0.01
    online_update: bool = False
    memory_lr: float = 0.05
    low_freq_reuse_ratio: float = 0.2


@dataclass
class DenseFeedbackConfig:
    enabled: bool = True
    n_routes: int = 10
    warmup_steps: int = 2000
    # True implements a differentiable two-pass feedback path in the current forward.
    recompute_current_forward: bool = True
    cached_next_forward: bool = False


@dataclass
class LossConfig:
    lm_loss: float = 1.0
    ddpm_loss: float = 0.5
    anchor_loss: float = 0.05
    slot_diversity_loss: float = 0.03
    exit_entropy_loss: float = 0.02
    shortcut_consistency_loss: float = 0.005
    dense_feedback_reg_loss: float = 0.01
    controller_balance_loss: float = 0.01
    memory_surprise_loss: float = 0.03
    graph_sparsity_loss: float = 0.01
    memory_stability_loss: float = 0.005
    controller_balance_warmup_steps: int = 2000
    anchor_warmup_steps: int = 5000
    anchor_loss_initial: float = 0.01


@dataclass
class OptimConfig:
    lr: float = 1e-3
    min_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 100000
    stable_until_ratio: float = 0.8
    grad_clip: float = 1.0
    micro_batch_size: int = 1
    grad_accum: int = 64
    bf16: bool = True
    chunked_ce_size: int = 128


@dataclass
class QwenFrozenConfig:
    base_model: str = "Qwen/Qwen3-4B-Base"
    hidden_size: int = 2560
    latent_dim: int = 768
    tap_layers: List[int] = field(default_factory=lambda: [8, 18, 24])
    primary_injection_layer: int = 24
    secondary_injection_layer: int = 30
    load_in_4bit: bool = True
    freeze_backbone: bool = True
    use_cache: bool = False


@dataclass
class LatentLoopConfig:
    training_mode: Literal["from_scratch", "qwen_frozen"] = "from_scratch"
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    dense_feedback: DenseFeedbackConfig = field(default_factory=DenseFeedbackConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    qwen: QwenFrozenConfig = field(default_factory=QwenFrozenConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "LatentLoopConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML config files.")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatentLoopConfig":
        def build(dc, key):
            return dc(**data.get(key, {}))
        return cls(
            training_mode=data.get("training_mode", "from_scratch"),
            transformer=build(TransformerConfig, "transformer"),
            diffusion=build(DiffusionConfig, "diffusion"),
            loop=build(LoopConfig, "loop"),
            interaction=build(InteractionConfig, "interaction"),
            controller=build(ControllerConfig, "controller"),
            memory=build(MemoryConfig, "memory"),
            dense_feedback=build(DenseFeedbackConfig, "dense_feedback"),
            losses=build(LossConfig, "losses"),
            optim=build(OptimConfig, "optim"),
            qwen=build(QwenFrozenConfig, "qwen"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = self.to_dict()
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to write YAML config files.")
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
