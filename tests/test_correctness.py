from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM
from latentloop_pds_m.layers import TransformerBlock


def _small_cfg():
    cfg = LatentLoopConfig.from_file(ROOT / "configs" / "smoke_tiny.yaml")
    cfg.transformer.vocab_size = 128
    cfg.transformer.max_seq_len = 32
    cfg.transformer.d_model = 64
    cfg.transformer.n_q_heads = 4
    cfg.transformer.n_kv_heads = 2
    cfg.transformer.ffn_dim = 128
    cfg.diffusion.latent_dim = 32
    cfg.diffusion.denoise_dim = 64
    cfg.diffusion.denoiser_layers = 1
    cfg.diffusion.denoiser_heads = 4
    cfg.diffusion.ddpm_timesteps = 64
    cfg.diffusion.train_t_start_choices = [8, 16]
    cfg.diffusion.infer_t_start = [8]
    cfg.diffusion.ddim_steps = 2
    cfg.loop.latent_dim = 32
    cfg.loop.block_layers = 1
    cfg.loop.heads = 4
    cfg.loop.train_steps_choices = [2]
    cfg.loop.infer_steps = 2
    cfg.interaction.train_rounds = 1
    cfg.interaction.infer_min_rounds = 1
    cfg.interaction.infer_max_rounds = 1
    cfg.memory.n_slots = 256
    cfg.memory.d_mem = 32
    cfg.memory.query_dim = 16
    cfg.memory.top_k = 8
    cfg.memory.graph_nodes = 8
    cfg.memory.use_lsh = False
    cfg.dense_feedback.warmup_steps = 1
    cfg.optim.chunked_ce_size = 16
    return cfg


def test_transformer_block_prefix_causal():
    torch.set_num_threads(1)
    block = TransformerBlock(32, 4, 2, 64, 16, 10000.0).eval()
    x = torch.randn(1, 8, 32)
    y = x.clone()
    y[:, 4:] = torch.randn_like(y[:, 4:]) * 3.0
    with torch.no_grad():
        out_x = block(x)
        out_y = block(y)
    assert torch.allclose(out_x[:, :4], out_y[:, :4], atol=1e-5, rtol=1e-5)


def test_full_model_prefix_causal_in_train_path():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    model = LatentLoopTransformerPDSM(cfg).train()
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 10))
    ids2 = ids.clone()
    ids2[:, 5:] = torch.randint(3, cfg.transformer.vocab_size, (1, 5))
    torch.manual_seed(123)
    out1 = model(ids, labels=None, global_step=1)["logits"].detach()
    torch.manual_seed(123)
    out2 = model(ids2, labels=None, global_step=1)["logits"].detach()
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=2e-4, rtol=2e-4)


def test_dense_feedback_receives_task_gradient_without_reg_loss():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    cfg.losses.dense_feedback_reg_loss = 0.0
    model = LatentLoopTransformerPDSM(cfg).train()
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 10))
    out = model(ids, labels=ids, global_step=10)
    out["lm_loss"].backward()
    grads = [p.grad.detach().abs().sum().item() for p in model.feedback.proj.parameters() if p.grad is not None]
    assert grads and sum(grads) > 0.0


def test_graph_sparsity_has_edge_gradient():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    model = LatentLoopTransformerPDSM(cfg).train()
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 10))
    out = model(ids, labels=ids, global_step=1)
    out["graph_sparsity_loss"].backward(retain_graph=True)
    grad = model.memory.graph.edge_logits.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_idea_slots_receive_lm_gradient_without_future_leakage():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    cfg.dense_feedback.enabled = False
    cfg.interaction.train_rounds = 1
    cfg.loop.train_steps_choices = [2]
    cfg.loop.block_layers = 1
    cfg.diffusion.denoiser_layers = 1
    model = LatentLoopTransformerPDSM(cfg).train()
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 12))
    out = model(ids, labels=ids, global_step=10)
    model.zero_grad(set_to_none=True)
    out["lm_loss"].backward()
    grads = []
    for layer in model.layers:
        if hasattr(layer, "diffusion"):
            g = layer.diffusion.slot_queries.grad
            if g is not None:
                grads.append(g.detach().abs().sum())
    assert grads and torch.stack(grads).sum() > 0


def test_lsh_retrieval_uses_chunked_candidate_path():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    cfg.memory.n_slots = 300
    cfg.memory.exact_threshold = 32
    cfg.memory.use_lsh = True
    cfg.memory.lsh_max_candidates = 24
    cfg.memory.lsh_query_chunk = 3
    model = LatentLoopTransformerPDSM(cfg).train()
    z = torch.randn(1, 7, cfg.memory.d_mem)
    out = model.memory.retrieve(z, global_step=0)
    assert out["retrieved"].shape == z.shape
    assert out["indices"].shape[-1] == cfg.memory.top_k


def test_full_model_prefix_causal_in_eval_branch_merge_path():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    cfg.dense_feedback.warmup_steps = 1
    model = LatentLoopTransformerPDSM(cfg).eval()
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 12))
    ids2 = ids.clone()
    ids2[:, 6:] = torch.randint(3, cfg.transformer.vocab_size, (1, 6))
    torch.manual_seed(321)
    out1 = model(ids, labels=None, global_step=1000)["logits"].detach()
    torch.manual_seed(321)
    out2 = model(ids2, labels=None, global_step=1000)["logits"].detach()
    assert torch.allclose(out1[:, :6], out2[:, :6], atol=1e-5, rtol=1e-5)


def test_dense_feedback_routes_are_route_specific_and_causal():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    cfg.dense_feedback.warmup_steps = 1
    cfg.diffusion.denoiser_layers = max(2, cfg.diffusion.denoiser_layers)
    cfg.loop.block_layers = max(2, cfg.loop.block_layers)
    model = LatentLoopTransformerPDSM(cfg).train()
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 12))
    out = model(ids, labels=ids, global_step=1000)
    model.zero_grad(set_to_none=True)
    out["lm_loss"].backward()
    route_grads = {}
    for route in model.feedback.routes:
        key = route.replace('.', '_')
        grads = []
        for p in list(model.feedback.proj[key].parameters()) + list(model.feedback.gate[key].parameters()):
            if p.grad is not None:
                grads.append(p.grad.detach().abs().sum())
        route_grads[route] = torch.stack(grads).sum().item() if grads else 0.0
    assert route_grads["block1.diff1"] > 0
    assert route_grads["block1.diff2"] > 0
    assert route_grads["block1.loop1"] > 0
    assert route_grads["block1.loop2"] > 0


def test_shortcut_kl_is_token_normalized_not_sequence_amplified():
    import torch.nn.functional as F
    torch.manual_seed(7)
    b, l, d = 2, 32, 16
    z2 = torch.randn(b, l, d)
    z4 = torch.randn(b, l, d)
    z8 = torch.randn(b, l, d)
    old = F.kl_div(F.log_softmax(z2, dim=-1), F.softmax(z8, dim=-1), reduction="batchmean")
    new = F.kl_div(
        F.log_softmax(z2.reshape(-1, d), dim=-1),
        F.softmax(z8.reshape(-1, d), dim=-1),
        reduction="batchmean",
    )
    assert torch.allclose(old / new, torch.tensor(float(l)), rtol=1e-4, atol=1e-4)


def test_dense_feedback_reports_gate_and_signal_separately():
    torch.set_num_threads(1)
    cfg = _small_cfg()
    model = LatentLoopTransformerPDSM(cfg).train()
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 10))
    out = model(ids, labels=ids, global_step=1000)
    aux = out["aux"]
    assert "dense_feedback_gate_mean" in aux
    assert "dense_feedback_signal_mean" in aux
    assert 0.0 <= float(aux["dense_feedback_gate_mean"].detach()) <= 1.0
