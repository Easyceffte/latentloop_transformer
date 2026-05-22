from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentloop_pds_m import LatentLoopConfig, LatentLoopTransformerPDSM


def test_tiny_forward_backward():
    torch.set_num_threads(1)
    cfg = LatentLoopConfig.from_file(ROOT / "configs" / "smoke_tiny.yaml")
    model = LatentLoopTransformerPDSM(cfg)
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 12))
    out = model(ids, labels=ids, global_step=0)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_memory_isolation_shapes():
    torch.set_num_threads(1)
    cfg = LatentLoopConfig.from_file(ROOT / "configs" / "smoke_tiny.yaml")
    model = LatentLoopTransformerPDSM(cfg)
    ids = torch.randint(3, cfg.transformer.vocab_size, (1, 8))
    out = model(ids, labels=None, global_step=0)
    assert out["logits"].shape[:2] == ids.shape
    assert "top10_concentration" in out["aux"]
