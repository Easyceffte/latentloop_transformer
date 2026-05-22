from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.data.common import STAGE_TRAIN_BLOCKS, source_weights_for_domain, load_yaml
from run_local_v2_16k_from_zero import mtokens_to_stage, default_paths


def test_local_v2_stage_block_counts_are_seq256_token_budgets():
    assert STAGE_TRAIN_BLOCKS["local_v2_1m"] == 4096
    assert STAGE_TRAIN_BLOCKS["local_v2_3m"] == 12288
    assert STAGE_TRAIN_BLOCKS["local_v2_5m"] == 20480
    assert mtokens_to_stage(1) == "local_v2_1m"
    assert mtokens_to_stage(3) == "local_v2_3m"
    assert mtokens_to_stage(5) == "local_v2_5m"


def test_local_v2_sources_have_required_domains_and_weights():
    cfg = load_yaml(ROOT / "data" / "sources" / "local_v2_sources.yaml")
    sources = cfg["sources"]
    assert any(s.get("domain") == "general" for s in sources.values())
    assert any(s.get("domain") == "reasoning" for s in sources.values())
    general = source_weights_for_domain(sources, "general")
    reasoning = source_weights_for_domain(sources, "reasoning")
    assert abs(sum(general.values()) - 1.0) < 1e-8
    assert abs(sum(reasoning.values()) - 1.0) < 1e-8
    assert sources["openmathreasoning_cot"]["split"] == "cot"
    assert sources["openmathreasoning_cot"]["formatter"] == "problem_solution"


def test_runner_default_paths_are_stage_scoped():
    class Args:
        mtokens = 1
        data_stage = ""
    stage, train, val, test, phase1_dir, ckpt, features, memory_dir = default_paths(Args())
    assert stage == "local_v2_1m"
    assert train.name == "local_v2_1m.train.jsonl"
    assert val.name == "local_v2_1m.val.jsonl"
    assert test.name == "local_v2_1m.test.jsonl"
    assert ckpt.name == "last.pt"
    assert "local_v2_1m" in str(features)
    assert "local_v2_1m" in str(memory_dir)
