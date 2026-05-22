from __future__ import annotations

import json
from pathlib import Path

from scripts.data.common import DOMAIN_RATIOS, STAGE_TRAIN_BLOCKS, selected_text_from_row
from scripts.run_local_v3_phase0_to_phase3 import stage_name


def test_local_v3_stage_budgets_and_chinese_dialogue_ratio():
    assert STAGE_TRAIN_BLOCKS["local_v3_100m"] == 409600
    assert DOMAIN_RATIOS == {"general": 0.60, "dialogue": 0.20, "reasoning": 0.15, "divergent": 0.05}
    assert stage_name(20) == "local_v3_20m"


def test_instruction_response_formatter_uses_chat_tags():
    row = {"instruction": "你好", "input": "", "output": "你好，我在。"}
    text, field = selected_text_from_row(row, {"formatter": "instruction_response"})
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "你好，我在" in text
    assert field == "instruction+response"


def test_conversation_formatter_accepts_sharegpt_shape():
    row = {"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好，有什么想聊的？"}]}
    text, field = selected_text_from_row(row, {"formatter": "conversation"})
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "有什么想聊" in text
    assert field == "conversation"
