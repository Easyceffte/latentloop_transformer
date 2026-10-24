from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.data.audit_dataset import main
from scripts.data.common import write_jsonl


def _args(packed: Path, report: Path) -> list[str]:
    return ["audit_dataset", "--train", str(packed), "--seq_len", "4", "--vocab_size", "64", "--expected_mix", "", "--out", str(report), "--require_zh_dialogue"]


def test_cli_blocks_when_required_raw_data_is_not_provided(tmp_path: Path, monkeypatch) -> None:
    packed = tmp_path / "train.jsonl"
    write_jsonl(packed, [{"input_ids": [2, 12, 14, 3], "source": "chat", "domain": "dialogue"}])
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", _args(packed, output))

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "BLOCKED"
    assert "--raw_train is required" in report["blocked_reasons"][0]
    assert report["splits"]["train"]["decision"] == "PASS"


def test_cli_reports_and_gates_raw_chinese_dialogue(tmp_path: Path, monkeypatch) -> None:
    packed = tmp_path / "train.jsonl"
    raw = tmp_path / "train.raw.jsonl"
    output = tmp_path / "report.json"
    write_jsonl(packed, [{"input_ids": [2, 12, 14, 3], "source": "chat", "domain": "dialogue"}])
    write_jsonl(raw, [{"text": "<|user|>你好<|assistant|>你好", "source": "chat", "domain": "dialogue"}])
    monkeypatch.setattr(sys, "argv", _args(packed, output) + ["--raw_train", str(raw)])

    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "PASS"
    assert report["language_coverage"]["train"]["by_source"]["chat"]["han_characters"] == 4
    assert report["splits"]["train"]["decision"] == "PASS"
