from __future__ import annotations

from pathlib import Path

from scripts.data.audit_language import add_raw_language_coverage, audit_raw_language
from scripts.data.common import write_jsonl


def test_english_dialogue_is_blocked_even_with_chinese_general(tmp_path: Path) -> None:
    path = tmp_path / "train.raw.jsonl"
    write_jsonl(path, [
        {"source": "wikipedia_zh", "domain": "general", "text": "中国历史与文化。"},
        {"source": "chinese_instruction_alpaca_gpt4", "domain": "dialogue", "text": "<|user|>\nHello\n<|assistant|>\nHi"},
        {"source": "belle_multiturn", "domain": "dialogue", "text": "<|user|>\nHow are you?\n<|assistant|>\nFine"},
    ])

    report = audit_raw_language(path, require_zh_dialogue=True)
    assert report["decision"] == "BLOCKED"
    assert any("zero Han characters" in reason for reason in report["blocked_reasons"])
    assert report["by_domain"]["dialogue"]["records"] == 2
    assert report["by_domain"]["dialogue"]["han_characters"] == 0
    assert report["by_domain"]["general"]["records_with_han"] == 1
    assert report["by_source"]["chinese_instruction_alpaca_gpt4"]["records_with_han"] == 0


def test_mixed_language_counts_actual_script_by_source_and_domain(tmp_path: Path) -> None:
    path = tmp_path / "train.raw.jsonl"
    write_jsonl(path, [
        {"source": "english_named_source", "domain": "dialogue", "text": "<|user|>\n你好\n<|assistant|>\n你好，我在。"},
        {"source": "zh_named_source", "domain": "dialogue", "text": "<|user|>\nHello\n<|assistant|>\nHello there"},
        {"source": "wikipedia_zh", "domain": "general", "text": "中文百科内容。"},
        {"source": "math", "domain": "reasoning", "text": "Find the value of x."},
    ])

    report = audit_raw_language(path, require_zh_dialogue=True)
    assert report["decision"] == "PASS"
    assert report["overall"]["records"] == 4
    assert report["overall"]["records_with_han"] == 2
    assert report["by_domain"]["dialogue"]["records_with_han"] == 1
    assert report["by_domain"]["dialogue"]["fraction_records_with_han"] == 0.5
    assert report["by_domain"]["dialogue"]["han_characters"] == 6
    assert report["by_domain"]["general"]["records_with_han"] == 1
    assert report["by_source"]["english_named_source"]["records_with_han"] == 1
    assert report["by_source"]["zh_named_source"]["records_with_han"] == 0
    assert report["by_source_domain"]["english_named_source"]["dialogue"]["han_characters"] == 6
    assert 0 < report["overall"]["fraction_han_characters"] < 1


def test_missing_dialogue_records_are_blocked(tmp_path: Path) -> None:
    path = tmp_path / "train.raw.jsonl"
    write_jsonl(path, [{"source": "zh_general", "domain": "general", "text": "你好世界"}])
    report = audit_raw_language(path, require_zh_dialogue=True)
    assert report["decision"] == "BLOCKED"
    assert "no dialogue records" in report["blocked_reasons"][0]


def test_language_audit_is_descriptive_for_other_splits_and_legacy(tmp_path: Path) -> None:
    path = tmp_path / "val.raw.jsonl"
    write_jsonl(path, [{"source": "english", "domain": "dialogue", "text": "English only"}])
    report = {"blocked_reasons": ["existing block issue"]}
    add_raw_language_coverage(report, "val", path)
    assert report["blocked_reasons"] == ["existing block issue"]
    assert report["language_coverage"]["val"]["decision"] == "PASS"
    add_raw_language_coverage(report, "train", path, require_zh_dialogue=True)
    assert any(x.startswith("train raw language:") for x in report["blocked_reasons"])


def test_missing_raw_path_is_actionable(tmp_path: Path) -> None:
    report = audit_raw_language(tmp_path / "absent.raw.jsonl", require_zh_dialogue=True)
    assert report["decision"] == "BLOCKED"
    assert "run Phase0 data collection" in report["blocked_reasons"][0]
    assert report["overall"]["records"] == 0


def test_non_string_and_supplementary_han_text(tmp_path: Path) -> None:
    path = tmp_path / "mixed.raw.jsonl"
    write_jsonl(path, [
        {"source": "chat", "domain": "dialogue", "text": "𠀀"},
        {"source": "chat", "domain": "dialogue", "text": None},
    ])
    report = audit_raw_language(path, require_zh_dialogue=True)
    assert report["decision"] == "PASS"
    assert report["by_source"]["chat"]["han_characters"] == 1
    assert report["by_source"]["chat"]["records_without_text"] == 1


def test_phase0_audit_runner_passes_raw_split_paths_and_chinese_gate(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_local_v3_phase0_to_phase3 as runner

    captured = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "run", lambda command, env=None: captured.append(command))
    monkeypatch.setattr("sys.argv", ["runner", "--stage", "audit_data", "--mtokens", "1"])
    runner.main()

    assert len(captured) == 1
    command = captured[0]
    assert "--require_zh_dialogue" in command
    for split in ("train", "val", "test"):
        assert command[command.index(f"--raw_{split}") + 1] == f"data/intermediate/local_v3_1m.{split}.raw.jsonl"
