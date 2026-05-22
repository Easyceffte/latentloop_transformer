from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train import _validate_processed_data_path


def test_train_requires_existing_processed_jsonl(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed"
    processed_root.mkdir(parents=True)
    data_path = processed_root / "stage.train.jsonl"
    data_path.write_text('{"input_ids":[2,3,4],"labels":[2,3,4]}\n', encoding="utf-8")

    assert _validate_processed_data_path(str(data_path), processed_root=processed_root) == data_path

    with pytest.raises(ValueError, match="--data is required"):
        _validate_processed_data_path("", processed_root=processed_root)

    with pytest.raises(FileNotFoundError, match="training data does not exist"):
        _validate_processed_data_path(str(processed_root / "missing.jsonl"), processed_root=processed_root)

    outside_path = tmp_path / "raw.jsonl"
    outside_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training data must be under"):
        _validate_processed_data_path(str(outside_path), processed_root=processed_root)

    bad_suffix = processed_root / "stage.train.txt"
    bad_suffix.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="processed .jsonl"):
        _validate_processed_data_path(str(bad_suffix), processed_root=processed_root)

from scripts.train import _validate_vocab_compatibility


def test_vocab_compatibility_preflight_fails_before_cuda_for_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"input_ids":[2, 3, 31999]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Data/config vocab mismatch"):
        _validate_vocab_compatibility(p, vocab_size=512)


def test_vocab_compatibility_preflight_passes_for_matching_vocab(tmp_path: Path) -> None:
    p = tmp_path / "ok.jsonl"
    p.write_text('{"input_ids":[2, 3, 31999]}\n', encoding="utf-8")
    stats = _validate_vocab_compatibility(p, vocab_size=32000)
    assert stats["max_token_id"] == 31999
