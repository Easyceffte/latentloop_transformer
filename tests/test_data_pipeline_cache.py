from __future__ import annotations

import json
from pathlib import Path

from scripts.data import run_data_pipeline


def test_collect_source_docs_reuses_raw_cache_without_remote_access(tmp_path: Path) -> None:
    spec = {"hf_id": "dummy", "domain": "general"}
    cache = tmp_path / "fineweb.raw.jsonl"

    docs, first = run_data_pipeline.collect_source_docs(
        "fineweb_edu",
        spec,
        2,
        seed=123,
        offline_mock=True,
        raw_cache_path=cache,
        reuse_raw_cache=False,
        force_redownload=False,
    )
    assert len(docs) == 2
    assert cache.exists()
    assert first["cache_hit"] is False
    assert first["remote_accessed"] is False
    assert first["records_written"] >= 2

    def boom(*args, **kwargs):
        raise AssertionError("load_hf_iter must not be called on raw-cache hit")

    old_loader = run_data_pipeline.load_hf_iter
    run_data_pipeline.load_hf_iter = boom
    try:
        docs2, second = run_data_pipeline.collect_source_docs(
            "fineweb_edu",
            spec,
            2,
            seed=999,
            offline_mock=False,
            raw_cache_path=cache,
            reuse_raw_cache=True,
            force_redownload=False,
        )
    finally:
        run_data_pipeline.load_hf_iter = old_loader
    assert len(docs2) == 2
    assert second["cache_hit"] is True
    assert second["remote_accessed"] is False
    assert second["records_loaded"] >= 2
    assert second["records_written"] == 0
    assert Path(second["raw_cache_path"]) == cache


def test_source_cache_report_has_required_fields(tmp_path: Path) -> None:
    report = run_data_pipeline._source_cache_report(
        cache_hit=True,
        raw_cache_path=tmp_path / "x.raw.jsonl",
        records_loaded=3,
        records_written=0,
        remote_accessed=False,
    )
    assert set(report) == {"cache_hit", "raw_cache_path", "records_loaded", "records_written", "remote_accessed"}
    assert report["cache_hit"] is True
    assert report["remote_accessed"] is False


def test_collect_divergent_docs_reuses_cache(tmp_path: Path) -> None:
    cache = tmp_path / "synthetic_divergent.raw.jsonl"
    docs, first = run_data_pipeline.collect_divergent_docs(2, seed=1, raw_cache_path=cache, reuse_raw_cache=False, force_redownload=False)
    assert len(docs) == 2
    assert first["cache_hit"] is False
    docs2, second = run_data_pipeline.collect_divergent_docs(2, seed=999, raw_cache_path=cache, reuse_raw_cache=True, force_redownload=False)
    assert len(docs2) == 2
    assert second["cache_hit"] is True
    assert second["remote_accessed"] is False
