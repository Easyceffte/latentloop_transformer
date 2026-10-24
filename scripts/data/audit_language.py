"""Chinese-script coverage audit of selected, pre-tokenization JSONL records.

This measures Han characters in the text field, not inferred language labels or
packed-token proportions. Han script is a proxy, not a Chinese-language classifier.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from scripts.data.common import read_jsonl

# Unified/compatibility Han ideographs, including supplementary CJK extensions.
# This does not identify the language: Japanese and other languages also use Han.
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]")


def _count_record(stats: Counter[str], han_chars: int, text_chars: int, has_text: bool) -> None:
    stats["records"] += 1
    stats["records_with_han"] += int(han_chars > 0)
    stats["han_characters"] += han_chars
    stats["non_whitespace_characters"] += text_chars
    stats["records_without_text"] += int(not has_text)


def _summary(stats: Counter[str]) -> Dict[str, int | float]:
    records = stats["records"]
    text_chars = stats["non_whitespace_characters"]
    return {
        "records": records,
        "records_with_han": stats["records_with_han"],
        "fraction_records_with_han": stats["records_with_han"] / records if records else 0.0,
        "han_characters": stats["han_characters"],
        "non_whitespace_characters": text_chars,
        "fraction_han_characters": stats["han_characters"] / text_chars if text_chars else 0.0,
        "records_without_text": stats["records_without_text"],
    }


def audit_raw_language(path: str | Path, *, require_zh_dialogue: bool = False) -> Dict[str, Any]:
    """Measure actual Han-script coverage by domain, source, and source/domain.

    require_zh_dialogue is a minimal train-only presence gate. It blocks an
    empty dialogue domain or dialogue with zero Han characters, not a low but
    nonzero fraction. No percentage floor is implied by Local V3's spec.
    """
    path = Path(path)
    overall: Counter[str] = Counter()
    domains: Dict[str, Counter[str]] = defaultdict(Counter)
    sources: Dict[str, Counter[str]] = defaultdict(Counter)
    source_domains: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    blocked: list[str] = []

    if not path.is_file():
        blocked.append(f"raw JSONL not found: {path}; run Phase0 data collection before auditing")
    else:
        for rec in read_jsonl(path):
            raw_text = rec.get("text")
            text = raw_text if isinstance(raw_text, str) else ""
            domain = str(rec.get("domain") or "unknown")
            source = str(rec.get("source") or "unknown")
            han_chars = len(HAN_RE.findall(text))
            text_chars = sum(not c.isspace() for c in text)
            for stats in (overall, domains[domain], sources[source], source_domains[source][domain]):
                _count_record(stats, han_chars, text_chars, bool(text.strip()))

    by_domain = {name: _summary(stats) for name, stats in sorted(domains.items())}
    dialogue = by_domain.get("dialogue")
    if require_zh_dialogue and path.is_file():
        if not dialogue or dialogue["records"] == 0:
            blocked.append("no dialogue records in raw training data; check dialogue sources and filtering")
        elif dialogue["han_characters"] == 0:
            blocked.append("dialogue records contain zero Han characters; inspect Chinese dialogue source extraction and filtering")

    return {
        "path": str(path),
        "measurement_scope": "selected_raw_records_before_tokenization_and_packing",
        "script_detector": "Unicode Han ideographs; not a Chinese-language classifier",
        "require_zh_dialogue": require_zh_dialogue,
        "overall": _summary(overall),
        "by_domain": by_domain,
        "by_source": {name: _summary(stats) for name, stats in sorted(sources.items())},
        "by_source_domain": {
            name: {domain: _summary(stats) for domain, stats in sorted(group.items())}
            for name, group in sorted(source_domains.items())
        },
        "blocked_reasons": blocked,
        "decision": "BLOCKED" if blocked else "PASS",
    }


def add_raw_language_coverage(
    report: Dict[str, Any], split: str, path: str | Path, *, require_zh_dialogue: bool = False
) -> Dict[str, Any]:
    """Attach a split's raw-language audit and propagate failures to its report."""
    result = audit_raw_language(path, require_zh_dialogue=require_zh_dialogue)
    report.setdefault("language_coverage", {})[split] = result
    report.setdefault("blocked_reasons", []).extend(
        f"{split} raw language: {reason}" for reason in result["blocked_reasons"]
    )
    return result
