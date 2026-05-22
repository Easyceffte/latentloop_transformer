from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from scripts.data.common import (
    DOMAIN_RATIOS,
    STAGE_TRAIN_BLOCKS,
    allocate_counts,
    build_divergent_documents,
    load_yaml,
    make_mock_documents,
    normalize_text,
    passes_quality,
    read_jsonl,
    selected_text_from_row,
    sha1_text,
    source_weights_for_domain,
    split_counts,
    write_jsonl,
)
from scripts.data.tokenize_and_pack import pack_records


def load_hf_iter(name: str, spec: Dict[str, Any], hf_cache_dir: str | None = None) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(f"datasets is required for remote sources. Install with pip install -e .[data]. Error: {e}")
    hf_id = spec["hf_id"]
    subset = spec.get("subset")
    split = spec.get("split", "train")
    kwargs = {"split": split, "streaming": bool(spec.get("streaming", True))}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir
    ds = load_dataset(hf_id, subset, **kwargs) if subset else load_dataset(hf_id, **kwargs)
    for row in ds:
        text, field = selected_text_from_row(dict(row), spec)
        yield {
            "text": text,
            "source": name,
            "domain": spec.get("domain", "unknown"),
            "subdomain": spec.get("subdomain", "unknown"),
            "selected_text_field": field,
        }


def _filter_cached_docs(
    rows: Iterable[Dict[str, Any]],
    source_name: str,
    num_docs: int,
    exclude_hashes: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    seen = set()
    excluded = exclude_hashes or set()
    for row in rows:
        text = normalize_text(row.get("text", ""))
        if not text:
            continue
        h = str(row.get("text_hash") or sha1_text(text))
        if h in excluded or h in seen:
            continue
        seen.add(h)
        row["text"] = text
        row["text_hash"] = h
        docs.append(row)
        if len(docs) >= num_docs:
            break
    if len(docs) < num_docs:
        raise RuntimeError(f"raw cache for source {source_name} yielded {len(docs)} usable docs, need {num_docs}")
    return docs


def _source_cache_report(
    *,
    cache_hit: bool,
    raw_cache_path: Path,
    records_loaded: int,
    records_written: int,
    remote_accessed: bool,
) -> Dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "raw_cache_path": str(raw_cache_path),
        "records_loaded": records_loaded,
        "records_written": records_written,
        "remote_accessed": remote_accessed,
    }


def collect_source_docs(
    name: str,
    spec: Dict[str, Any],
    num_docs: int,
    seed: int,
    offline_mock: bool,
    raw_cache_path: Path,
    reuse_raw_cache: bool,
    force_redownload: bool,
    hf_cache_dir: str | None = None,
    exclude_hashes: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if reuse_raw_cache and raw_cache_path.exists() and not force_redownload:
        cached_rows = list(read_jsonl(raw_cache_path))
        docs = _filter_cached_docs(cached_rows, name, num_docs, exclude_hashes)
        return docs, _source_cache_report(
            cache_hit=True,
            raw_cache_path=raw_cache_path,
            records_loaded=len(cached_rows),
            records_written=0,
            remote_accessed=False,
        )

    if offline_mock:
        raw_iter = make_mock_documents(name, spec.get("domain", "general"), num_docs * 2 + 32, seed=seed)
    else:
        raw_iter = load_hf_iter(name, spec, hf_cache_dir=hf_cache_dir)
    docs: List[Dict[str, Any]] = []
    seen = set()
    exclude_hashes = exclude_hashes or set()
    filter_counts: Counter[str] = Counter()
    rows_seen = 0
    for row in raw_iter:
        rows_seen += 1
        text = normalize_text(row.get("text", ""))
        row["text"] = text
        ok, reason = passes_quality(text, row.get("domain", "general"))
        # Offline mock rows are deterministic stress fixtures, not quality-filter benchmarks.
        # Keep them unless they are truly empty/too short so CI can exercise the pipeline without Internet.
        if offline_mock and len(text) >= 128:
            ok, reason = True, "mock_kept"
        filter_counts[reason] += 1
        if not ok:
            continue
        h = sha1_text(text)
        if h in exclude_hashes:
            filter_counts["excluded_previous_split"] += 1
            continue
        if h in seen:
            filter_counts["duplicate"] += 1
            continue
        seen.add(h)
        row["text_hash"] = h
        docs.append(row)
        if len(docs) >= num_docs:
            break
    if len(docs) < num_docs:
        raise RuntimeError(f"source {name} yielded {len(docs)} usable docs, need {num_docs}; filters={dict(filter_counts)}")
    records_written = write_jsonl(raw_cache_path, docs)
    return docs, _source_cache_report(
        cache_hit=False,
        raw_cache_path=raw_cache_path,
        records_loaded=rows_seen,
        records_written=records_written,
        remote_accessed=not offline_mock,
    )


def collect_divergent_docs(
    num_docs: int,
    seed: int,
    raw_cache_path: Path,
    reuse_raw_cache: bool,
    force_redownload: bool,
    exclude_hashes: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if reuse_raw_cache and raw_cache_path.exists() and not force_redownload:
        cached_rows = list(read_jsonl(raw_cache_path))
        docs = _filter_cached_docs(cached_rows, "synthetic_divergent", num_docs, exclude_hashes)
        return docs, _source_cache_report(
            cache_hit=True,
            raw_cache_path=raw_cache_path,
            records_loaded=len(cached_rows),
            records_written=0,
            remote_accessed=False,
        )

    docs: List[Dict[str, Any]] = []
    seen = set()
    exclude_hashes = exclude_hashes or set()
    for rec in build_divergent_documents(num_docs * 4, seed + 17):
        h = sha1_text(rec["text"])
        if h in exclude_hashes or h in seen:
            continue
        seen.add(h)
        rec["text_hash"] = h
        docs.append(rec)
        if len(docs) >= num_docs:
            break
    if len(docs) < num_docs:
        raise RuntimeError(f"divergent yielded {len(docs)} usable docs, need {num_docs}")
    records_written = write_jsonl(raw_cache_path, docs)
    return docs, _source_cache_report(
        cache_hit=False,
        raw_cache_path=raw_cache_path,
        records_loaded=len(docs),
        records_written=records_written,
        remote_accessed=False,
    )


def make_docs_for_split(
    cfg: Dict[str, Any],
    stage: str,
    split: str,
    target_blocks: int,
    seed: int,
    offline_mock: bool,
    raw_cache_dir: Path,
    reuse_raw_cache: bool,
    force_redownload: bool,
    hf_cache_dir: str | None = None,
    exclude_hashes: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    sources = cfg.get("sources", {})
    domain_block_counts = allocate_counts(target_blocks, DOMAIN_RATIOS)
    docs: List[Dict[str, Any]] = []
    source_reports: Dict[str, Dict[str, Any]] = {}
    exclude_hashes = exclude_hashes or set()
    # A conservative heuristic: for 512-token blocks, 3-6 medium docs per block is enough after packing.
    docs_per_block = 4
    for domain, block_count in domain_block_counts.items():
        if domain == "divergent":
            need = max(64, block_count * docs_per_block)
            raw_cache_path = raw_cache_dir / stage / split / "synthetic_divergent.raw.jsonl"
            source_docs, source_reports["synthetic_divergent"] = collect_divergent_docs(
                need,
                seed,
                raw_cache_path,
                reuse_raw_cache,
                force_redownload,
                exclude_hashes=exclude_hashes,
            )
            docs.extend(source_docs)
            continue
        weights = source_weights_for_domain(sources, domain)
        source_counts = allocate_counts(block_count * docs_per_block, weights)
        for name, doc_count in source_counts.items():
            spec = sources[name]
            raw_cache_path = raw_cache_dir / stage / split / f"{name}.raw.jsonl"
            source_docs, source_reports[name] = collect_source_docs(
                name,
                spec,
                doc_count,
                seed=seed + len(docs),
                offline_mock=offline_mock,
                raw_cache_path=raw_cache_path,
                reuse_raw_cache=reuse_raw_cache,
                force_redownload=force_redownload,
                hf_cache_dir=hf_cache_dir,
                exclude_hashes=exclude_hashes,
            )
            docs.extend(source_docs)
    rng.shuffle(docs)
    for rec in docs:
        rec["split"] = split
    return docs, source_reports


def train_tokenizer_if_needed(raw_jsonl: Path, tokenizer_out: Path, reuse_tokenizer: str | None, vocab_size: int, allow_mock_fallback: bool = False) -> Path:
    if reuse_tokenizer:
        p = Path(reuse_tokenizer)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    tokenizer_out.mkdir(parents=True, exist_ok=True)
    try:
        from scripts.data.train_tokenizer import train_sentencepiece
    except Exception:
        train_sentencepiece = None
    train_txt = tokenizer_out / "tokenizer_train.txt"
    chars = 0
    docs = 0
    with train_txt.open("w", encoding="utf-8") as f:
        for rec in read_jsonl(raw_jsonl):
            text = (rec.get("text") or "").strip()
            if text:
                f.write(text.replace("\n", " ") + "\n")
                chars += len(text)
                docs += 1
    if docs == 0:
        raise RuntimeError("No docs for tokenizer training")
    try:
        if train_sentencepiece is None:
            raise RuntimeError("sentencepiece trainer unavailable")
        model = train_sentencepiece(train_txt, tokenizer_out, vocab_size=vocab_size, character_coverage=0.9995, input_sentence_size=min(2_000_000, max(1000, docs)))
        tokenizer_type = "sentencepiece_bpe"
    except Exception as e:
        if not allow_mock_fallback:
            raise
        from scripts.data.tokenizer_io import save_fallback_tokenizer
        model = save_fallback_tokenizer(tokenizer_out / "latentloop_spm32k.mock.json", vocab_size)
        tokenizer_type = "hash_fallback_mock_only"
    report = {
        "decision": "PASS",
        "model_path": str(model),
        "tokenizer_type": tokenizer_type,
        "requested_vocab_size": vocab_size,
        "training_docs": docs,
        "training_chars": chars,
    }
    (tokenizer_out / "tokenizer_config.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def write_split_blocks(docs: List[Dict[str, Any]], tokenizer_model: str, seq_len: int, target_blocks: int, out_path: Path) -> int:
    # Pack per domain to preserve the configured ratio exactly by block count.
    from scripts.data.common import DOMAIN_RATIOS, allocate_counts
    rng = random.Random(12345 + target_blocks)
    domain_counts = allocate_counts(target_blocks, DOMAIN_RATIOS)
    all_blocks: List[Dict[str, Any]] = []
    seen_block_hashes: set[str] = set()
    by_domain: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAIN_RATIOS}
    for rec in docs:
        d = rec.get("domain", "unknown")
        if d in by_domain:
            by_domain[d].append(rec)
    for domain, n_blocks in domain_counts.items():
        if n_blocks <= 0:
            continue
        domain_docs = by_domain[domain]
        if not domain_docs:
            raise RuntimeError(f"no docs for domain {domain}")
        blocks = []
        for block in pack_records(domain_docs, tokenizer_model, seq_len=seq_len, target_blocks=None, min_valid_tokens=min(480, seq_len)):
            h = str(block.get("block_hash", ""))
            if h in seen_block_hashes:
                continue
            seen_block_hashes.add(h)
            blocks.append(block)
            if len(blocks) >= n_blocks:
                break
        if len(blocks) != n_blocks:
            raise RuntimeError(f"packing produced {len(blocks)} {domain} blocks for {out_path}, expected {n_blocks}")
        all_blocks.extend(blocks)
    rng.shuffle(all_blocks)
    n = write_jsonl(out_path, all_blocks)
    if n != target_blocks:
        raise RuntimeError(f"packing produced {n} blocks for {out_path}, expected {target_blocks}")
    return n

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LatentLoop real-data preparation pipeline.")
    ap.add_argument("--stage", choices=sorted(STAGE_TRAIN_BLOCKS), default="real_smoke_1m")
    ap.add_argument("--sources", default="data/sources/data_sources.yaml")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--tokenizer_out", default="data/tokenizer")
    ap.add_argument("--reuse_tokenizer", default=None)
    ap.add_argument("--output_dir", default="data/processed")
    ap.add_argument("--report_dir", default="data/reports")
    ap.add_argument("--raw_cache_dir", default="data/raw_cache")
    ap.add_argument("--intermediate_dir", default="data/intermediate")
    ap.add_argument("--reuse_raw_cache", action="store_true")
    ap.add_argument("--force_redownload", action="store_true")
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline_mock", action="store_true", help="Use local deterministic mock docs; does not verify remote downloadability.")
    ap.add_argument("--tokenizer_vocab_size", type=int, default=32000)
    ap.add_argument("--max_train_blocks_override", type=int, default=None, help="Testing hook: override stage train blocks.")
    args = ap.parse_args()
    if args.reuse_raw_cache and args.force_redownload:
        raise ValueError("--reuse_raw_cache and --force_redownload are mutually exclusive")

    cfg = load_yaml(args.sources)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = Path(args.raw_cache_dir); raw_cache_dir.mkdir(parents=True, exist_ok=True)
    intermediate = Path(args.intermediate_dir); intermediate.mkdir(parents=True, exist_ok=True)

    train_blocks = args.max_train_blocks_override or STAGE_TRAIN_BLOCKS[args.stage]
    counts = split_counts(train_blocks)
    pipeline_report: Dict[str, Any] = {
        "stage": args.stage,
        "seq_len": args.seq_len,
        "offline_mock": args.offline_mock,
        "raw_cache_dir": str(raw_cache_dir),
        "intermediate_dir": str(intermediate),
        "reuse_raw_cache": args.reuse_raw_cache,
        "force_redownload": args.force_redownload,
        "hf_cache_dir": args.hf_cache_dir,
        "train_blocks": train_blocks,
        "split_counts": counts,
        "blocked_reasons": [],
        "source_cache": {},
    }

    raw_paths: Dict[str, Path] = {}
    # Collect train first; tokenizer trains from train docs only.
    all_train_docs, train_source_reports = make_docs_for_split(
        cfg,
        args.stage,
        "train",
        train_blocks,
        args.seed,
        args.offline_mock,
        raw_cache_dir,
        args.reuse_raw_cache,
        args.force_redownload,
        hf_cache_dir=args.hf_cache_dir,
    )
    pipeline_report["source_cache"]["train"] = train_source_reports
    used_text_hashes = {sha1_text(str(rec.get("text", ""))) for rec in all_train_docs}
    raw_train = intermediate / f"{args.stage}.train.raw.jsonl"
    write_jsonl(raw_train, all_train_docs)
    raw_paths["train"] = raw_train

    tokenizer_model = train_tokenizer_if_needed(raw_train, Path(args.tokenizer_out), args.reuse_tokenizer, args.tokenizer_vocab_size, allow_mock_fallback=args.offline_mock)
    pipeline_report["tokenizer_model"] = str(tokenizer_model)

    for split, n_blocks in counts.items():
        if split == "train":
            docs = all_train_docs
        else:
            docs, split_source_reports = make_docs_for_split(
                cfg,
                args.stage,
                split,
                n_blocks,
                args.seed + (101 if split == "val" else 202),
                args.offline_mock,
                raw_cache_dir,
                args.reuse_raw_cache,
                args.force_redownload,
                hf_cache_dir=args.hf_cache_dir,
                exclude_hashes=used_text_hashes,
            )
            pipeline_report["source_cache"][split] = split_source_reports
            used_text_hashes.update(sha1_text(str(rec.get("text", ""))) for rec in docs)
            raw_path = intermediate / f"{args.stage}.{split}.raw.jsonl"
            write_jsonl(raw_path, docs)
            raw_paths[split] = raw_path
        out_path = output_dir / f"{args.stage}.{split}.jsonl"
        n = write_split_blocks(docs, str(tokenizer_model), args.seq_len, n_blocks, out_path)
        pipeline_report[f"{split}_path"] = str(out_path)
        pipeline_report[f"{split}_blocks"] = n

    # Audit dataset. In offline mock with a tiny vocab override, audit against that vocab; otherwise 32k.
    from scripts.data.audit_dataset import audit_one
    expected = DOMAIN_RATIOS
    split_reports = {}
    split_hashes = {}
    for split in ["train", "val", "test"]:
        p = output_dir / f"{args.stage}.{split}.jsonl"
        # Enforce the exact 85/10/5 mix on train. Val/test can be too small for <1% integer-ratio tolerance.
        res = audit_one(p, args.seq_len, args.tokenizer_vocab_size, expected if split == "train" else None)
        split_hashes[split] = set(res.pop("hashes"))
        split_reports[split] = res
        pipeline_report["blocked_reasons"].extend([f"{split}: {x}" for x in res.get("blocked_reasons", [])])
    for a in split_hashes:
        for b in split_hashes:
            if a >= b:
                continue
            ov = len(split_hashes[a] & split_hashes[b])
            if ov:
                pipeline_report["blocked_reasons"].append(f"{a}/{b} exact block overlap: {ov}")
    audit_report = {
        "decision": "PASS" if not pipeline_report["blocked_reasons"] else "BLOCKED",
        "stage": args.stage,
        "splits": split_reports,
        "blocked_reasons": pipeline_report["blocked_reasons"],
    }
    audit_path = report_dir / f"audit_{args.stage}.json"
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    pipeline_report["audit_path"] = str(audit_path)
    pipeline_report["decision"] = audit_report["decision"]
    report_path = report_dir / f"pipeline_{args.stage}.json"
    report_path.write_text(json.dumps(pipeline_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(pipeline_report, ensure_ascii=False, indent=2))
    if pipeline_report["blocked_reasons"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
