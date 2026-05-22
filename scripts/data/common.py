from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

STAGE_TRAIN_BLOCKS = {
    # Legacy 512-token stages.
    "real_smoke_1m": 2048,
    "real_short_5m": 10240,
    "warmup_probe_70m": 143360,
    # Local V2 4070 stages are defined for seq_len=256 so the stage name
    # approximately matches train-token budget: blocks = mtokens * 1_000_000 / 256.
    "local_v2_1m": 4096,
    "local_v2_3m": 12288,
    "local_v2_5m": 20480,
    # Local V3 stages, seq_len=256. The naming is approximate train-token budget.
    "local_v3_1m": 4096,
    "local_v3_5m": 20480,
    "local_v3_20m": 81920,
    "local_v3_100m": 409600,
}
# V3 targets native Chinese dialog capability, not only base-LM web continuation.
# Domain names are preserved into packed blocks and audited.
DOMAIN_RATIOS = {"general": 0.60, "dialogue": 0.20, "reasoning": 0.15, "divergent": 0.05}

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
URL_RE = re.compile(r"https?://|www\.", re.I)
WS_RE = re.compile(r"[ \t\r\f\v]+")


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_int_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)


def sha1_ids(ids: Sequence[int]) -> str:
    return hashlib.sha1((",".join(map(str, ids))).encode("ascii")).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    # Collapse excessive blank lines but preserve paragraph boundaries.
    out: List[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip()


def repeated_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    repeats = 0
    prev = None
    run = 0
    for ch in text:
        if ch == prev:
            run += 1
            if run >= 3:
                repeats += 1
        else:
            prev = ch
            run = 1
    return repeats / max(1, len(text))


def repeated_ngram_ratio(text: str, n: int = 5) -> float:
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return repeated / max(1, len(grams))


def alpha_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def url_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(URL_RE.findall(text)) / max(1, len(text) / 100)


def passes_quality(text: str, domain: str = "general") -> Tuple[bool, str]:
    if len(text) < 128:
        return False, "too_short_chars"
    if repeated_char_ratio(text) > 0.20:
        return False, "repeated_chars"
    if repeated_ngram_ratio(text) > 0.30:
        return False, "repeated_5grams"
    if url_ratio(text) > 0.25:
        return False, "too_many_urls"
    if domain != "reasoning" and alpha_ratio(text) < 0.35:
        return False, "low_alpha_ratio"
    # Avoid minified code / pathological one-liners unless code subdomain handles it upstream.
    longest_line = max((len(x) for x in text.splitlines()), default=0)
    if longest_line > 4000:
        return False, "very_long_line"
    return True, "ok"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def selected_text_from_row(row: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    formatter = spec.get("formatter")
    if formatter == "problem_solution":
        problem = stringify_text_value(row.get("problem", row.get("question", ""))).strip()
        solution = stringify_text_value(row.get("generated_solution", row.get("solution", row.get("answer", "")))).strip()
        if problem and solution:
            return f"Problem:\n{problem}\n\nSolution:\n{solution}", "problem+generated_solution"
        # Fall through to priority fields if a remote schema unexpectedly lacks one side.
    if formatter in {"instruction_response", "chatml"}:
        inst = stringify_text_value(row.get("instruction", row.get("prompt", row.get("query", row.get("question", ""))))).strip()
        inp = stringify_text_value(row.get("input", "")).strip()
        resp = stringify_text_value(row.get("output", row.get("response", row.get("answer", row.get("chosen", ""))))).strip()
        if inst and resp:
            user = inst if not inp else f"{inst}\n{inp}"
            return f"<|user|>\n{user}\n<|assistant|>\n{resp}", "instruction+response"
    if formatter == "conversation":
        conv = row.get("conversations", row.get("messages", row.get("conversation", None)))
        if isinstance(conv, list):
            parts = []
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("from", turn.get("role", "user"))).lower()
                val = stringify_text_value(turn.get("value", turn.get("content", turn.get("text", "")))).strip()
                if not val:
                    continue
                tag = "assistant" if role in {"assistant", "gpt", "bot"} else "user"
                parts.append(f"<|{tag}|>\n{val}")
            if len(parts) >= 2:
                return "\n".join(parts), "conversation"
    if "text_field" in spec and spec.get("text_field"):
        field = spec["text_field"]
        value = row.get(field, "")
        return stringify_text_value(value), field
    for field in spec.get("text_field_priority", []) or []:
        if field in row and row.get(field) not in (None, ""):
            return stringify_text_value(row.get(field)), field
    # Common fallback: concatenate text-like strings, but report the synthetic field name.
    parts = []
    for k, v in row.items():
        if isinstance(v, str) and len(v) > 20 and k.lower() not in {"id", "url", "license"}:
            parts.append(v)
    return "\n".join(parts), "__concat_text_like__" if parts else None


def stringify_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(stringify_text_value(x) for x in value)
    if isinstance(value, dict):
        # Prefer stable, readable order.
        return "\n".join(f"{k}: {stringify_text_value(v)}" for k, v in sorted(value.items()))
    return str(value)


def allocate_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {k: total * v for k, v in ratios.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    rest = total - sum(counts.values())
    for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True)[:rest]:
        counts[k] += 1
    return counts


def source_weights_for_domain(sources: Dict[str, Any], domain: str) -> Dict[str, float]:
    out = {}
    for name, spec in sources.items():
        if spec.get("domain") == domain:
            out[name] = float(spec.get("default_weight", 1.0))
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def make_mock_documents(source_name: str, domain: str, n: int, seed: int = 0) -> Iterator[Dict[str, Any]]:
    rng = random.Random(stable_int_hash(source_name) ^ seed)
    objects = ["paper clip", "glass cup", "rope", "brick", "key", "rubber band", "spoon"]
    verbs = ["compare", "measure", "classify", "observe", "predict", "explain", "revise", "connect", "separate", "estimate"]
    topics = ["energy", "shape", "motion", "language", "memory", "planning", "causality", "design", "evidence", "constraint"]
    endings = ["rule", "example", "counterexample", "boundary", "mechanism", "test", "diagram", "story", "calculation", "analogy"]
    for i in range(n):
        obj = objects[(i + rng.randint(0, 100)) % len(objects)]
        sentences = []
        if domain == "general":
            for j in range(96):
                sentences.append(
                    f"Educational note {i}-{j}: learners {verbs[(i+j)%len(verbs)]} topic {topics[(i*3+j)%len(topics)]} with object {obj} and marker {rng.randint(1000,9999)}. "
                    f"Then they form {endings[(j+i)%len(endings)]} {rng.randint(100,999)} using context {topics[(j*7+i)%len(topics)]}."
                )
        elif domain == "reasoning":
            for j in range(96):
                a, b, c = rng.randint(3, 99), rng.randint(3, 99), rng.randint(1, 20)
                sentences.append(
                    f"Problem {i}-{j}: a value starts at {a}, changes by {b}, and is checked with offset {c}. "
                    f"Step {j} computes {a}+{b}={a+b}, compares {a+b}-{c}={a+b-c}, and explains assumption {rng.randint(100,999)}."
                )
        else:
            cats = ["science", "repair", "art", "education", "game", "measurement"]
            for j, cat in enumerate(cats * 16):
                sentences.append(
                    f"Divergent idea {i}-{j} for object {obj} in category {cat}: relation {rng.randint(10,99)} changes property {topics[j%len(topics)]} with limit {rng.randint(100,999)}."
                )
        text = "\n".join(sentences)
        yield {"text": text, "source": source_name, "domain": domain, "subdomain": "mock"}

def build_divergent_documents(n: int, seed: int = 0) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    objects = [
        "paper clip", "brick", "glass cup", "rubber band", "spoon", "shoelace", "cardboard box", "coin", "toothpick", "magnet",
        "clothespin", "plastic bottle", "key", "rope", "button", "pencil", "binder clip", "paper towel", "straw", "jar lid",
    ]
    categories = ["physics", "repair", "education", "art", "game", "measurement", "sound", "interface", "safety", "organization"]
    for i in range(n):
        obj = objects[i % len(objects)]
        cats = rng.sample(categories, 6)
        expanded_cats = (cats * 8)[:48]
        uses = []
        for j, cat in enumerate(expanded_cats, start=1):
            uses.append(
                f"{j}. {cat}: Use the {obj} as a constraint element in scenario {rng.randint(100,999)}. "
                f"Mechanism: change property {categories[(j+i)%len(categories)]}, position {rng.randint(10,99)}, texture, or relation to another object. "
                f"Limit: keep the action safe, reversible, and distinct from ordinary storage pattern {rng.randint(1000,9999)}."
            )
        text = (
            f"Alternative Uses Task. Object: {obj}. Requirement: produce ideas across different categories, not variants of the same use.\n"
            + "\n".join(uses)
            + "\nEvaluation: count category coverage, remove duplicates, and prefer mechanisms over vague slogans."
        )
        yield {
            "text": text,
            "source": "synthetic_divergent",
            "domain": "divergent",
            "prompt_type": "alternative_uses",
            "object": obj,
            "category_labels": cats,
        }


def ensure_clean_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_counts(train_blocks: int) -> Dict[str, int]:
    # Train target follows the spec. Val/test are small additional audited sets.
    # Tiny test overrides should not spend minutes building 32-block eval splits.
    min_eval = 32 if train_blocks >= 2048 else 4
    return {
        "train": train_blocks,
        "val": max(min_eval, train_blocks // 100),
        "test": max(min_eval, train_blocks // 100),
    }
