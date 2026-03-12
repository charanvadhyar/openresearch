"""
Error Memory — persistent store of (error pattern → working fix) pairs.

Persists across all runs in ~/.autoresearch/error_memory.json.

How it works:
  1. Fingerprint: extract error type + key terms from a traceback
  2. On fix attempt: recall() returns similar past fixes to inject into prompt
  3. After fix succeeds: remember() stores the (fingerprint, fix) pair
  4. Similarity is term-overlap — no external deps, no network calls
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MEMORY_PATH = Path.home() / ".autoresearch" / "error_memory.json"
_MAX_ENTRIES  = 200   # cap so file doesn't grow unbounded
_MIN_SCORE    = 0.20  # minimum similarity to surface a past fix


# ── Fingerprinting ─────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "was", "with", "this", "that", "it", "be", "as", "by", "from",
    "line", "file", "cell", "block", "error", "exception", "traceback",
    "most", "recent", "last", "call",
}

def _tokenise(text: str) -> set[str]:
    """Lower-case alphanum tokens, filtered."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _error_type(text: str) -> str:
    """Extract the leading exception class name, e.g. 'KeyError'."""
    m = re.search(r"([A-Z][a-zA-Z]+Error|[A-Z][a-zA-Z]+Exception|[A-Z][a-zA-Z]+Warning)", text)
    return m.group(1) if m else "UnknownError"


def fingerprint(error_text: str) -> str:
    """SHA-256 of the normalised error tokens — used as a stable dict key."""
    tokens = sorted(_tokenise(error_text))
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()[:16]


def similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


# ── Persistence ────────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    try:
        if _MEMORY_PATH.exists():
            return json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Error memory load failed: {e}")
    return []


def _save(entries: list[dict]) -> None:
    try:
        _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MEMORY_PATH.write_text(
            json.dumps(entries[-_MAX_ENTRIES:], indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"Error memory save failed: {e}")


# ── Public API ─────────────────────────────────────────────────────────────────

def recall(error_text: str, top_k: int = 3) -> list[dict]:
    """
    Return up to top_k past entries whose error is similar to error_text,
    sorted by similarity descending.

    Each entry: {"error_type": str, "error_snippet": str, "fix_snippet": str, "score": float}
    """
    query_tokens = _tokenise(error_text)
    query_type = _error_type(error_text)
    entries = _load()

    scored = []
    for e in entries:
        if not e.get("worked", True):        # skip known-bad fixes
            continue
        stored_tokens = set(e.get("tokens", []))
        score = similarity(query_tokens, stored_tokens)
        if e.get("error_type") == query_type and query_type != "UnknownError":
            score += 0.15  # prioritize same exception class
        if score >= _MIN_SCORE:
            scored.append({
                "error_type":    e.get("error_type", ""),
                "error_snippet": e.get("error_snippet", "")[:400],
                "fix_snippet":   e.get("fix_snippet",   "")[:800],
                "score":         round(min(score, 1.0), 3),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def remember(error_text: str, fix_text: str, worked: bool = True) -> None:
    """
    Store an (error, fix) pair. Call with worked=True after the retry succeeds,
    worked=False if the fix itself also failed (so we don't replay bad advice).
    """
    entries = _load()
    entries.append({
        "fp":            fingerprint(error_text),
        "tokens":        sorted(_tokenise(error_text)),
        "error_type":    _error_type(error_text),
        "error_snippet": error_text[:600],
        "fix_snippet":   fix_text[:1200],
        "worked":        worked,
    })
    _save(entries)


def mark_failed(error_text: str, fix_text: str) -> None:
    """Mark a previously stored fix as not working."""
    entries = _load()
    fp = fingerprint(error_text)
    fix_fp = hashlib.sha256(fix_text[:400].encode()).hexdigest()[:16]
    for e in entries:
        if e.get("fp") == fp and fix_text[:400] in e.get("fix_snippet", ""):
            e["worked"] = False
    _save(entries)


def format_for_prompt(past_fixes: list[dict]) -> str:
    """
    Format recalled fixes into a block that can be appended to a fix prompt.
    Returns empty string if no past fixes.
    """
    if not past_fixes:
        return ""

    lines = [
        "## Relevant fixes from memory (similar errors that were solved before):",
        "",
    ]
    for i, fix in enumerate(past_fixes, 1):
        lines += [
            f"### Past fix {i}  (similarity {fix['score']:.0%}, error type: {fix['error_type']})",
            "Previous error:",
            f"  {fix['error_snippet'][:200]}",
            "What fixed it:",
            f"  {fix['fix_snippet'][:400]}",
            "",
        ]
    lines.append("Apply the same pattern if relevant, but adapt to the current error.")
    return "\n".join(lines)
