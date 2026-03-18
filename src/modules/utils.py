"""Shared utilities used across pipeline modules."""

import re
import time
from pathlib import Path
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Timestamped log line (replaces bare print throughout the project)."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_HTML_UNESCAPE = (
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&amp;", "&"),
)


def unescape_html(s: str) -> str:
    for a, b in _HTML_UNESCAPE:
        s = s.replace(a, b)
    return s


def clean_block(s: str) -> str:
    """Unescape HTML entities and strip whitespace."""
    s = unescape_html(s or "")
    s = s.strip("\n\r\t ")
    return s


# ---------------------------------------------------------------------------
# Tag extraction  (<code>…</code>, <advice>…</advice>, etc.)
# ---------------------------------------------------------------------------

def extract_tagged_blocks(text: str, tag: str) -> List[str]:
    """Return non-empty content blocks wrapped in *tag* (case-insensitive).

    Example:  extract_tagged_blocks(text, "code")  →  contents of <code>…</code>
    """
    pat = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    blocks: List[str] = []
    for m in pat.finditer(text):
        blk = clean_block(m.group(1))
        if blk == "..." or not blk:
            continue
        blocks.append(blk)
    return blocks


def extract_single_block(text: str, tag: str) -> Optional[str]:
    """Return the single block for *tag*, or None if absent."""
    blocks = extract_tagged_blocks(text, tag)
    if not blocks:
        return None
    return blocks[0]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path
