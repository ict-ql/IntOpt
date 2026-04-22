#!/usr/bin/env python3
"""Extract intrinsic declare signatures from LLVM test cases.

Scans llvm/test/ for `declare ... @llvm.*` lines, deduplicates,
and saves a JSON mapping: {intrinsic_name: declare_line}.

Usage:
  python scripts/extract_intrinsic_declares.py \
    --llvm-root /home/amax/yangz/Env/llvm-project \
    --output kb/intrinsic_declares.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from modules.utils import log


def _normalize_declare(decl: str) -> str:
    """Normalize a declare line for deduplication:
    - Strip trailing #N attributes
    - Remove parameter names (%foo, %0, etc.), keep only types
    - Collapse whitespace
    """
    # Remove trailing attributes
    decl = re.sub(r"\s*#\d+\s*$", "", decl).strip()
    # Remove parameter names: %word after a type
    decl = re.sub(r"(%[\w.]+)", "", decl)
    # Remove 'noundef', 'nocapture', 'readonly', 'writeonly', 'immarg', etc.
    for attr in ("noundef", "nocapture", "readonly", "writeonly",
                 "immarg", "nonnull", "signext", "zeroext", "inreg",
                 "dereferenceable\\(\\d+\\)"):
        decl = re.sub(r"\b" + attr + r"\b", "", decl)
    # Collapse multiple spaces/commas
    decl = re.sub(r"\s+", " ", decl)
    decl = re.sub(r",\s*,", ",", decl)
    decl = re.sub(r"\(\s*,", "(", decl)
    decl = re.sub(r",\s*\)", ")", decl)
    return decl.strip()


def extract_declares(llvm_root: str) -> dict:
    """Scan LLVM test directory for declare @llvm.* lines.

    Returns {intrinsic_name: declare_signature}.
    For overloaded intrinsics (multiple type variants), keeps all unique
    signatures as a list."""

    test_dir = Path(llvm_root) / "llvm" / "test"
    if not test_dir.is_dir():
        sys.exit(f"LLVM test dir not found: {test_dir}")

    # Pattern: declare <ret_type> @llvm.xxx.yyy(<args>) [#N]
    declare_re = re.compile(r"^\s*declare\s+(.+?@llvm\.[^\s(]+\([^)]*\))")

    # Collect all declares, grouped by base intrinsic name
    declares: dict = defaultdict(set)
    file_count = 0

    for ll_file in test_dir.rglob("*.ll"):
        file_count += 1
        try:
            text = ll_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for line in text.splitlines():
            m = declare_re.match(line)
            if not m:
                continue
            decl = m.group(1).strip()
            # Normalize: strip attrs, param names, collapse whitespace
            decl = _normalize_declare(decl)
            # Extract intrinsic name
            name_m = re.search(r"@(llvm\.[^\s(]+)", decl)
            if name_m:
                name = name_m.group(1)
                declares[name].add(decl)

        if file_count % 5000 == 0:
            log(f"  Scanned {file_count} files, {len(declares)} intrinsics so far ...")

    log(f"Scanned {file_count} files total, found {len(declares)} unique intrinsics")

    # Convert sets to sorted lists, deduplicated
    result = {}
    for name, decl_set in sorted(declares.items()):
        sigs = sorted(decl_set)
        if len(sigs) == 1:
            # Non-overloaded: store as single string
            result[name] = sigs[0]
        else:
            # Overloaded: keep up to 3 representative variants
            result[name] = sigs[:3]

    return result


def main():
    p = argparse.ArgumentParser(description="Extract intrinsic declare signatures")
    p.add_argument("--llvm-root", default="/home/amax/yangz/Env/llvm-project")
    p.add_argument("--output", default="kb/intrinsic_declares.json")
    args = p.parse_args()

    log(f"Scanning {args.llvm_root}/llvm/test/ ...")
    declares = extract_declares(args.llvm_root)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(declares, f, ensure_ascii=False, indent=2)

    log(f"Saved {len(declares)} intrinsic signatures to {out_path}")


if __name__ == "__main__":
    main()
