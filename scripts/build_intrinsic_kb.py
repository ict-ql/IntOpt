#!/usr/bin/env python3
"""Offline script: build the intrinsic knowledge base.

Usage:
  python scripts/build_intrinsic_kb.py --config config/config.yaml [--archs x86,generic]
"""

import argparse
import os
import sys
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.llm_client import LLMClient
from modules.intrinsic_advisor import build_kb
from modules.utils import log


def main():
    p = argparse.ArgumentParser(description="Build intrinsic knowledge base")
    p.add_argument("--config", default="../config/config.yaml")
    p.add_argument("--archs", default="x86,generic",
                   help="Comma-separated target architectures to include")
    p.add_argument("--inc-path", default="",
                   help="Path to IntrinsicImpl.inc (auto-detected from config)")
    p.add_argument("--batch-size", type=int, default=20,
                   help="Number of intrinsics per LLM batch call")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process first N intrinsics (0 = all, useful for debugging)")
    args = p.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    ia_cfg = cfg.get("intrinsic_advisor", {})
    kb_path = ia_cfg.get("intrinsic_kb_path", "kb")

    # Redirect TMPDIR to a subdirectory next to the KB path,
    # so PyTorch/sentence-transformers don't fill up /tmp
    tmpdir = os.path.join(os.path.abspath(kb_path), "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    os.environ["TEMP"] = tmpdir
    os.environ["TMP"] = tmpdir

    llm_cfg = cfg.get("llm", {})
    llvm_cfg = cfg.get("llvm", {})

    # Resolve paths
    inc_path = args.inc_path
    if not inc_path:
        opt_bin = llvm_cfg.get("opt_bin", "")
        if opt_bin:
            # Derive from opt_bin: .../build/bin/opt → .../build/include/llvm/IR/IntrinsicImpl.inc
            build_dir = str(os.path.dirname(os.path.dirname(opt_bin)))
            inc_path = os.path.join(build_dir, "include", "llvm", "IR", "IntrinsicImpl.inc")
        else:
            inc_path = "/home/amax/yangz/Env/llvm-project/build/include/llvm/IR/IntrinsicImpl.inc"

    if not os.path.exists(inc_path):
        sys.exit(f"IntrinsicImpl.inc not found: {inc_path}")

    embedding_model = ia_cfg.get("intrinsic_embedding_model", "all-MiniLM-L6-v2")
    model = llm_cfg.get("llm_model", "gpt-5")
    api_mode = llm_cfg.get("api_mode", "auto")

    archs = set(args.archs.split(","))

    log(f"IntrinsicImpl.inc: {inc_path}")
    log(f"KB output: {kb_path}")
    log(f"Embedding model: {embedding_model}")
    log(f"LLM model: {model}")
    log(f"Architectures: {archs}")

    # Create LLM client
    llm = LLMClient(
        base_url=llm_cfg.get("base_url", ""),
        api_key=llm_cfg.get("api_key", ""),
    )

    # Build KB
    result = build_kb(
        inc_path=inc_path,
        output_path=kb_path,
        llm_client=llm,
        embedding_model=embedding_model,
        archs=archs,
        model=model,
        api_mode=api_mode,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    log(f"Done. KB saved to: {result}")


if __name__ == "__main__":
    main()
