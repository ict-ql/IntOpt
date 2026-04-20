#!/usr/bin/env python3
"""Quick LLM query tool.

Usage:
  # Interactive (type prompt, Ctrl+D to send):
  python scripts/ask_llm.py --config config/config.yaml

  # From a file:
  python scripts/ask_llm.py --config config/config.yaml --file prompt.txt

  # Inline:
  python scripts/ask_llm.py --config config/config.yaml -p "What is llvm.fma?"

  # Override model:
  python scripts/ask_llm.py --config config/config.yaml --model gpt-5 -p "hello"
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.llm_client import LLMClient


def main():
    p = argparse.ArgumentParser(description="Quick LLM query")
    p.add_argument("--config", default="../config/config.yaml")
    p.add_argument("-p", "--prompt", default="", help="Inline prompt text")
    p.add_argument("--file", default="", help="Read prompt from file")
    p.add_argument("--model", default="", help="Override model name")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    llm_cfg = cfg.get("llm", {})

    client = LLMClient(
        base_url=llm_cfg.get("base_url", ""),
        api_key=llm_cfg.get("api_key", ""),
    )

    model = args.model or llm_cfg.get("llm_model", "gpt-5")
    api_mode = llm_cfg.get("api_mode", "auto")

    # Get prompt
    if args.file:
        prompt = open(args.file, "r", encoding="utf-8").read()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Enter prompt (Ctrl+D to send):", file=sys.stderr)
        prompt = sys.stdin.read()

    if not prompt.strip():
        sys.exit("Empty prompt")

    print(f"[Model: {model}  Tokens: {args.max_tokens}]", file=sys.stderr)

    response = client._call_with_retry(
        prompt_text=prompt,
        model=model,
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
        truncation="auto",
        store=False,
        api_mode=api_mode,
        max_retries=2,
        base_backoff=1.0,
    )

    print(response)


if __name__ == "__main__":
    main()
