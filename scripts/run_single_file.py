#!/usr/bin/env python3
"""Run the optimisation pipeline on a single LLVM IR file.

Usage:
    python run_single_file.py --input /path/to/input.ll --output /path/to/output_dir
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Optimise a single LLVM IR file")
    parser.add_argument("--input", required=True, help="Input .ll file")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}")
        return

    Path(args.output).mkdir(parents=True, exist_ok=True)

    cmd = (f"python ../src/main.py "
           f"--mode single --input {args.input} --output {args.output}")
    print(f"Running: {cmd}\n")
    os.system(cmd)
    print(f"\nDone. Results in: {args.output}")


if __name__ == "__main__":
    main()
