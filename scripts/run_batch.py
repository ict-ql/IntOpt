#!/usr/bin/env python3
"""Run the optimisation pipeline on a directory of LLVM IR files.

Usage:
    python run_batch.py --input /path/to/data_dir --output /path/to/output_dir
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch-optimise LLVM IR files")
    parser.add_argument("--input", required=True, help="Input directory with *.ll files")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    if not Path(args.input).is_dir():
        print(f"Error: input directory not found: {args.input}")
        return

    Path(args.output).mkdir(parents=True, exist_ok=True)

    cmd = (f"python ../src/main.py "
           f"--mode batch --input {args.input} --output {args.output}")
    print(f"Running: {cmd}\n")
    os.system(cmd)
    print(f"\nDone. Results in: {args.output}")


if __name__ == "__main__":
    main()
