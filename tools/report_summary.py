#!/usr/bin/env python3
"""Report summary statistics for a given experiment directory.

Usage:
    python tools/report_summary.py /home/amax/yangz/intop/test/taco/intopt++gpt4.pp
"""
import csv
import os
import sys


def load_csv(path):
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    base = sys.argv[1].rstrip("/")
    verify_path = os.path.join(base, "res/verify/verify_report.csv")
    alive2_path = os.path.join(base, "res/verify/alive2_report.csv")
    perf_path = os.path.join(base, "res/perf_test/perf_report.csv")

    verify_rows = load_csv(verify_path)
    alive2_rows = load_csv(alive2_path)
    perf_rows = load_csv(perf_path)

    print(f"Experiment: {base}")
    print("=" * 60)

    # --- alive2 ---
    if alive2_rows:
        total = len(alive2_rows)
        passed = sum(1 for r in alive2_rows if r["status"] == "PASS")
        print(f"\n[Alive2]  PASS: {passed}/{total}  ({passed/total*100:.1f}%)")
    else:
        print("\n[Alive2]  No data")

    # --- verify (alive2 + diff_test) ---
    if verify_rows:
        total = len(verify_rows)
        passed = sum(1 for r in verify_rows if r["status"] == "PASS")
        print(f"[Verify]  PASS: {passed}/{total}  ({passed/total*100:.1f}%)")
    else:
        print("[Verify]  No data")

    # --- perf ---
    if perf_rows:
        pass_rows = [r for r in perf_rows if r["status"] == "PASS"]
        speedups = [float(r["speedup"]) for r in pass_rows]
        n = len(speedups)
        gt1_1 = sum(1 for s in speedups if s > 1.1)
        gt1_5 = sum(1 for s in speedups if s > 1.5)
        gt2 = sum(1 for s in speedups if s > 2.0)
        print(f"\n[Perf]    PASS cases: {n}")
        print(f"          Speedup > 1.1 : {gt1_1}")
        print(f"          Speedup > 1.5 : {gt1_5}")
        print(f"          Speedup > 2.0 : {gt2}")
    else:
        print("\n[Perf]    No data")


if __name__ == "__main__":
    main()
