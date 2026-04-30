#!/usr/bin/env python3
"""Compare TSVC baseline vs IntOpt results.

Usage:
  python scripts/compare_tsvc.py \
    --baseline test/TSVC_2/intoptplus_results/tsvc_eval/baseline_o3.tsv \
    --intopt test/TSVC_2/intoptplus_results/tsvc_eval/intopt.tsv
"""

import argparse
import math
import sys


def load_tsv(path: str) -> dict:
    """Parse TSVC output TSV. Handles variable column formats:
    - "name\\ttime\\tchecksum"
    - "name\\tname\\ttime\\tchecksum" (TSVC prints name twice)
    """
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("name"):
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 2:
                continue

            name = parts[0]

            # Handle COMPILE_FAIL etc.
            if any(x in line for x in ("COMPILE_FAIL", "LINK_FAIL", "RUN_FAIL")):
                d[name] = {"time": -1, "checksum": "0"}
                continue

            # Find time and checksum: last two numeric fields
            nums = []
            for p in parts[1:]:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass

            if len(nums) >= 2:
                d[name] = {"time": nums[-2], "checksum": str(nums[-1])}
            elif len(nums) == 1:
                d[name] = {"time": nums[0], "checksum": "0"}

    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--intopt", required=True)
    p.add_argument("--output", default="", help="Write CSV report to file")
    args = p.parse_args()

    base = load_tsv(args.baseline)
    opt = load_tsv(args.intopt)

    print(f"Baseline: {len(base)} benchmarks")
    print(f"IntOpt:   {len(opt)} benchmarks")
    print()

    correct = wrong = faster = slower = same = compile_fail = 0
    speedups = []
    rows = []

    print(f"{'name':>10} {'O3_time':>10} {'opt_time':>10} {'speedup':>8} {'check':>10}")
    print("-" * 55)

    for name in sorted(set(base) & set(opt)):
        b = base[name]
        o = opt[name]

        if o["time"] < 0:
            compile_fail += 1
            rows.append((name, -1, -1, 0, "COMPILE_FAIL"))
            continue

        # Checksum comparison
        try:
            b_ck = float(b["checksum"])
            o_ck = float(o["checksum"])
            ck_ok = abs(b_ck - o_ck) < max(abs(b_ck) * 1e-4, 1e-6)
        except Exception:
            ck_ok = b["checksum"] == o["checksum"]

        if ck_ok:
            correct += 1
        else:
            wrong += 1

        if b["time"] > 0 and o["time"] > 0:
            sp = b["time"] / o["time"]
            speedups.append(sp)
            if sp > 1.05:
                faster += 1
            elif sp < 0.95:
                slower += 1
            else:
                same += 1
            ck_mark = "OK" if ck_ok else "WRONG"
            print(f"{name:>10} {b['time']:>10.4f} {o['time']:>10.4f} {sp:>7.3f}x {ck_mark:>10}")
            rows.append((name, b["time"], o["time"], sp, ck_mark))

    print()
    print(f"Correctness: {correct} OK, {wrong} WRONG, {compile_fail} compile_fail")
    print(f"Performance: {faster} faster, {same} same, {slower} slower")
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"Geometric mean speedup: {geo:.4f}x")

    if args.output:
        with open(args.output, "w") as f:
            f.write("name,o3_time,opt_time,speedup,checksum\n")
            for name, bt, ot, sp, ck in rows:
                f.write(f"{name},{bt:.4f},{ot:.4f},{sp:.4f},{ck}\n")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
