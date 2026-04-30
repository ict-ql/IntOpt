#!/usr/bin/env python3
"""Split TSVC benchmark into individual compilable units.

Each benchmark function becomes {name}.c containing:
  - #include "common.h" / "array_defs.h"
  - The benchmark function
  - A main() that calls init, runs the bench, prints time + checksum

These are compiled and linked with:
  - common.c (init, checksum, array helpers)
  - dummy.c (dummy function)
  - globals.c (array definitions, extracted from tsvc.c)

Usage:
  python scripts/split_tsvc.py --tsvc-dir test/TSVC_2/src --output-dir test/TSVC_2/split --compile
  python scripts/split_tsvc.py ... --compile --mem2reg   # also run opt -mem2reg on .ll
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from modules.utils import log


def extract_functions(tsvc_c: str) -> dict:
    """Extract each benchmark function from tsvc.c.
    Returns {func_name: func_body}."""
    text = Path(tsvc_c).read_text(encoding="utf-8")
    lines = text.splitlines()

    funcs = {}
    current_name = None
    current_lines = []
    brace_depth = 0
    in_func = False

    for line in lines:
        m = re.match(r'^real_t\s+(\w+)\s*\(', line)
        if m and not in_func:
            current_name = m.group(1)
            current_lines = [line]
            brace_depth = line.count('{') - line.count('}')
            in_func = brace_depth > 0
            continue

        if current_name is not None:
            current_lines.append(line)
            brace_depth += line.count('{') - line.count('}')
            if '{' in line:
                in_func = True
            if in_func and brace_depth == 0:
                funcs[current_name] = "\n".join(current_lines)
                current_name = None
                current_lines = []
                in_func = False

    return funcs


# Template for bench-only .c file (no main, just the function)
_BENCH_TEMPLATE = """\
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "common.h"
#include "array_defs.h"

{func_body}
"""

# Template for main wrapper .c file (calls the bench function)
_MAIN_TEMPLATE = """\
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "common.h"
#include "array_defs.h"

extern real_t {func_name}(struct args_t *);

int main() {{
    int *ip;
    real_t s1, s2;
    init(&ip, &s1, &s2);

    struct args_t args;
    args.arg_info = NULL;

    real_t checksum = {func_name}(&args);

    double elapsed = (args.t2.tv_sec - args.t1.tv_sec)
                   + (args.t2.tv_usec - args.t1.tv_usec) / 1000000.0;

    printf("{func_name}\\t%.6f\\t%.6f\\n", elapsed, checksum);
    return 0;
}}
"""

# globals.c: array definitions shared by all benchmarks
_GLOBALS_TEMPLATE = """\
#include "common.h"
#include "array_defs.h"

// Global array definitions (from tsvc.c)
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t flat_2d_array[LEN_2D*LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t x[LEN_1D];
__attribute__((aligned(ARRAY_ALIGNMENT))) real_t a[LEN_1D],b[LEN_1D],c[LEN_1D],d[LEN_1D],e[LEN_1D],
    aa[LEN_2D][LEN_2D],bb[LEN_2D][LEN_2D],cc[LEN_2D][LEN_2D],tt[LEN_2D][LEN_2D];
__attribute__((aligned(ARRAY_ALIGNMENT))) int indx[LEN_1D];
real_t* __restrict__ xx;
real_t* yy;
"""


def main():
    p = argparse.ArgumentParser(description="Split TSVC into individual benchmarks")
    p.add_argument("--tsvc-dir", default="test/TSVC_2/src")
    p.add_argument("--output-dir", default="test/TSVC_2/split")
    p.add_argument("--clang", default="/home/amax/yangz/Env/llvm-project/build/bin/clang")
    p.add_argument("--opt", default="/home/amax/yangz/Env/llvm-project/build/bin/opt")
    p.add_argument("--llc", default="/home/amax/yangz/Env/llvm-project/build/bin/llc")
    p.add_argument("--compile", action="store_true", help="Also compile to .ll and binary")
    p.add_argument("--mem2reg", action="store_true",
                   help="Run early cleanup passes on .ll to reduce prompt length")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    tsvc_c = os.path.join(args.tsvc_dir, "tsvc.c")
    out_dir = Path(args.output_dir)
    c_dir = out_dir / "c"
    ll_dir = out_dir / "ll"
    bin_dir = out_dir / "bin"
    c_dir.mkdir(parents=True, exist_ok=True)
    ll_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    skip = {"test", "f"}

    globals_file = c_dir / "globals.c"

    # Step 1: Extract functions (skip if .c files already exist)
    bench_files = list(c_dir.glob("*.c"))
    bench_files = [f for f in bench_files
                   if not f.name.startswith("globals") and not f.name.endswith("_main.c")]
    if bench_files:
        log(f"Found {len(bench_files)} existing bench .c files, skipping extraction")
        names = sorted(f.stem for f in bench_files if f.stem not in skip)
        if args.limit > 0:
            names = names[:args.limit]
    else:
        log("Extracting functions from tsvc.c ...")
        funcs = extract_functions(tsvc_c)
        log(f"Found {len(funcs)} functions")

        globals_file.write_text(_GLOBALS_TEMPLATE, encoding="utf-8")
        log(f"Wrote {globals_file}")

        count = 0
        names = []
        for name, body in sorted(funcs.items()):
            if name in skip:
                continue
            if args.limit > 0 and count >= args.limit:
                break

            bench_file = c_dir / f"{name}.c"
            bench_file.write_text(
                _BENCH_TEMPLATE.format(func_body=body), encoding="utf-8")

            main_file = c_dir / f"{name}_main.c"
            main_file.write_text(
                _MAIN_TEMPLATE.format(func_name=name), encoding="utf-8")

            names.append(name)
            count += 1

        log(f"Generated {count} bench + main .c pairs in {c_dir}")

    if not args.compile:
        log("Done. Use --compile to also build .ll and binaries.")
        return

    common_c = os.path.join(args.tsvc_dir, "common.c")
    dummy_c = os.path.join(args.tsvc_dir, "dummy.c")
    globals_c = str(globals_file)

    # Step 2: Compile bench .c → .ll
    #   Use -O3 -Xclang -disable-llvm-passes to get frontend-optimized IR
    #   without running LLVM's middle-end passes.
    log("Compiling bench functions to .ll ...")
    ok = fail = skip_count = 0
    for name in names:
        c_file = c_dir / f"{name}.c"
        ll_file = ll_dir / f"{name}.ll"

        if ll_file.exists() and ll_file.stat().st_size > 0:
            skip_count += 1
            continue

        r = subprocess.run(
            [args.clang, "-O3", "-Xclang", "-disable-llvm-passes",
             "-S", "-emit-llvm", "-Wno-everything",
             f"-I{args.tsvc_dir}", str(c_file), "-o", str(ll_file)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            fail += 1
            log(f"  FAIL .ll: {name}: {r.stderr[:150]}")
            continue

        # Optionally run early cleanup passes to reduce IR size
        if args.mem2reg:
            tmp_ll = ll_file.with_suffix(".early.ll")
            r2 = subprocess.run(
                [args.opt,
                 "-passes=mem2reg,instcombine,simplifycfg,early-cse,sroa",
                 "-S", str(ll_file), "-o", str(tmp_ll)],
                capture_output=True, text=True, timeout=30,
            )
            if r2.returncode == 0:
                tmp_ll.replace(ll_file)
            else:
                tmp_ll.unlink(missing_ok=True)
                log(f"  WARN early passes: {name}: {r2.stderr[:100]}")

        ok += 1

    log(f"Compiled .ll: ok={ok}  skip={skip_count}  fail={fail}")


    # Step 3: Build O3 baseline binaries
    #   Match the IntOpt evaluation pipeline: opt -O3 → llc → link
    #   so the baseline uses the same backend path as optimized code.
    log("Building O3 baseline binaries ...")
    ok = fail = 0
    for name in names:
        c_file = c_dir / f"{name}.c"
        main_c = c_dir / f"{name}_main.c"
        bin_file = bin_dir / name

        if bin_file.exists():
            ok += 1
            continue

        # clang → unoptimised .ll (bench function only)
        raw_ll = bin_dir / f"{name}.raw.ll"
        r = subprocess.run(
            [args.clang, "-O3", "-Xclang", "-disable-llvm-passes",
             "-S", "-emit-llvm", "-Wno-everything",
             f"-I{args.tsvc_dir}", str(c_file), "-o", str(raw_ll)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            fail += 1
            log(f"  FAIL raw .ll: {name}: {r.stderr[:150]}")
            raw_ll.unlink(missing_ok=True)
            continue

        # opt -O3
        o3_ll = bin_dir / f"{name}.o3.ll"
        r = subprocess.run(
            [args.opt, "-O3", "-S", str(raw_ll), "-o", str(o3_ll)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            fail += 1
            log(f"  FAIL opt -O3: {name}: {r.stderr[:150]}")
            raw_ll.unlink(missing_ok=True)
            o3_ll.unlink(missing_ok=True)
            continue

        # llc → .s
        asm_file = bin_dir / f"{name}.s"
        r = subprocess.run(
            [args.llc, "-O3", "--relocation-model=pic",
             str(o3_ll), "-o", str(asm_file)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            fail += 1
            log(f"  FAIL llc: {name}: {r.stderr[:150]}")
            for tmp in [raw_ll, o3_ll, asm_file]:
                tmp.unlink(missing_ok=True)
            continue

        # link: .s + main + common + dummy + globals → binary
        r = subprocess.run(
            [args.clang, "-O3", "-Wno-everything",
             f"-I{args.tsvc_dir}",
             str(asm_file), str(main_c), common_c, dummy_c, globals_c,
             "-lm", "-o", str(bin_file)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            fail += 1
            log(f"  FAIL link: {name}: {r.stderr[:150]}")
        else:
            ok += 1

        # Clean up intermediates
        for tmp in [raw_ll, o3_ll, asm_file]:
            tmp.unlink(missing_ok=True)

    log(f"Built baseline binaries: ok={ok}  fail={fail}")
    log(f"Output: .ll={ll_dir}  bin={bin_dir}")


if __name__ == "__main__":
    main()
