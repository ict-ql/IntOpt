"""Performance testing: given diff-tested (original, optimised) IR pairs,
generate benchmark harnesses, compile them, and measure speedup.

Pipeline:
  1. collect_corpus   — run diff-test fuzzing bins briefly to produce diverse inputs
  2. generate_bench   — transform fuzz.cc → bench.cc (timing instrumentation)
  3. build_binaries   — llc + clang++ (no sanitizers) → bench binary
  4. run_benchmarks   — execute each binary on corpus, collect timing results
"""

import csv
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modules.utils import log


# ======================================================================
# Corpus collection
# ======================================================================

def _is_executable(p: Path) -> bool:
    try:
        st = p.stat()
    except FileNotFoundError:
        return False
    return p.is_file() and os.access(str(p), os.X_OK) and st.st_size > 0


def _collect_one(
    bin_path: Path,
    corpus_dir: Path,
    max_total_time: int,
    timeout: int,
    force: bool,
) -> Tuple[str, int, str]:
    """Run a single fuzzing binary to generate corpus files.
    Returns (stem, returncode, error_or_empty)."""
    stem = bin_path.parent.name

    if corpus_dir.exists() and any(corpus_dir.iterdir()) and not force:
        return stem, 0, ""

    corpus_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(bin_path), str(corpus_dir), f"-max_total_time={max_total_time}"]

    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout, cwd=str(bin_path.parent),
        )
        return stem, p.returncode, ""
    except subprocess.TimeoutExpired:
        return stem, 124, "timeout"
    except Exception as e:
        return stem, 127, str(e)


# ======================================================================
# Harness transformation: fuzz.cc → bench.cc
# ======================================================================

def _extract_pairs(src: str) -> List[Tuple[str, str]]:
    """Extract (base_func, opt_func) pairs from fuzz harness source.

    Strategy A: identifier-based — find all IDENT( tokens, pair foo / foo_opt.
    Strategy B: asm-alias-based — match asm("SYM") declarations, pair by symbol.
    """
    cand_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(", re.M)
    blacklist = {
        "if", "for", "while", "switch", "return", "sizeof",
        "reinterpret_cast", "static_cast", "const_cast", "dynamic_cast",
        "catch", "throw", "new", "delete", "assert", "abort", "exit",
    }
    cands = {n for n in cand_re.findall(src) if n not in blacklist}

    pairs: set = set()
    for name in cands:
        if name.endswith("_opt"):
            base = name[:-4]
            if base in cands:
                pairs.add((base, name))

    # asm-alias fallback
    asm_decl = re.compile(
        r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*asm\s*\(\s*\"([^\"]+)\"\s*\)\s*;",
        re.M | re.S,
    )
    sym_to_func: Dict[str, str] = {}
    for func, sym in asm_decl.findall(src):
        sym_to_func[sym] = func

    for sym, func in sym_to_func.items():
        if sym.endswith("_opt"):
            base_sym = sym[:-4]
            if base_sym in sym_to_func:
                pairs.add((sym_to_func[base_sym], func))

    return sorted(pairs)


def _rename_fuzzer_entry(src: str) -> str:
    """Rename LLVMFuzzerTestOneInput → decode_input (static)."""
    return re.sub(
        r'extern\s+"C"\s+int\s+LLVMFuzzerTestOneInput\s*\(',
        "static int decode_input(",
        src,
    )


_TIMING_INFRA = r"""
#include <time.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t g_t_baseline_ns = 0;
static uint64_t g_t_opt_ns      = 0;
static uint64_t g_n_baseline    = 0;
static uint64_t g_n_opt         = 0;

// Prevent DCE: accumulate address of any lhs into a volatile sink
static volatile uintptr_t g_sink = 0;
"""

_MAIN_CODE = r"""
static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open: " << path << "\n";
        exit(1);
    }
    return std::vector<uint8_t>(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>()
    );
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <corpus_file> [iters]\n";
        return 1;
    }
    uint64_t iters = (argc >= 3) ? strtoull(argv[2], nullptr, 10) : 1000000ull;

    auto data = read_file(argv[1]);
    if (data.empty()) return 0;

    // Warmup (not counted)
    uint64_t sb=g_t_baseline_ns, so=g_t_opt_ns, nb=g_n_baseline, no=g_n_opt;
    for (int i = 0; i < 1000; i++) (void)decode_input(data.data(), data.size());
    g_t_baseline_ns=sb; g_t_opt_ns=so; g_n_baseline=nb; g_n_opt=no;

    for (uint64_t i = 0; i < iters; i++) {
        (void)decode_input(data.data(), data.size());
    }

    double avg_b = g_n_baseline ? (double)g_t_baseline_ns / (double)g_n_baseline : 0.0;
    double avg_o = g_n_opt      ? (double)g_t_opt_ns      / (double)g_n_opt      : 0.0;

    std::cout << "iters=" << iters << "\n";
    std::cout << "baseline calls=" << g_n_baseline << " avg(ns/call)=" << avg_b << "\n";
    std::cout << "opt      calls=" << g_n_opt      << " avg(ns/call)=" << avg_o << "\n";
    if (avg_o > 0) std::cout << "speedup=" << (avg_b / avg_o) << "x\n";
    std::cout << "(ignore) sink=" << (unsigned long long)g_sink << "\n";
    return 0;
}
"""


def _inject_timing_infra(src: str) -> str:
    """Insert timing infrastructure after the last #include line."""
    includes = list(re.finditer(r"^\s*#include[^\n]*\n", src, flags=re.M))
    pos = includes[-1].end() if includes else 0
    return src[:pos] + _TIMING_INFRA + src[pos:]


def _is_decl_like(line: str, fname: str) -> bool:
    """Heuristic: if the char before fname is alphanumeric/_, it's a declaration."""
    idx = line.find(fname)
    if idx < 0:
        return False
    j = idx - 1
    while j >= 0 and line[j].isspace():
        j -= 1
    if j < 0:
        return False
    return line[j].isalnum() or line[j] == "_"


def _instrument_calls(src: str, base: str, opt: str) -> str:
    """Wrap single-line call statements to base/opt with timing code."""

    def _stmt_re(fname: str):
        return re.compile(
            rf'(?m)^(?P<indent>\s*)(?P<line>(?!\s*extern\s+"C").*?\b'
            rf"{re.escape(fname)}"
            rf"\s*\([^;]*\)\s*;)\s*$",
        )

    def _extract_lhs(line: str, fname: str) -> Optional[str]:
        m = re.search(rf"\b([A-Za-z_]\w*)\s*=\s*{re.escape(fname)}\s*\(", line)
        return m.group(1) if m else None

    def _make_repl(which: str, fname: str):
        t0 = "__t0" if which == "base" else "__t2"
        t1 = "__t1" if which == "base" else "__t3"
        acc_t = "g_t_baseline_ns" if which == "base" else "g_t_opt_ns"
        acc_n = "g_n_baseline" if which == "base" else "g_n_opt"

        def _do(m: re.Match) -> str:
            indent = m.group("indent")
            line = m.group("line")
            if _is_decl_like(line, fname):
                return f"{indent}{line}"
            lhs = _extract_lhs(line, fname)
            sink = ""
            if lhs:
                sink = f"{indent}g_sink ^= (uintptr_t)(const void*)&{lhs};\n"
            return (
                f"{indent}uint64_t {t0} = now_ns();\n"
                f"{indent}{line}\n"
                f"{indent}uint64_t {t1} = now_ns();\n"
                f"{indent}{acc_t} += ({t1} - {t0});\n"
                f"{indent}{acc_n}++;\n"
                f"{sink}".rstrip("\n")
            )
        return _do

    src = _stmt_re(base).sub(_make_repl("base", base), src)
    src = _stmt_re(opt).sub(_make_repl("opt", opt), src)
    return src


def _transform_fuzz_to_bench(fuzz_src: str) -> Optional[str]:
    """Convert a fuzz.cc harness into a bench.cc with timing instrumentation.
    Returns None if no (base, opt) pair is found."""
    pairs = _extract_pairs(fuzz_src)
    if len(pairs) != 1:
        return None

    src = _rename_fuzzer_entry(fuzz_src)
    src = _inject_timing_infra(src)
    base, opt = pairs[0]
    src = _instrument_calls(src, base, opt)
    src += "\n" + _MAIN_CODE
    return src


# ======================================================================
# Build helpers
# ======================================================================

def _run_cmd(cmd: List[str], timeout: int = 60) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        return p.returncode, p.stdout or ""
    except subprocess.TimeoutExpired:
        return 124, "[timeout]"
    except Exception as e:
        return 127, str(e)


def _rename_main_in_ir(ir_text: str) -> str:
    """If both @main and @main_opt exist, rename to @main_1 / @main_1_opt
    so they don't conflict with the bench binary's main()."""
    main_pat = re.compile(r"^\s*define\b[^\n]*\s@main\s*\(", re.MULTILINE)
    opt_pat = re.compile(r"^\s*define\b[^\n]*\s@main_opt\s*\(", re.MULTILINE)
    if not (main_pat.search(ir_text) and opt_pat.search(ir_text)):
        return ir_text
    ir_text = re.sub(
        r"^(\s*define\b[^\n]*\s)@main_opt(\s*\()", r"\1@main_1_opt\2",
        ir_text, flags=re.MULTILINE,
    )
    ir_text = re.sub(
        r"^(\s*define\b[^\n]*\s)@main(\s*\()", r"\1@main_1\2",
        ir_text, flags=re.MULTILINE,
    )
    return ir_text


def _build_one_bench(
    bench_cc: Path,
    combined_ll: Path,
    out_bin: Path,
    llc: str,
    clangxx: str,
    overwrite: bool,
) -> Tuple[Path, Optional[Path], Optional[str]]:
    """Compile combined.ll → .s via llc, then link with bench.cc → binary.
    No sanitizers (performance measurement)."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    if out_bin.exists() and not overwrite:
        return bench_cc, out_bin, None

    # Rename @main/@main_opt in IR if needed
    ir_text = combined_ll.read_text(encoding="utf-8")
    new_ir = _rename_main_in_ir(ir_text)
    if new_ir != ir_text:
        combined_ll.write_text(new_ir, encoding="utf-8")

    asm = out_bin.with_suffix(".s")

    rc, out = _run_cmd([llc, "-O3", str(combined_ll), "-o", str(asm)])
    if rc != 0:
        return bench_cc, None, f"llc failed (rc={rc}): {out[:500]}"

    # Link without sanitizers for accurate timing
    rc, out = _run_cmd([
        clangxx, str(bench_cc), str(asm),
        "-O1", "-g", "-o", str(out_bin),
    ])
    if rc != 0:
        return bench_cc, None, f"clang++ link failed (rc={rc}): {out[:500]}"

    return bench_cc, out_bin, None


# ======================================================================
# Benchmark runner
# ======================================================================

def _parse_bench_output(output: str) -> Dict[str, float]:
    """Parse timing output from a bench binary."""
    result: Dict[str, float] = {}
    for line in output.splitlines():
        if "avg(ns/call)=" in line:
            m = re.search(r"avg\(ns/call\)=([0-9.e+\-]+)", line)
            if m:
                val = float(m.group(1))
                if "baseline" in line:
                    result["baseline_ns"] = val
                elif "opt" in line:
                    result["opt_ns"] = val
        if "speedup=" in line:
            m = re.search(r"speedup=([0-9.e+\-]+)", line)
            if m:
                result["speedup"] = float(m.group(1))
    return result


def _run_one_bench(
    bin_path: Path,
    corpus_file: Path,
    iters: int,
    timeout: int,
) -> Tuple[str, Dict[str, float], str]:
    """Run a bench binary on a single corpus file.
    Returns (status, metrics_dict, raw_output)."""
    cmd = [str(bin_path), str(corpus_file), str(iters)]
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        output = p.stdout or ""
        if p.returncode != 0:
            return "FAIL", {}, output[-2000:]
        metrics = _parse_bench_output(output)
        return "PASS", metrics, output[-2000:]
    except subprocess.TimeoutExpired:
        return "TIMEOUT", {}, ""
    except Exception as e:
        return "ERROR", {}, str(e)



# ======================================================================
# Main class
# ======================================================================

class PerfTester:
    """Orchestrates performance benchmarking after diff testing passes."""

    def __init__(self, llc: str, clangxx: str):
        self.llc = llc
        self.clangxx = clangxx

    # ------------------------------------------------------------------
    # Step 1: collect corpus from diff-test fuzzing binaries
    # ------------------------------------------------------------------

    def collect_corpus(
        self,
        fuzz_bin_dir: str,
        corpus_dir: str,
        max_total_time: int = 10,
        timeout: int = 120,
        workers: int = 0,
        force: bool = False,
    ) -> str:
        """Run each *_fuzz binary briefly to generate diverse corpus files.
        Returns the corpus output directory."""
        bin_p = Path(fuzz_bin_dir)
        corpus_p = Path(corpus_dir)
        corpus_p.mkdir(parents=True, exist_ok=True)

        bins = sorted(
            b for b in bin_p.rglob("*_fuzz") if _is_executable(b)
        )
        if not bins:
            raise SystemExit(f"No executable *_fuzz binaries under {bin_p}")

        if workers <= 0:
            workers = max(1, (os.cpu_count() or 8) // 2)

        log(f"Collecting corpus from {len(bins)} binaries "
            f"(time={max_total_time}s, workers={workers}) ...")

        ok = fail = skip = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _collect_one, b, corpus_p / b.parent.name,
                    max_total_time, timeout, force,
                ): b
                for b in bins
            }
            for fut in as_completed(futs):
                stem, rc, err = fut.result()
                if rc == 0:
                    if err == "":
                        ok += 1
                    else:
                        skip += 1
                else:
                    fail += 1
                    log(f"  WARN: corpus collection failed for {stem}: {err}")

        log(f"Corpus done. ok={ok}  skip={skip}  fail={fail}")
        return str(corpus_p)

    # ------------------------------------------------------------------
    # Step 2: generate bench harnesses from fuzz harnesses
    # ------------------------------------------------------------------

    def generate_bench_harnesses(
        self,
        harness_dir: str,
        out_dir: str,
        overwrite: bool = False,
    ) -> str:
        """Transform each *.fuzz.cc → *.bench.cc with timing instrumentation.
        Returns the bench harness output directory."""
        harness_p = Path(harness_dir)
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        fuzz_files = sorted(harness_p.glob("*.fuzz.cc"))
        if not fuzz_files:
            raise SystemExit(f"No *.fuzz.cc found in {harness_p}")

        ok = fail = 0
        for fuzz_cc in fuzz_files:
            stem = fuzz_cc.name[: -len(".fuzz.cc")]
            bench_cc = out_p / f"{stem}.bench.cc"
            if bench_cc.exists() and not overwrite:
                ok += 1
                continue

            src = fuzz_cc.read_text(encoding="utf-8", errors="ignore")
            bench_src = _transform_fuzz_to_bench(src)
            if bench_src is None:
                fail += 1
                log(f"  WARN: no valid (base, opt) pair in {fuzz_cc.name}, skipping")
                continue

            bench_cc.write_text(bench_src, encoding="utf-8")
            ok += 1

        log(f"Generated {ok} bench harnesses ({fail} skipped) in {out_p}")
        return str(out_p)

    # ------------------------------------------------------------------
    # Step 3: build bench binaries
    # ------------------------------------------------------------------

    def build_binaries(
        self,
        combined_dir: str,
        bench_dir: str,
        out_dir: str,
        workers: int = 16,
        overwrite: bool = False,
    ) -> str:
        """Compile each (combined.ll, bench.cc) pair into a bench binary.
        Returns the binary output directory."""
        combined_p = Path(combined_dir)
        bench_p = Path(bench_dir)
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        bench_files = sorted(bench_p.glob("*.bench.cc"))
        if not bench_files:
            raise SystemExit(f"No *.bench.cc found in {bench_p}")

        jobs = []
        for bench_cc in bench_files:
            stem = bench_cc.name[: -len(".bench.cc")]
            ll_file = combined_p / stem / "combined.ll"
            if not ll_file.exists():
                log(f"  WARN: no combined.ll for {stem}, skipping")
                continue
            out_bin = out_p / stem / f"{stem}_bench"
            jobs.append((bench_cc, ll_file, out_bin))

        if not jobs:
            raise SystemExit("No buildable (bench.cc, combined.ll) pairs found")

        log(f"Building {len(jobs)} bench binaries (workers={workers}) ...")

        ok = fail = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [
                ex.submit(
                    _build_one_bench, bcc, ll, obin,
                    self.llc, self.clangxx, overwrite,
                )
                for bcc, ll, obin in jobs
            ]
            total = len(futs)
            for i, fut in enumerate(as_completed(futs), start=1):
                bcc, obin, err = fut.result()
                if err is None and obin is not None:
                    ok += 1
                    if i % 20 == 0 or i == total:
                        log(f"  [{i}/{total}] OK: {obin}")
                else:
                    fail += 1
                    log(f"  [{i}/{total}] FAIL: {bcc.name} — {err}")

        log(f"Bench build done. ok={ok}  fail={fail}")
        return str(out_p)

    # ------------------------------------------------------------------
    # Step 4: run benchmarks and collect results
    # ------------------------------------------------------------------

    def run_benchmarks(
        self,
        bench_bin_dir: str,
        corpus_dir: str,
        iters: int = 1000000,
        timeout: int = 300,
        workers: int = 1,
    ) -> Dict[str, dict]:
        """Run each bench binary on its corpus and collect timing results.

        *workers* controls parallelism: 1 = sequential (default, most
        accurate timing), >1 = parallel (faster but may introduce noise
        from CPU contention).

        Returns a dict mapping stem → {status, baseline_ns, opt_ns, speedup}."""
        bin_p = Path(bench_bin_dir)
        corpus_p = Path(corpus_dir)

        bins = sorted(bin_p.rglob("*_bench"))
        bins = [b for b in bins if b.is_file() and _is_executable(b)]
        if not bins:
            raise SystemExit(f"No bench binaries found under {bin_p}")

        log(f"Running {len(bins)} benchmarks "
            f"(iters={iters}, workers={workers}) ...")

        # Build (stem, bin_path, [corpus_files]) jobs
        jobs: List[Tuple[str, Path, List[Path]]] = []
        skip_results: Dict[str, dict] = {}
        for b in bins:
            stem = b.parent.name
            stem_corpus = corpus_p / stem
            if not stem_corpus.is_dir():
                skip_results[stem] = {"status": "NO_CORPUS"}
                continue
            corpus_files = sorted(
                (f for f in stem_corpus.iterdir() if f.is_file()),
                key=lambda f: f.stat().st_size,
                reverse=True,
            )
            if not corpus_files:
                skip_results[stem] = {"status": "NO_CORPUS"}
                continue
            jobs.append((stem, b, corpus_files))

        for stem in skip_results:
            log(f"  {stem}: no corpus, skipping")

        results: Dict[str, dict] = dict(skip_results)

        def _bench_all_corpus(
            stem: str, b: Path, cfiles: List[Path],
        ) -> Tuple[str, dict]:
            """Run bench binary on every corpus file, return aggregated result."""
            per_corpus: List[dict] = []
            for cf in cfiles:
                status, metrics, _ = _run_one_bench(b, cf, iters, timeout)
                entry = {"corpus": cf.name, "status": status, **metrics}
                per_corpus.append(entry)

            passed = [e for e in per_corpus if e["status"] == "PASS"]
            if not passed:
                return stem, {
                    "status": "FAIL",
                    "per_corpus": per_corpus,
                }

            avg_baseline = (
                sum(e.get("baseline_ns", 0) for e in passed) / len(passed)
            )
            avg_opt = (
                sum(e.get("opt_ns", 0) for e in passed) / len(passed)
            )
            avg_speedup = (
                sum(e.get("speedup", 0) for e in passed) / len(passed)
            )
            return stem, {
                "status": "PASS",
                "baseline_ns": avg_baseline,
                "opt_ns": avg_opt,
                "speedup": avg_speedup,
                "n_corpus": len(passed),
                "per_corpus": per_corpus,
            }

        if workers <= 1:
            for i, (stem, b, cfiles) in enumerate(jobs, start=1):
                stem, entry = _bench_all_corpus(stem, b, cfiles)
                results[stem] = entry
                if entry["status"] == "PASS":
                    log(f"  [{i}/{len(jobs)}] {stem}: "
                        f"speedup={entry['speedup']:.3f}x "
                        f"(avg over {entry['n_corpus']} corpus files)")
                else:
                    log(f"  [{i}/{len(jobs)}] {stem}: {entry['status']}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(_bench_all_corpus, stem, b, cfiles): stem
                    for stem, b, cfiles in jobs
                }
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    stem, entry = fut.result()
                    results[stem] = entry
                    if entry["status"] == "PASS":
                        log(f"  [{done}/{len(jobs)}] {stem}: "
                            f"speedup={entry['speedup']:.3f}x "
                            f"(avg over {entry['n_corpus']} corpus files)")
                    else:
                        log(f"  [{done}/{len(jobs)}] {stem}: {entry['status']}")

        n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
        speedups = [
            r["speedup"] for r in results.values()
            if r.get("speedup") is not None
        ]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        log(f"Benchmark done. measured={n_pass}/{len(bins)}  "
            f"avg_speedup={avg_speedup:.3f}x")
        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full(
        self,
        combined_dir: str,
        harness_dir: str,
        fuzz_bin_dir: str,
        work_dir: str,
        corpus_dir: str = "",
        corpus_time: int = 10,
        corpus_workers: int = 0,
        build_workers: int = 16,
        bench_iters: int = 1000000,
        bench_timeout: int = 300,
        bench_workers: int = 1,
    ) -> Dict[str, dict]:
        """End-to-end performance testing.  Returns benchmark results dict.

        If *corpus_dir* is provided and non-empty, skip corpus generation
        and use the given directory directly."""
        work = Path(work_dir)
        bench_harness_dir = work / "bench_harness"
        bench_bin_dir = work / "bench_bins"

        # Use provided corpus or generate one
        if corpus_dir and Path(corpus_dir).is_dir():
            log(f"Using existing corpus: {corpus_dir}")
        else:
            corpus_dir = str(work / "corpus")
            self.collect_corpus(
                fuzz_bin_dir, corpus_dir,
                max_total_time=corpus_time, workers=corpus_workers,
            )

        # Check if harness already exist
        if not (bench_harness_dir.is_dir() and any(bench_harness_dir.rglob("*.bench.cc"))):
            self.generate_bench_harnesses(harness_dir, str(bench_harness_dir))
        else:
            log(f"Find existing harnesses: {bench_harness_dir}")
        # Check if bins already exist
        if not (bench_bin_dir.is_dir() and any(bench_bin_dir.rglob("*_bench"))):
            self.build_binaries(
                combined_dir, bench_harness_dir, str(bench_bin_dir),
                workers=build_workers,
            )
        else:
            log(f"Find existing bench binaries: {bench_bin_dir}")

        return self.run_benchmarks(
            str(bench_bin_dir), corpus_dir,
            iters=bench_iters, timeout=bench_timeout,
            workers=bench_workers,
        )
