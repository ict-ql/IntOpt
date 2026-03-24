"""Correctness verification for optimised IR.

Provides two verification strategies:
  1. Alive2 (alive-tv) — formal translation validation
  2. Differential testing — libFuzzer-based runtime comparison

Typical flow: run alive2 first (fast, formal); for cases that fail or
timeout, fall back to differential testing (slower, empirical)."""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modules.llm_client import LLMClient
from modules.utils import log


# ======================================================================
# IR manipulation helpers
# ======================================================================

# Robust define detector: handles tabs, BOM, leading whitespace, etc.
_DEFINE_RE = re.compile(r"^\s*\ufeff?define\b")

# Strip <code ...> / </code> tags that LLMs sometimes emit
_CODE_TAG_RE = re.compile(r"</?code\b[^>]*>", flags=re.IGNORECASE)


def _clean_llm_artifacts(text: str) -> str:
    """Remove common LLM wrapper artifacts from IR text."""
    # Unescape literal \\n, drop BOM, NBSP → space, zero-width chars
    text = text.replace("\\n", "\n")
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    for zw in ("\u200b", "\u200c", "\u200d", "\u2060"):
        text = text.replace(zw, "")
    # Remove control chars except \n \r \t
    text = "".join(
        ch for ch in text
        if ord(ch) >= 32 or ch in ("\n", "\r", "\t")
    )
    # Remove <code> tags
    text = _CODE_TAG_RE.sub("", text)
    # Remove common wrappers
    for tag in ("[INST]", "[/INST]", "```llvm", "```", "<s>", "</s>"):
        text = text.replace(tag, "")
    # Remove preamble lines
    text = re.sub(
        r"^\s*Opt IR\s*:\s*", "", text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return text.strip()


def _remove_metadata(text: str) -> str:
    """Strip LLVM metadata lines and inline metadata annotations."""
    tail_pat = re.compile(r"(,\s*)?![\w\.]+\s+(?:!\d+|!\{.*?\})")
    cleaned = []
    for line in text.splitlines():
        if line.strip().startswith("!"):
            continue
        if "!" in line:
            line = tail_pat.sub("", line).rstrip()
            if line.endswith(","):
                line = line[:-1]
        cleaned.append(line)
    return "\n".join(cleaned)


def _parse_ir_structure(
    text: str,
) -> Tuple[List[str], List[str]]:
    """Split IR into context lines (module-level) and function definition
    blocks using brace-counting.

    Returns (context_lines, definitions) where each definition is the
    full text of one ``define ... { ... }`` block."""
    lines = text.splitlines()
    context_lines: List[str] = []
    definitions: List[str] = []
    current_func: List[str] = []
    in_function = False
    brace_count = 0

    for line in lines:
        line = line.lstrip("\ufeff")
        if not line.strip():
            continue

        if _DEFINE_RE.match(line):
            in_function = True
            brace_count = 0

        if in_function:
            current_func.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0:
                in_function = False
                definitions.append("\n".join(current_func))
                current_func = []
        else:
            context_lines.append(line)

    return context_lines, definitions


def _extract_func_name(func_body: str) -> Optional[str]:
    """Extract the function name from a ``define`` block header.

    Handles both plain names (``@foo``) and quoted names (``@"bar"``)."""
    m = re.search(r'define\s+.*?@(?:"([^"]+)"|([\w\.\$\-]+))', func_body)
    if m:
        return m.group(1) or m.group(2)
    return None


def build_combined_ir(original_ir: str, optimized_ir: str) -> str:
    """Merge original and optimised IR into a single module.

    Approach (following alive2-icml.py):
      1. Clean LLM artifacts and strip metadata from both IRs.
      2. Parse each into context lines + function definitions.
      3. Take the first function from each side.
      4. If names collide, rename the target's to ``<name>_opt``
         (only inside its own define header and internal calls).
      5. If either function is ``@main``, rename to ``@main_1`` /
         ``@main_1_opt`` so the fuzzer entry point is not shadowed.
      6. Combine using ONLY source context (avoids duplicate globals /
         attributes) plus both function bodies.
    """
    src_clean = _remove_metadata(_clean_llm_artifacts(original_ir))
    tgt_clean = _remove_metadata(_clean_llm_artifacts(optimized_ir))

    src_ctx, src_defs = _parse_ir_structure(src_clean)
    _, tgt_defs = _parse_ir_structure(tgt_clean)

    if not src_defs:
        log("  WARN: no function definition found in original IR")
        return original_ir
    if not tgt_defs:
        log("  WARN: no function definition found in optimised IR")
        return original_ir

    src_func = src_defs[0]
    tgt_func = tgt_defs[0]
    src_name = _extract_func_name(src_func)
    tgt_name = _extract_func_name(tgt_func)

    if not src_name or not tgt_name:
        log("  WARN: could not extract function name from IR")
        return original_ir

    # Rename target function if names collide
    final_tgt_name = tgt_name
    if src_name == tgt_name:
        final_tgt_name = f"{tgt_name}_opt"
        escaped = re.escape(tgt_name)
        # Rename in define header
        tgt_func = re.sub(
            rf"@{escaped}(\(|\s)", f"@{final_tgt_name}\\1",
            tgt_func, count=1,
        )
        # Rename in internal call sites
        tgt_func = re.sub(
            rf"\bcall\s+(.*?)@{escaped}(\(|\s)",
            rf"call \1@{final_tgt_name}\2",
            tgt_func,
        )

    # If @main is involved, rename to @main_1 / @main_1_opt to avoid
    # conflicting with the fuzzer entry point
    if src_name == "main":
        src_func = re.sub(r"@main(\(|\s)", r"@main_1\1", src_func)
        src_name = "main_1"
    if final_tgt_name == "main_opt":
        tgt_func = re.sub(r"@main_opt(\(|\s)", r"@main_1_opt\1", tgt_func)
        final_tgt_name = "main_1_opt"
    elif final_tgt_name == "main":
        tgt_func = re.sub(r"@main(\(|\s)", r"@main_1\1", tgt_func)
        final_tgt_name = "main_1"

    # Combine: source context + both function definitions
    parts = []
    parts.extend(src_ctx)
    parts.append("")
    parts.append(src_func)
    parts.append("")
    parts.append(tgt_func)
    return "\n".join(parts)


# ======================================================================
# Harness prompt
# ======================================================================

_HARNESS_PROMPT = """\
You are generating a C++ libFuzzer harness file named fuzz.cc.

Input:
- A single LLVM IR (.ll) file contains multiple function definitions.
- Functions that should be differentially fuzzed appear as pairs:
  - base: <name>
  - opt : <name>_opt
  Both have identical signatures.
- The .ll file will be compiled and linked together with fuzz.cc into one binary.

Task:
Generate fuzz.cc that performs differential fuzzing between each (base,opt) pair.

Hard requirements:
1) Output ONLY valid C++ code for fuzz.cc. No markdown fences. No explanations.
2) Include necessary #includes.

LLVM IR content:
{ll_text}
"""


# ======================================================================
# Build helpers
# ======================================================================

def _run_cmd(
    cmd: List[str], timeout: int = 60,
) -> Tuple[int, str]:
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


def _build_one(
    fuzz_cc: Path,
    combined_ll: Path,
    out_bin: Path,
    llc: str,
    clangxx: str,
    overwrite: bool,
) -> Tuple[Path, Optional[Path], Optional[str]]:
    """Compile combined.ll → .s via llc, then link with fuzz.cc → binary."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    if out_bin.exists() and not overwrite:
        return fuzz_cc, out_bin, None

    asm = out_bin.with_suffix(".s")

    # llc: IR → assembly
    rc, out = _run_cmd([llc, "-O3", str(combined_ll), "-o", str(asm)])
    if rc != 0:
        return fuzz_cc, None, f"llc failed (rc={rc}): {out[:500]}"

    # clang++: link fuzz.cc + asm → binary with sanitizers
    rc, out = _run_cmd([
        clangxx, str(fuzz_cc), str(asm),
        "-fsanitize=fuzzer,address,undefined",
        "-O1", "-g", "-o", str(out_bin),
    ])
    if rc != 0:
        return fuzz_cc, None, f"clang++ link failed (rc={rc}): {out[:500]}"

    return fuzz_cc, out_bin, None


# ======================================================================
# Alive2 (alive-tv) runner
# ======================================================================

def _run_alive2_one(
    combined_ll: Path,
    alive2_bin: str,
    src_fn: str,
    tgt_fn: str,
    timeout: int = 60,
    strict: bool = False,
) -> Tuple[str, str]:
    """Run alive-tv on a single combined.ll file.

    Returns (status, log_text) where status is one of:
      "PASS"    — transformation verified correct
      "FAIL"    — alive2 found a counter-example
      "TIMEOUT" — alive2 exceeded the time limit
      "ERROR"   — alive2 crashed or could not run
    """
    cmd = [alive2_bin, str(combined_ll),
           f"--src-fn={src_fn}", f"--tgt-fn={tgt_fn}"]
    log_text = f"CMD: {' '.join(cmd)}\n"

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", timeout=timeout,
        )
        log_text += f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nrc={proc.returncode}\n"

        ok_phrase = "Transformation seems to be correct!"
        if strict:
            if proc.returncode == 0 and ok_phrase in proc.stdout:
                return "PASS", log_text
            return "FAIL", log_text
        else:
            if ok_phrase in proc.stdout:
                return "PASS", log_text
            return "FAIL", log_text

    except subprocess.TimeoutExpired:
        log_text += "ERROR: Alive2 Timeout\n"
        return "TIMEOUT", log_text
    except Exception as e:
        log_text += f"ERROR: {e}\n"
        return "ERROR", log_text


# ======================================================================
# Fuzzing runner
# ======================================================================

def _run_fuzz_bin(
    bin_path: Path,
    fuzz_runs: int,
    fuzz_timeout: int,
) -> Tuple[str, int, str]:
    """Run a libFuzzer binary.  Returns (status, exit_code, output_tail)."""
    cmd = [str(bin_path)]
    if fuzz_runs > 0:
        cmd.append(f"-runs={fuzz_runs}")
    if fuzz_timeout > 0:
        cmd.append(f"-max_total_time={fuzz_timeout}")

    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=fuzz_timeout + 60 if fuzz_timeout > 0 else None,
        )
        snippet = (p.stdout or "")[-4000:]
        return ("PASS" if p.returncode == 0 else "FAIL"), p.returncode, snippet
    except subprocess.TimeoutExpired as e:
        return "TIMEOUT", 124, (e.stdout or "")[-4000:] if e.stdout else ""
    except Exception as e:
        return "ERROR", 127, str(e)



# ======================================================================
# Main class
# ======================================================================

class Verifier:
    """Orchestrates alive2 verification, harness generation, compilation, and fuzzing."""

    def __init__(
        self,
        llm: LLMClient,
        llc: str,
        clangxx: str,
    ):
        self.llm = llm
        self.llc = llc
        self.clangxx = clangxx

    # ------------------------------------------------------------------
    # Alive2 formal verification
    # ------------------------------------------------------------------

    def run_alive2(
        self,
        combined_dir: str,
        out_dir: str,
        alive2_bin: str,
        timeout: int = 60,
        workers: int = 16,
        strict: bool = False,
    ) -> Dict[str, dict]:
        """Run alive-tv on each <stem>/combined.ll.

        For each combined IR, extracts the two function names (src and
        src_opt) and invokes alive-tv with --src-fn / --tgt-fn.

        Returns dict mapping stem → {status, log_file}.
        """
        combined_p = Path(combined_dir)
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        ll_files = sorted(combined_p.glob("*/combined.ll"))
        if not ll_files:
            raise SystemExit(f"No combined.ll files found under {combined_p}")

        # Build job list: (stem, combined_ll, src_fn, tgt_fn)
        jobs: List[Tuple[str, Path, str, str]] = []
        skip: Dict[str, dict] = {}
        for ll_file in ll_files:
            stem = ll_file.parent.name
            ir_text = ll_file.read_text(encoding="utf-8", errors="ignore")
            _, defs = _parse_ir_structure(ir_text)
            if len(defs) < 2:
                skip[stem] = {"status": "PARSE_ERROR"}
                log(f"  {stem}: <2 functions in combined.ll, skipping")
                continue
            src_fn = _extract_func_name(defs[0])
            tgt_fn = _extract_func_name(defs[1])
            if not src_fn or not tgt_fn:
                skip[stem] = {"status": "PARSE_ERROR"}
                log(f"  {stem}: could not extract function names, skipping")
                continue
            jobs.append((stem, ll_file, src_fn, tgt_fn))

        log(f"Running alive2 on {len(jobs)} cases "
            f"(timeout={timeout}s, workers={workers}, strict={strict}) ...")

        results: Dict[str, dict] = dict(skip)

        def _do_one(stem, ll_file, src_fn, tgt_fn):
            status, log_text = _run_alive2_one(
                ll_file, alive2_bin, src_fn, tgt_fn,
                timeout=timeout, strict=strict,
            )
            # Save per-case log
            case_dir = out_p / stem
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "alive2_log.txt").write_text(log_text, encoding="utf-8")
            return stem, status

        if workers <= 1:
            for i, (stem, ll_file, src_fn, tgt_fn) in enumerate(jobs, 1):
                stem, status = _do_one(stem, ll_file, src_fn, tgt_fn)
                results[stem] = {"status": status, "log_file": str(out_p / stem / "alive2_log.txt")}
                log(f"  [{i}/{len(jobs)}] {stem}: {status}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(_do_one, s, ll, sf, tf): s
                    for s, ll, sf, tf in jobs
                }
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    stem, status = fut.result()
                    results[stem] = {"status": status, "log_file": str(out_p / stem / "alive2_log.txt")}
                    log(f"  [{done}/{len(jobs)}] {stem}: {status}")

        n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
        n_fail = sum(1 for r in results.values() if r["status"] == "FAIL")
        n_timeout = sum(1 for r in results.values() if r["status"] == "TIMEOUT")
        n_other = len(results) - n_pass - n_fail - n_timeout
        log(f"Alive2 done. PASS={n_pass}  FAIL={n_fail}  "
            f"TIMEOUT={n_timeout}  OTHER={n_other}")
        return results

    # ------------------------------------------------------------------
    # Step A: prepare combined IR files
    # ------------------------------------------------------------------

    def prepare_combined(
        self,
        original_dir: str,
        optimized_dir: str,
        out_dir: str,
    ) -> str:
        """For each *.optimized.ll in *optimized_dir*, find the matching
        original *.ll in *original_dir*, build combined.ll with _opt
        renamed functions.  Returns the output directory."""

        orig_p = Path(original_dir)
        opt_p = Path(optimized_dir)
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        opt_files = sorted(opt_p.glob("*.optimized.ll"))
        if not opt_files:
            # Also try plain *.ll if no *.optimized.ll
            opt_files = sorted(opt_p.glob("*.ll"))
        if not opt_files:
            raise SystemExit(f"No optimised .ll files found in {opt_p}")

        count = 0
        for opt_file in opt_files:
            # Derive the original stem: "12.optimized.ll" → "12"
            stem = opt_file.stem
            if stem.endswith(".optimized"):
                stem = stem[: -len(".optimized")]

            orig_file = orig_p / f"{stem}.ll"
            if not orig_file.exists():
                log(f"  WARN: no original IR for {stem}, skipping")
                continue

            orig_ir = orig_file.read_text(encoding="utf-8")
            opt_ir = opt_file.read_text(encoding="utf-8")
            combined = build_combined_ir(orig_ir, opt_ir)

            case_dir = out_p / stem
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "combined.ll").write_text(combined, encoding="utf-8")
            count += 1

        log(f"Prepared {count} combined IR files in {out_p}")
        return str(out_p)

    # ------------------------------------------------------------------
    # Step B: generate harness via LLM
    # ------------------------------------------------------------------

    def generate_harnesses(
        self,
        combined_dir: str,
        out_dir: str,
        model: str = "gpt-5",
        api_mode: str = "auto",
        workers: int = 50,
        max_output_tokens: int = 2048,
        overwrite: bool = False,
    ) -> str:
        """For each <stem>/combined.ll, ask the LLM to generate <stem>.fuzz.cc.
        Returns the harness output directory."""

        combined_p = Path(combined_dir)
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        ll_files = sorted(combined_p.glob("*/combined.ll"))
        if not ll_files:
            raise SystemExit(f"No combined.ll files found under {combined_p}")

        # Write prompt files so we can reuse LLMClient
        prompt_dir = out_p / "_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for ll_file in ll_files:
            stem = ll_file.parent.name
            fuzz_cc = out_p / f"{stem}.fuzz.cc"
            if fuzz_cc.exists() and not overwrite:
                continue
            ll_text = ll_file.read_text(encoding="utf-8", errors="ignore")
            prompt = _HARNESS_PROMPT.format(ll_text=ll_text)
            (prompt_dir / f"{stem}.prompt.ll").write_text(prompt, encoding="utf-8")

        # Use LLMClient to batch-query
        prompt_files = sorted(prompt_dir.glob("*.prompt.ll"))
        if not prompt_files:
            log("All harnesses already exist, skipping LLM calls")
            return str(out_p)

        log(f"Generating {len(prompt_files)} harnesses via LLM ...")
        pred_dir = self.llm.batch_query(
            in_dir=str(prompt_dir),
            out_dir=str(prompt_dir),
            model=model,
            api_mode=api_mode,
            workers=workers,
            max_output_tokens=max_output_tokens,
        )

        # Move predictions → <stem>.fuzz.cc
        for pred_file in Path(pred_dir).glob("*.model.predict.ll"):
            stem = pred_file.name.replace(".model.predict.ll", "")
            fuzz_cc = out_p / f"{stem}.fuzz.cc"
            content = pred_file.read_text(encoding="utf-8")
            # Strip markdown fences if the LLM wrapped the code
            content = re.sub(r"^```(?:cpp|c\+\+)?\s*\n", "", content)
            content = re.sub(r"\n```\s*$", "", content)
            fuzz_cc.write_text(content, encoding="utf-8")

        log(f"Harnesses saved to {out_p}")
        return str(out_p)

    # ------------------------------------------------------------------
    # Step C: build binaries
    # ------------------------------------------------------------------

    def build_binaries(
        self,
        combined_dir: str,
        harness_dir: str,
        out_dir: str,
        workers: int = 16,
        overwrite: bool = False,
    ) -> str:
        """Compile each (combined.ll, fuzz.cc) pair into a fuzzing binary.
        Returns the binary output directory."""

        combined_p = Path(combined_dir)
        harness_p = Path(harness_dir)
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        fuzz_files = sorted(harness_p.glob("*.fuzz.cc"))
        if not fuzz_files:
            raise SystemExit(f"No *.fuzz.cc found in {harness_p}")

        jobs = []
        for fuzz_cc in fuzz_files:
            stem = fuzz_cc.name[: -len(".fuzz.cc")]
            ll_file = combined_p / stem / "combined.ll"
            if not ll_file.exists():
                log(f"  WARN: no combined.ll for {stem}, skipping")
                continue
            out_bin = out_p / stem / f"{stem}_fuzz"
            jobs.append((fuzz_cc, ll_file, out_bin))

        if not jobs:
            raise SystemExit("No buildable (fuzz.cc, combined.ll) pairs found")

        log(f"Building {len(jobs)} fuzzing binaries (workers={workers}) ...")

        ok = fail = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [
                ex.submit(
                    _build_one, fcc, ll, obin,
                    self.llc, self.clangxx, overwrite,
                )
                for fcc, ll, obin in jobs
            ]
            total = len(futs)
            for i, fut in enumerate(as_completed(futs), start=1):
                fcc, obin, err = fut.result()
                if err is None and obin is not None:
                    ok += 1
                    if i % 20 == 0 or i == total:
                        log(f"  [{i}/{total}] OK: {obin}")
                else:
                    fail += 1
                    log(f"  [{i}/{total}] FAIL: {fcc.name} — {err}")

        log(f"Build done. ok={ok}  fail={fail}")
        return str(out_p)

    # ------------------------------------------------------------------
    # Step D: run fuzzing
    # ------------------------------------------------------------------

    def run_fuzzing(
        self,
        bin_dir: str,
        fuzz_runs: int = 200000,
        fuzz_timeout: int = 600,
        workers: int = 1,
    ) -> Dict[str, dict]:
        """Run each fuzzing binary and collect results.

        *workers* controls parallelism: 1 = sequential, >1 = parallel.

        Returns a dict mapping stem → {status, exit_code, output_tail}."""

        bin_p = Path(bin_dir)
        bins = sorted(bin_p.rglob("*_fuzz"))
        bins = [b for b in bins if b.is_file()]
        if not bins:
            raise SystemExit(f"No fuzzing binaries found under {bin_p}")

        log(f"Running {len(bins)} fuzzing binaries "
            f"(runs={fuzz_runs}, timeout={fuzz_timeout}s, workers={workers}) ...")

        results: Dict[str, dict] = {}

        if workers <= 1:
            for i, b in enumerate(bins, start=1):
                stem = b.parent.name
                status, rc, tail = _run_fuzz_bin(b, fuzz_runs, fuzz_timeout)
                results[stem] = {
                    "status": status, "exit_code": rc, "output_tail": tail,
                }
                marker = "PASS" if status == "PASS" else f"**{status}**"
                log(f"  [{i}/{len(bins)}] {stem}: {marker} (rc={rc})")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(_run_fuzz_bin, b, fuzz_runs, fuzz_timeout): b
                    for b in bins
                }
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    b = futs[fut]
                    stem = b.parent.name
                    status, rc, tail = fut.result()
                    results[stem] = {
                        "status": status, "exit_code": rc, "output_tail": tail,
                    }
                    marker = "PASS" if status == "PASS" else f"**{status}**"
                    log(f"  [{done}/{len(bins)}] {stem}: {marker} (rc={rc})")

        n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
        n_fail = sum(1 for r in results.values() if r["status"] == "FAIL")
        n_other = len(results) - n_pass - n_fail
        log(f"Fuzzing done. PASS={n_pass}  FAIL={n_fail}  OTHER={n_other}")
        return results

    # ------------------------------------------------------------------
    # Full pipeline: prepare → harness → build → fuzz
    # ------------------------------------------------------------------

    def run_full(
        self,
        original_dir: str,
        optimized_dir: str,
        work_dir: str,
        harness_dir: str = "",
        model: str = "gpt-5",
        api_mode: str = "auto",
        workers: int = 50,
        build_workers: int = 16,
        fuzz_runs: int = 200000,
        fuzz_timeout: int = 600,
        fuzz_workers: int = 1,
    ) -> Dict[str, dict]:
        """End-to-end diff testing.  Returns fuzzing results dict.

        If *harness_dir* is provided and contains *.fuzz.cc files, skip
        harness generation and use the given directory directly."""

        work = Path(work_dir)
        combined_dir = str(work / "combined")
        bin_dir = str(work / "bins")

        self.prepare_combined(original_dir, optimized_dir, combined_dir)

        # Use provided harness dir or generate one
        if harness_dir and Path(harness_dir).is_dir() and any(Path(harness_dir).glob("*.fuzz.cc")):
            log(f"Using existing harnesses: {harness_dir}")
        else:
            harness_dir = str(work / "harness")
            self.generate_harnesses(
                combined_dir, harness_dir,
                model=model, api_mode=api_mode, workers=workers,
            )

        self.build_binaries(
            combined_dir, harness_dir, bin_dir, workers=build_workers,
        )
        return self.run_fuzzing(
            bin_dir, fuzz_runs=fuzz_runs, fuzz_timeout=fuzz_timeout,
            workers=fuzz_workers,
        )
