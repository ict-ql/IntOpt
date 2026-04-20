"""Refine optimisation strategies by running LLVM opt analysis passes
and building enriched prompts with analysis info."""

import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modules.utils import clean_block, extract_tagged_blocks, log


class StrategyRefinement:
    def __init__(self, opt_bin: str):
        self.OPT_BIN = Path(opt_bin)
        self.PRINT_TOKEN_RE = re.compile(r"^print<([^>]+)>$")
        self.DENY_ANALYSIS_TOKENS = {"scalar-evolution", "memoryssa", "postdomtree"}
        self.DOMTREE_NODE_RE = re.compile(r"^(\s*\[\d+\]\s+%[^\s]+)")
        self.BRANCH_PROB_RE = re.compile(
            r"^(\s*edge .* probability is )0x[0-9A-Fa-f]+ / 0x[0-9A-Fa-f]+ = ([0-9.]+%)(.*)$"
        )

    # ------------------------------------------------------------------
    # Advice extraction (reuses shared tag extractor)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_advice(in_dir: Path, prefix: str) -> str:
        p = in_dir / f"{prefix}.model.predict.ll"
        if not p.exists():
            return ""
        text = p.read_text(encoding="utf-8", errors="ignore")
        blocks = extract_tagged_blocks(text, "code")
        return blocks[0] if blocks else ""

    # ------------------------------------------------------------------
    # Prefix / index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _txt_to_prefix(txt_path: Path) -> str:
        stem = txt_path.stem
        if stem.endswith(".analysis_passes"):
            return stem[: -len(".analysis_passes")]
        return stem.split(".", 1)[0]

    @staticmethod
    def _build_ll_index(ll_dir: Path) -> Dict[str, List[Path]]:
        idx: Dict[str, List[Path]] = {}
        for p in ll_dir.rglob("*.ll"):
            stem = p.stem
            for key in {stem, stem.split(".", 1)[0]}:
                idx.setdefault(key, []).append(p)
        for k in idx:
            idx[k].sort(key=lambda x: (len(x.name), x.name))
        return idx

    @staticmethod
    def _choose_ll(prefix: str, candidates: List[Path], ll_dir: Path) -> Optional[Path]:
        direct = ll_dir / f"{prefix}.ll"
        if direct.exists():
            return direct
        exact = [p for p in candidates if p.name == f"{prefix}.ll"]
        if exact:
            return exact[0]
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # opt runner
    # ------------------------------------------------------------------

    @staticmethod
    def _run_opt(
        opt_bin: Path, ll_path: Path, pipeline: str, timeout_sec: int,
    ) -> Tuple[int, str, str, float]:
        cmd = [str(opt_bin), "-disable-output", f"-passes={pipeline}", str(ll_path)]
        t0 = time.time()
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=timeout_sec,
            )
            dt = time.time() - t0
            return p.returncode, (p.stdout or ""), (p.stderr or ""), dt
        except subprocess.TimeoutExpired as e:
            dt = time.time() - t0
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            return 124, stdout, stderr + f"\n[timeout after {timeout_sec}s]\n", dt

    def _opt_accepts(
        self, pipeline: str, probe_ll: Path, timeout: int,
    ) -> bool:
        rc, out, err, _ = self._run_opt(self.OPT_BIN, probe_ll, pipeline, timeout)
        text = (out + "\n" + err).lower()
        if "unknown pass name" in text or "unknown pass" in text:
            return False
        return rc == 0

    # ------------------------------------------------------------------
    # Pipeline selection / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pipeline_like(s: str) -> bool:
        s = s.strip()
        return bool(s) and (("<" in s and ">" in s) or ("(" in s and ")" in s))

    @staticmethod
    def _read_pass_lines(txt_path: Path) -> List[str]:
        return [
            s for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if (s := line.strip()) and not s.startswith("#")
        ]

    @staticmethod
    def _make_probe_ir(tmp_dir: Path) -> Path:
        p = tmp_dir / "probe.ll"
        p.write_text(
            'source_filename = "probe.ll"\n'
            "define void @f() {\nentry:\n  ret void\n}\n",
            encoding="utf-8",
        )
        return p

    def _token_from_pipeline(self, p: str) -> Optional[str]:
        m = self.PRINT_TOKEN_RE.match((p or "").strip())
        return m.group(1) if m else None

    def _select_pipelines(
        self,
        raw_lines: List[str],
        probe_ll: Path,
        verify_timeout: int,
        cache_ok: Dict[str, bool],
    ) -> Tuple[List[str], List[str], str]:
        pipeline_lines = [s for s in raw_lines if self._is_pipeline_like(s)]

        if pipeline_lines:
            pipeline_lines = [
                p for p in pipeline_lines
                if self._token_from_pipeline(p) not in self.DENY_ANALYSIS_TOKENS
            ]

        if pipeline_lines:
            ok, dropped = [], []
            for p in pipeline_lines:
                if p not in cache_ok:
                    cache_ok[p] = self._opt_accepts(p, probe_ll, verify_timeout)
                (ok if cache_ok[p] else dropped).append(p)
            return list(dict.fromkeys(ok)), dropped, "pipeline_section"

        # Fall back: treat each line as a bare token
        ok, dropped = [], []
        for tok in raw_lines:
            if tok in self.DENY_ANALYSIS_TOKENS:
                dropped.append(tok)
                continue
            cand = f"print<{tok}>"
            if cand not in cache_ok:
                cache_ok[cand] = self._opt_accepts(cand, probe_ll, verify_timeout)
            (ok if cache_ok[cand] else dropped).append(cand if cache_ok.get(cand) else tok)
        return list(dict.fromkeys(ok)), dropped, "token_to_print"

    # ------------------------------------------------------------------
    # Analysis output formatting
    # ------------------------------------------------------------------

    def _simplify_domtree(self, lines: List[str]) -> List[str]:
        out = []
        for ln in lines:
            m = self.DOMTREE_NODE_RE.match(ln)
            out.append(m.group(1) if m else ln)
        return out

    def _sanitize_assumptions(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        skipping_bfi = False
        for ln in lines:
            if not skipping_bfi and ln.startswith("Printing analysis results of BFI"):
                skipping_bfi = True
                continue
            if skipping_bfi:
                if "Branch Probability Analysis" in ln:
                    skipping_bfi = False
                else:
                    continue
            if ln.startswith("block-frequency-info:"):
                continue
            if ln.startswith(" - B") or ln.startswith("- B"):
                continue
            m = self.BRANCH_PROB_RE.match(ln)
            if m:
                ln = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            out.append(ln)
        return out

    def _build_start_patterns(self) -> List[Tuple[re.Pattern, str]]:
        pairs = [
            (r"^Cached assumptions for function:", "assumptions"),
            (r"^Assumptions for function:", "assumptions"),
            (r"^DominatorTree for function:", "domtree"),
            (r"^IV Users for loop", "iv-users"),
            (r"^Loop info for function", "loops"),
            (r"^PostDominatorTree for function:", "postdomtree"),
            (r"^Printing analysis 'Scalar Evolution Analysis' for function", "scalar-evolution"),
            (r"^Classifying expressions for:", "scalar-evolution"),
            (r"^Determining loop execution counts for:", "scalar-evolution"),
            (r"^LVI for function", "lazy-value-info"),
        ]
        return [(re.compile(p), tok) for p, tok in pairs]

    def _format_analysis(self, pipelines: List[str], stdout: str, stderr: str) -> str:
        want_tokens = []
        for p in pipelines:
            m = self.PRINT_TOKEN_RE.match(p.strip())
            if m and m.group(1) not in self.DENY_ANALYSIS_TOKENS:
                want_tokens.append(m.group(1))

        text = (stderr or "")
        if stdout and stdout.strip():
            text += ("\n" if text and not text.endswith("\n") else "") + stdout

        start_patterns = self._build_start_patterns()
        blocks: Dict[str, List[str]] = {}
        cur_tok: Optional[str] = None

        for ln in text.splitlines():
            matched = None
            for pat, tok in start_patterns:
                if pat.search(ln):
                    matched = tok
                    break
            if matched is not None:
                cur_tok = matched
                blocks.setdefault(cur_tok, []).append(ln)
            elif cur_tok is not None:
                blocks[cur_tok].append(ln)

        parts: List[str] = []
        for tok in want_tokens:
            content = blocks.get(tok, []).copy()
            while content and not content[0].strip():
                content.pop(0)
            while content and not content[-1].strip():
                content.pop()

            if tok == "domtree":
                content = self._simplify_domtree(content)
            elif tok == "assumptions":
                content = self._sanitize_assumptions(content)

            if tok == "loops" and len(content) <= 1:
                parts += ["Loops: there is no loops in this function", ""]
                continue
            if tok == "lazy-value-info" and len(content) <= 1:
                parts += ["LazyValueInfo: No lazy value info for this function", ""]
                continue
            if len(content) <= 1:
                continue

            title = tok[0].upper() + tok[1:] if tok else tok
            parts.append(f"{title}:")
            parts.extend(content)
            parts.append("")

        while parts and not parts[-1].strip():
            parts.pop()
        return "\n".join(parts) + ("\n" if parts else "")

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(ir: str, advice: str, analysis: str,
                      intrinsic_advice: str = "") -> str:
        ir = (ir or "").strip()
        advice = (advice or "").strip()
        analysis = (analysis or "").strip()
        intrinsic_advice = (intrinsic_advice or "").strip()

        parts = [
            "Please optimize the following code to outperform LLVM -O3.",
            "", "<code>", ir, "</code>", "",
        ]
        if advice:
            parts += [
                "You may refer to the following advice, but feel free to "
                "adapt, extend, or deviate from it as you see fit.",
                "<advice>", advice, "</advice>", "",
            ]
        parts += ["The corresponding analysis info is below.", "<analysis>"]
        if analysis:
            parts += ["", analysis]
        parts += ["</analysis>", ""]
        if intrinsic_advice:
            parts += [
                "IMPORTANT: The following hardware intrinsics are available on "
                "the target CPU and are highly relevant to this code. You SHOULD "
                "actively try to use them in your optimized IR where applicable. "
                "These intrinsics map directly to efficient hardware instructions "
                "and can provide significant speedups over scalar/generic code.",
                "",
                "<intrinsics>", intrinsic_advice, "</intrinsics>", "",
                "In your <advice>, explicitly state which intrinsics you plan to "
                "use and why. In your <code>, use the LLVM IR call syntax to "
                "invoke them (e.g., `call <ret_ty> @llvm.x86.avx512.vfmadd.ps.512(...)`).",
                "",
            ]
        parts += [
            "You need to keep boundary checks.",
            "Please output the final optimization advice wrapped in "
            "<advice>...</advice> and the full optimized LLVM IR wrapped "
            "in <code>...</code>.",
            "",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def refine(
        self,
        in_dir: str,
        ll_dir: str,
        initial_dir: str,
        out_dir: str,
        timeout: int = 30,
        verify_timeout: int = 4,
        max_files: int = 0,
        continue_on_error: bool = False,
        save_intermediate: bool = False,
        intrinsic_advice: str = "",
    ) -> str:
        """Build analysis-enriched prompts.  Returns the output directory."""

        in_dir_p = Path(in_dir)
        ll_dir_p = Path(ll_dir)
        initial_dir_p = Path(initial_dir)

        for label, p in [("in_dir", in_dir_p), ("ll_dir", ll_dir_p), ("initial_dir", initial_dir_p)]:
            if not p.is_dir():
                raise SystemExit(f"{label} not a directory: {p}")
        if not self.OPT_BIN.exists():
            raise SystemExit(f"opt binary not found: {self.OPT_BIN}")

        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(in_dir_p.glob("*.analysis_passes.txt"))
        if max_files > 0:
            txt_files = txt_files[:max_files]
        if not txt_files:
            raise SystemExit(f"No *.analysis_passes.txt found under {in_dir_p}")

        log(f"Building .ll index under: {ll_dir_p}")
        ll_index = self._build_ll_index(ll_dir_p)
        log(f"Found ll index keys: {len(ll_index)}  |  txt files: {len(txt_files)}")

        cache_ok: Dict[str, bool] = {}
        tmp_root = out_dir_p / "_tmp_probe"
        tmp_root.mkdir(parents=True, exist_ok=True)
        probe_ll = self._make_probe_ir(tmp_root)

        for i, txt_path in enumerate(txt_files, start=1):
            prefix = self._txt_to_prefix(txt_path)
            candidates = ll_index.get(prefix, [])
            src_ll = self._choose_ll(prefix, candidates, ll_dir_p)
            out_prompt = out_dir_p / f"{prefix}.prompt.ll"

            log(f"[{i}/{len(txt_files)}] prefix={prefix}")

            if not src_ll:
                msg = f"No .ll found for prefix={prefix} under {ll_dir_p}"
                log(f"  ERROR: {msg}")
                if not continue_on_error:
                    raise SystemExit(msg)
                out_prompt.write_text("", encoding="utf-8")
                continue

            ir = src_ll.read_text(encoding="utf-8", errors="ignore")
            if not ir:
                msg = f"Empty IR file: {src_ll}"
                log(f"  ERROR: {msg}")
                if not continue_on_error:
                    raise SystemExit(msg)
                out_prompt.write_text("", encoding="utf-8")
                continue

            advice = self._read_advice(initial_dir_p, prefix)
            raw_lines = self._read_pass_lines(txt_path)
            pipelines_ok, dropped, mode = self._select_pipelines(
                raw_lines, probe_ll, verify_timeout, cache_ok,
            )

            analysis_text = ""
            if pipelines_ok:
                extracted_ll = out_dir_p / f"{prefix}.extracted.ll"
                extracted_ll.write_text(
                    ir + ("" if ir.endswith("\n") else "\n"), encoding="utf-8",
                )
                pipeline = ",".join(pipelines_ok)
                rc, out, err, dt = self._run_opt(
                    self.OPT_BIN, extracted_ll, pipeline, timeout,
                )
                analysis_text = self._format_analysis(pipelines_ok, out, err)
                log(f"  opt: mode={mode} ok={len(pipelines_ok)} "
                    f"dropped={len(dropped)} rc={rc} {dt:.3f}s")

                if save_intermediate:
                    (out_dir_p / f"{prefix}.advice.txt").write_text(
                        advice + ("\n" if advice and not advice.endswith("\n") else ""),
                        encoding="utf-8",
                    )
                    (out_dir_p / f"{prefix}.raw_opt.txt").write_text(
                        (err or "") + ("\n" if err and not err.endswith("\n") else "") + (out or ""),
                        encoding="utf-8",
                    )
            else:
                log("  WARN: no valid pipelines; analysis section will be empty")

            prompt = self._build_prompt(
                ir=ir, advice=advice, analysis=analysis_text,
                intrinsic_advice=intrinsic_advice,
            )
            out_prompt.write_text(prompt, encoding="utf-8")
            log(f"  wrote: {out_prompt.name}")

        log("Done.")
        return str(out_dir_p)
