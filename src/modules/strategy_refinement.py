import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class StrategyRefinement:
    def __init__(self, opt_bin: str):
        self.OPT_BIN = Path(opt_bin)
        self.CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
        self.IR_RE = re.compile(r"<ir>(.*?)</ir>", re.DOTALL | re.IGNORECASE)
        self.PRINT_TOKEN_RE = re.compile(r"^print<([^>]+)>$")
        self.HTML_UNESCAPE = (
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&amp;", "&"),
        )
        self.DENY_ANALYSIS_TOKENS = {
            "scalar-evolution",
            "memoryssa",
            "postdomtree",
        }
        self.DOMTREE_NODE_RE = re.compile(r"^(\s*\[\d+\]\s+%[^\s]+)")
        self.BRANCH_PROB_RE = re.compile(
            r"^(\s*edge .* probability is )0x[0-9A-Fa-f]+ / 0x[0-9A-Fa-f]+ = ([0-9.]+%)(.*)$"
        )
    
    def read_advice_from_in_dir(self, in_dir: Path, prefix: str) -> str:
        p = in_dir / f"{prefix}.model.predict.ll"
        if not p.exists():
            return ""
        text = p.read_text(encoding="utf-8", errors="ignore")
        code_blocks: List[str] = []
        for m in self.CODE_RE.finditer(text):
            blk = self._clean_block(m.group(1))
            # Filter out the common placeholder caught from "single <code>...</code> block"
            if blk == "..." or not blk:
                continue
            code_blocks.append(blk)
        if code_blocks:
            advice = code_blocks[0] if len(code_blocks) >= 1 else ""
        return advice
    
    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
    
    def txt_to_prefix(self, txt_path: Path) -> str:
        stem = txt_path.stem
        if stem.endswith(".analysis_passes"):
            return stem[: -len(".analysis_passes")]
        return stem.split(".", 1)[0]
    
    def ll_to_prefix(self, ll_path: Path) -> str:
        return ll_path.stem
    
    def build_ll_index(self, ll_dir: Path) -> Dict[str, List[Path]]:
        idx: Dict[str, List[Path]] = {}
        for p in ll_dir.rglob("*.ll"):
            stem = self.ll_to_prefix(p)
            k1 = stem
            k2 = stem.split(".", 1)[0]
            idx.setdefault(k1, []).append(p)
            idx.setdefault(k2, []).append(p)
        for k in idx:
            idx[k].sort(key=lambda x: (len(x.name), x.name))
        return idx
    
    def choose_ll(self, prefix: str, candidates: List[Path], ll_dir: Path) -> Optional[Path]:
        direct = ll_dir / f"{prefix}.ll"
        if direct.exists():
            return direct
        exact = [p for p in candidates if p.name == f"{prefix}.ll"]
        if exact:
            return exact[0]
        return candidates[0] if candidates else None
    
    def _unescape_html(self, s: str) -> str:
        for a, b in self.HTML_UNESCAPE:
            s = s.replace(a, b)
        return s
    
    def _clean_block(self, s: str) -> str:
        s = self._unescape_html(s or "")
        s = s.strip("\n\r\t ")
        return s
    
    def run_opt(self, opt_bin: Path, ll_path: Path, pipeline: str, timeout_sec: int) -> Tuple[int, str, str, float]:
        cmd = [str(opt_bin), "-disable-output", f"-passes={pipeline}", str(ll_path)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)
            dt = time.time() - t0
            return p.returncode, (p.stdout or ""), (p.stderr or ""), dt
        except subprocess.TimeoutExpired as e:
            dt = time.time() - t0
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            return 124, stdout, stderr + f"\n[timeout after {timeout_sec}s]\n", dt
    
    def is_pipeline_like(self, s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        if ("<" in s and ">" in s):
            return True
        if "(" in s and ")" in s:
            return True
        return False
    
    def read_pass_lines(self, txt_path: Path) -> List[str]:
        lines: List[str] = []
        for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        return lines
    
    def make_probe_ir(self, tmp_dir: Path) -> Path:
        p = tmp_dir / "probe.ll"
        p.write_text(
            "source_filename = \"probe.ll\"\n"
            "define void @f() {\n"
            "entry:\n"
            "  ret void\n"
            "}\n",
            encoding="utf-8",
        )
        return p
    
    def opt_accepts_pipeline(self, opt_bin: Path, pipeline: str, probe_ll: Path, timeout_sec: int) -> bool:
        rc, out, err, _ = self.run_opt(opt_bin, probe_ll, pipeline, timeout_sec=timeout_sec)
        text = (out + "\n" + err).lower()
        if "unknown pass name" in text or "unknown pass" in text:
            return False
        return rc == 0
    
    def token_from_pipeline(self, p: str) -> Optional[str]:
        m = self.PRINT_TOKEN_RE.match((p or "").strip())
        return m.group(1) if m else None
    
    def select_pipelines(
        self,
        raw_lines: List[str],
        opt_bin: Path,
        probe_ll: Path,
        verify_timeout: int,
        cache_ok: Dict[str, bool],
    ) -> Tuple[List[str], List[str], str]:
        pipeline_lines = [s for s in raw_lines if self.is_pipeline_like(s)]

        if pipeline_lines:
            filtered: List[str] = []
            for p in pipeline_lines:
                tok = self.token_from_pipeline(p)
                if tok in self.DENY_ANALYSIS_TOKENS:
                    continue
                filtered.append(p)
            pipeline_lines = filtered

        if pipeline_lines:
            ok: List[str] = []
            dropped: List[str] = []
            for p in pipeline_lines:
                if p not in cache_ok:
                    cache_ok[p] = self.opt_accepts_pipeline(opt_bin, p, probe_ll, timeout_sec=verify_timeout)
                if cache_ok[p]:
                    ok.append(p)
                else:
                    dropped.append(p)

            seen = set()
            out: List[str] = []
            for p in ok:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out, dropped, "pipeline_section"

        ok = []
        dropped = []
        for tok in raw_lines:
            if tok in self.DENY_ANALYSIS_TOKENS:
                dropped.append(tok)
                continue
            cand = f"print<{tok}>"
            if cand not in cache_ok:
                cache_ok[cand] = self.opt_accepts_pipeline(opt_bin, cand, probe_ll, timeout_sec=verify_timeout)
            if cache_ok[cand]:
                ok.append(cand)
            else:
                dropped.append(tok)

        seen = set()
        out = []
        for p in ok:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out, dropped, "token_to_print"
    
    def simplify_domtree_lines(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        for ln in lines:
            m = self.DOMTREE_NODE_RE.match(ln)
            if m:
                out.append(m.group(1))
            else:
                out.append(ln)
        return out
    
    def sanitize_assumptions_lines(self, lines: List[str]) -> List[str]:
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
    
    def pretty_title(self, token: str) -> str:
        if not token:
            return token
        return token[0].upper() + token[1:]
    
    def build_start_patterns(self) -> List[Tuple[re.Pattern, str]]:
        pairs: List[Tuple[str, str]] = [
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
    
    def format_analysis_blocks(self, pipelines: List[str], stdout: str, stderr: str) -> str:
        want_tokens: List[str] = []
        for p in pipelines:
            m = self.PRINT_TOKEN_RE.match(p.strip())
            if m:
                tok = m.group(1)
                if tok in self.DENY_ANALYSIS_TOKENS:
                    continue
                want_tokens.append(tok)

        text = (stderr or "")
        if stdout and stdout.strip():
            text = text + ("\n" if text and not text.endswith("\n") else "") + stdout

        lines = text.splitlines()
        start_patterns = self.build_start_patterns()

        blocks: Dict[str, List[str]] = {}
        cur_tok: Optional[str] = None

        def start_token(ln: str) -> Optional[str]:
            for pat, tok in start_patterns:
                if pat.search(ln):
                    return tok
            return None

        for ln in lines:
            tok = start_token(ln)
            if tok is not None:
                cur_tok = tok
                blocks.setdefault(cur_tok, []).append(ln)
                continue
            if cur_tok is not None:
                blocks[cur_tok].append(ln)

        out_parts: List[str] = []
        for tok in want_tokens:
            content = blocks.get(tok, []).copy()

            while content and not content[0].strip():
                content.pop(0)
            while content and not content[-1].strip():
                content.pop()

            if tok == "domtree":
                content = self.simplify_domtree_lines(content)
            elif tok == "assumptions":
                content = self.sanitize_assumptions_lines(content)

            if tok == "loops" and len(content) <= 1:
                out_parts.append("Loops: there is no loops in this function")
                out_parts.append("")
                continue

            if tok == "lazy-value-info" and len(content) <= 1:
                out_parts.append("LazyValueInfo: No lazy value info for this function")
                out_parts.append("")
                continue

            if len(content) <= 1:
                continue

            out_parts.append(f"{self.pretty_title(tok)}:")
            out_parts.extend(content)
            out_parts.append("")

        while out_parts and not out_parts[-1].strip():
            out_parts.pop()

        return "\n".join(out_parts) + ("\n" if out_parts else "")
    
    def build_prompt(self, ir: str, advice: str, analysis: str) -> str:
        ir = (ir or "").strip()
        advice = (advice or "").strip()
        analysis = (analysis or "").strip()

        parts: List[str] = []
        parts.append("Please optimize the following code to outperform LLVM -O3.")
        parts.append("")
        parts.append("<code>")
        parts.append(ir)
        parts.append("</code>")
        parts.append("")

        if advice:
            parts.append("You may refer to the following advice, but feel free to adapt, extend, or deviate from it as you see fit.")
            parts.append("<advice>")
            parts.append(advice)
            parts.append("</advice>")
            parts.append("")

        parts.append("The corresponding analysis info is below.")
        parts.append("<analysis>")
        if analysis:
            parts.append("")
            parts.append(analysis)
        parts.append("</analysis>")
        parts.append("")
        parts.append("You need to keep boundary checks.")
        # `and the full optimized LLVM IR wrapped in <code>...</code>...` just for making the advice as brief as it can
        parts.append("Please output the final optimization advice wrapped in <advice>...</advice> and the full optimized LLVM IR wrapped in <code>...</code>.")
        parts.append("")

        return "\n".join(parts)
    
    def refine_strategies(self, in_dir: str, ll_dir: str, initial_dir: str, out_dir: str, timeout: int = 30, verify_timeout: int = 4, max_files: int = 0, continue_on_error: bool = False, save_intermediate: bool = False):
        in_dir = Path(in_dir)
        ll_dir = Path(ll_dir)
        initial_dir = Path(initial_dir)
        opt_bin = self.OPT_BIN

        if not in_dir.is_dir():
            raise SystemExit(f"in_dir not a directory: {in_dir}")
        if not ll_dir.is_dir():
            raise SystemExit(f"ll_dir not a directory: {ll_dir}")
        if not initial_dir.is_dir():
            raise SystemExit(f"initial_dir not a directory: {initial_dir}")
        if not opt_bin.exists():
            raise SystemExit(f"--opt_bin not found: {opt_bin}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(in_dir.glob("*.analysis_passes.txt"))
        if max_files > 0:
            txt_files = txt_files[: max_files]
        if not txt_files:
            raise SystemExit(f"No txt files matched: {in_dir}/*.analysis_passes.txt")

        self.log(f"Building .ll index under: {ll_dir}")
        ll_index = self.build_ll_index(ll_dir)
        self.log(f"Found ll index keys: {len(ll_index)}")
        self.log(f"Found txt files: {len(txt_files)}")

        cache_ok: Dict[str, bool] = {}
        tmp_root = out_dir / "_tmp_probe"
        tmp_root.mkdir(parents=True, exist_ok=True)
        probe_ll = self.make_probe_ir(tmp_root)

        for i, txt_path in enumerate(txt_files, start=1):
            prefix = self.txt_to_prefix(txt_path)
            self.log(prefix)
            candidates = ll_index.get(prefix, [])
            self.log(candidates)
            src_ll = self.choose_ll(prefix, candidates, ll_dir)

            out_prompt = out_dir / f"{prefix}.prompt.ll"

            self.log(f"[{i}/{len(txt_files)}] prefix={prefix}")

            if not src_ll:
                msg = f"ERROR: no .ll found for prefix={prefix} under {ll_dir}"
                self.log("  " + msg)
                if not continue_on_error:
                    raise SystemExit(msg)
                out_prompt.write_text("", encoding="utf-8")
                continue

            ir = src_ll.read_text(encoding="utf-8", errors="ignore")
            if not ir:
                msg = f"ERROR: no <ir>...</ir> (or fallback <code>...</code>) found in {src_ll}"
                self.log("  " + msg)
                if not continue_on_error:
                    raise SystemExit(msg)
                out_prompt.write_text("", encoding="utf-8")
                continue

            advice = self.read_advice_from_in_dir(initial_dir, prefix)

            raw_lines = self.read_pass_lines(txt_path)
            pipelines_ok, dropped, mode = self.select_pipelines(
                raw_lines,
                opt_bin=opt_bin,
                probe_ll=probe_ll,
                verify_timeout=verify_timeout,
                cache_ok=cache_ok,
            )

            analysis_text = ""
            if pipelines_ok:
                extracted_ll = out_dir / f"{prefix}.extracted.ll"
                extracted_ll.write_text(ir + ("\n" if not ir.endswith("\n") else ""), encoding="utf-8")

                pipeline = ",".join(pipelines_ok)
                rc, out, err, dt = self.run_opt(opt_bin, extracted_ll, pipeline, timeout_sec=timeout)
                analysis_text = self.format_analysis_blocks(pipelines_ok, stdout=out, stderr=err)

                self.log(f"  opt: mode={mode} ok={len(pipelines_ok)} dropped={len(dropped)} rc={rc} {dt:.3f}s")

                if save_intermediate:
                    (out_dir / f"{prefix}.advice.txt").write_text(advice + ("\n" if advice and not advice.endswith("\n") else ""), encoding="utf-8")
                    (out_dir / f"{prefix}.raw_opt.txt").write_text((err or "") + ("\n" if err and not err.endswith("\n") else "") + (out or ""), encoding="utf-8")
            else:
                self.log("  WARN: no valid pipelines; analysis section will be empty")

            prompt = self.build_prompt(ir=ir, advice=advice, analysis=analysis_text)
            out_prompt.write_text(prompt, encoding="utf-8")
            self.log(f"  wrote: {out_prompt.name}")

        self.log("Done.")
        return out_dir
