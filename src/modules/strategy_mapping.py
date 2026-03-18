"""Map model-predicted optimisation steps to concrete LLVM analysis passes
using TF-IDF similarity against the LLVM pass catalog."""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.utils import log


class StrategyMapping:
    def __init__(self, passregistry_def: str, llvm_lib_root: str, opt_bin: str):
        self.PASSREGISTRY_DEF = Path(passregistry_def)
        self.LLVM_LIB_ROOT = Path(llvm_lib_root)
        self.OPT_BIN = Path(opt_bin)
        self.PASSES_URL = "https://llvm.org/docs/Passes.html"

    # ------------------------------------------------------------------
    # Text normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower().replace("llvm.", "llvm ")
        text = re.sub(r"[^a-z0-9_\-\s]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------------------------------------------
    # Step extraction from model output
    # ------------------------------------------------------------------

    _STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL | re.IGNORECASE)
    _TRANS_RE = re.compile(r"\*\*Transformation\*\*:\s*(.+)", re.IGNORECASE)
    _CHANGE_RE = re.compile(r"\*\*Change\*\*:\s*(.+)", re.IGNORECASE | re.DOTALL)

    def _extract_steps(self, raw: str) -> List[str]:
        blocks = [b.strip() for b in self._STEP_RE.findall(raw)]
        return [b for b in blocks if self._normalize(b)]

    def _parse_step_fields(self, step_block: str) -> Tuple[str, str]:
        m_t = self._TRANS_RE.search(step_block)
        transformation = m_t.group(1).strip() if m_t else ""
        m_c = self._CHANGE_RE.search(step_block)
        change = m_c.group(1).strip() if m_c else ""
        return transformation, change

    # ------------------------------------------------------------------
    # LLVM Passes.html catalog
    # ------------------------------------------------------------------

    def _fetch_passes_html(self, url: str) -> str:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _find_scope(self, sec) -> str:
        for anc in [sec] + list(sec.parents):
            if getattr(anc, "name", None) == "section":
                sid = anc.get("id", "")
                if sid == "analysis-passes":
                    return "analysis"
                if sid == "transform-passes":
                    return "transform"
        return "unknown"

    def _build_pass_catalog(self, html: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        items: Dict[str, Dict[str, str]] = {}

        for sec in soup.find_all("section"):
            h3 = sec.find("h3")
            code = sec.find("code")
            if not h3 or not code:
                continue

            token_raw = code.get_text(strip=True)
            token = self._normalize(token_raw)
            if not token:
                continue

            scope = self._find_scope(sec)
            h3_text = " ".join(h3.get_text(" ", strip=True).split())
            title = h3_text.replace(token_raw, "").strip().lstrip(":").strip()

            desc_parts = []
            for node in sec.find_all(["p", "li"], recursive=True):
                txt = node.get_text(" ", strip=True)
                if txt and txt != h3_text:
                    desc_parts.append(txt)
            desc = " ".join(" ".join(desc_parts).split())

            cand = {"pass": token, "title": title, "desc": desc, "scope": scope}
            if token not in items:
                items[token] = cand
            else:
                old = items[token]
                if old.get("scope") != "transform" and scope == "transform":
                    items[token] = cand
                elif len(cand.get("desc", "")) > len(old.get("desc", "")):
                    items[token] = cand

        return list(items.values())

    def _get_transform_catalog(self) -> List[Dict[str, str]]:
        log("Fetching Passes.html ...")
        html = self._fetch_passes_html(self.PASSES_URL)
        log("Parsing Passes.html ...")
        catalog = self._build_pass_catalog(html)
        return [p for p in catalog if p.get("scope") == "transform"]

    # ------------------------------------------------------------------
    # TF-IDF ranker
    # ------------------------------------------------------------------

    _DOMAIN_STOPWORDS = {
        "remove", "delete", "eliminate", "cleanup", "clean", "canonicalize",
        "combine", "strengthen", "tighten", "simplify", "merge", "hoist",
        "reuse", "refine", "replace", "keep", "ensure", "enable", "improve",
        "reduce", "avoid", "mark", "add", "set", "use", "make", "perform",
        "prove", "directly", "equivalent", "existing", "associated", "similar",
        "followed", "consecutive", "adjacent", "single", "multiple", "per",
        "case", "write", "writes", "read", "reads", "value", "values",
        "code", "snippet", "snippets", "transformation", "change",
    }

    def _build_ranker(
        self, catalog: List[Dict[str, str]],
    ) -> Tuple[TfidfVectorizer, object, List[str]]:
        tokens = [p["pass"] for p in catalog]
        docs = [
            self._normalize(
                f'{p["pass"]} {p.get("title", "")} {p.get("desc", "")} '
                f'{p.get("scope", "")}'
            )
            for p in catalog
        ]
        stops = list(set(ENGLISH_STOP_WORDS) | self._DOMAIN_STOPWORDS)
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, stop_words=stops)
        X = vec.fit_transform(docs)
        return vec, X, tokens

    def _rank_steps(
        self, step_blocks: List[str],
        vec: TfidfVectorizer, X, pass_tokens: List[str], topk: int,
    ) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for blk in step_blocks:
            transformation, change = self._parse_step_fields(blk)
            query = re.sub(r"\([^)]*\)", "", transformation).strip()
            query = f"{query}. {change}".strip()
            qv = vec.transform([self._normalize(query)])
            sims = cosine_similarity(qv, X)[0]
            scored = sorted(
                enumerate(sims), key=lambda x: x[1], reverse=True,
            )
            for j, _ in scored[:topk]:
                t = pass_tokens[j]
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out

    # ------------------------------------------------------------------
    # ripgrep helpers (for C++ source analysis)
    # ------------------------------------------------------------------

    @staticmethod
    def _require_rg() -> None:
        try:
            p = subprocess.run(
                ["rg", "--version"], capture_output=True, text=True, check=False,
            )
            if p.returncode != 0:
                raise FileNotFoundError
        except FileNotFoundError:
            raise SystemExit("ripgrep (rg) not found — please install it.")

    @staticmethod
    def _rg_list(pattern: str, root: Path, glob: str, fixed: bool = False) -> List[Path]:
        cmd = ["rg", "-l"]
        if fixed:
            cmd.append("-F")
        cmd += ["-g", glob, pattern, str(root)]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode == 0:
            return [Path(l.strip()) for l in (p.stdout or "").splitlines()
                    if l.strip() and Path(l.strip()).exists()]
        return []

    # ------------------------------------------------------------------
    # PassRegistry.def parsing
    # ------------------------------------------------------------------

    def _parse_token_to_passclass(self, path: Path) -> Dict[str, str]:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        pat = re.compile(
            r'^\s*[A-Z_]+_PASS\(\s*"([^"]+)"\s*,\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\(',
            re.MULTILINE,
        )
        return {self._normalize(m.group(1)): m.group(2) for m in pat.finditer(txt)
                if self._normalize(m.group(1)) and m.group(2)}

    def _parse_analysis_class_to_token(self, path: Path) -> Dict[str, str]:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        pat = re.compile(
            r'^\s*[A-Z_]+_(ANALYSIS|ANALYSIS_PASS)\(\s*"([^"]+)"\s*,'
            r'\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\(',
            re.MULTILINE,
        )
        return {m.group(3): self._normalize(m.group(2)) for m in pat.finditer(txt)
                if self._normalize(m.group(2)) and m.group(3)}

    # ------------------------------------------------------------------
    # Analysis dependency resolution
    # ------------------------------------------------------------------

    _ALIAS = {
        "DominatorTree": "DominatorTreeAnalysis",
        "PostDominatorTree": "PostDominatorTreeAnalysis",
        "LoopInfo": "LoopAnalysis",
        "ScalarEvolution": "ScalarEvolutionAnalysis",
        "TargetLibraryInfo": "TargetLibraryAnalysis",
        "TargetTransformInfo": "TargetIRAnalysis",
    }

    def _resolve_analysis_token(
        self, dep_type: str, cls2tok: Dict[str, str],
    ) -> str:
        if dep_type in cls2tok:
            return cls2tok[dep_type]
        alias = self._ALIAS.get(dep_type)
        if alias and alias in cls2tok:
            return cls2tok[alias]
        if dep_type.endswith("Info"):
            cand = dep_type[:-4] + "Analysis"
            if cand in cls2tok:
                return cls2tok[cand]
        if not dep_type.endswith("Analysis"):
            cand = dep_type + "Analysis"
            if cand in cls2tok:
                return cls2tok[cand]
        return ""

    def _find_impl_files(self, pass_class: str, root: Path) -> List[Path]:
        pc = re.escape(pass_class)
        patterns = [
            rf"\b(class|struct)\s+{pc}\b",
            rf"\bPassInfoMixin\s*<\s*{pc}\s*>",
            rf"\b{pc}::run\s*\(",
        ]
        seen: Set[str] = set()
        files: List[Path] = []
        for pat in patterns:
            for f in self._rg_list(pat, root, glob="*.cpp"):
                s = str(f)
                if s not in seen:
                    seen.add(s)
                    files.append(f)
        return files

    _LOOP_STD_DEPS = [
        "DominatorTreeAnalysis", "LoopAnalysis", "ScalarEvolutionAnalysis",
        "TargetLibraryAnalysis", "TargetIRAnalysis", "AssumptionAnalysis",
    ]

    def _extract_analysis_deps(self, text: str) -> List[str]:
        deps: List[str] = []
        for m in re.finditer(r"addRequired(?:Transitive)?\s*<\s*([A-Za-z0-9_:]*)\s*>", text):
            deps.append(m.group(1))
        for m in re.finditer(r"get(?:Cached)?Result\s*<\s*([A-Za-z0-9_:]*)\s*>", text):
            deps.append(m.group(1))

        seen: Set[str] = set()
        out: List[str] = []
        for d in deps:
            if not d or d == "AnalysisT" or "Proxy" in d or "Manager" in d:
                continue
            keep = (
                d.endswith("Analysis") or d.endswith("AnalysisPass")
                or d in ("DominatorTree", "LoopInfo", "ScalarEvolution",
                         "AAResults", "TargetLibraryInfo",
                         "TargetTransformInfo", "PostDominatorTree")
                or (d.endswith("Info") and d != "Info")
                or "Analysis" in d
            )
            if keep and d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _infer_deps_from_files(self, files: List[Path]) -> List[str]:
        seen: Set[str] = set()
        deps: List[str] = []
        for p in files:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for d in self._extract_analysis_deps(txt):
                if d not in seen:
                    seen.add(d)
                    deps.append(d)
            if "LoopStandardAnalysisResults" in txt:
                for d in self._LOOP_STD_DEPS:
                    if d not in seen:
                        seen.add(d)
                        deps.append(d)
        return deps

    # ------------------------------------------------------------------
    # opt probe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_probe_ir() -> Path:
        td = tempfile.mkdtemp(prefix="probe_ll_")
        p = Path(td) / "probe.ll"
        p.write_text(
            'source_filename = "probe.ll"\n'
            "define void @f() {\nentry:\n  ret void\n}\n",
            encoding="utf-8",
        )
        return p

    def _opt_supports(self, pipeline: str, probe_ll: Path, timeout: int = 6) -> bool:
        cmd = [str(self.OPT_BIN), "-disable-output", f"-passes={pipeline}", str(probe_ll)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        text = ((p.stdout or "") + "\n" + (p.stderr or "")).lower()
        if "unknown pass name" in text or "unknown pass" in text:
            return False
        return p.returncode == 0

    # ------------------------------------------------------------------
    # File iteration
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_inputs(in_dir: Path, suffix: str) -> List[Path]:
        files = [
            Path(dp) / fn
            for dp, _, fns in os.walk(in_dir)
            for fn in fns if fn.endswith(suffix)
        ]
        files.sort()
        return files

    @staticmethod
    def _derive_out_path(
        in_path: Path, in_suffix: str, out_suffix: str, out_dir: Optional[Path],
    ) -> Path:
        name = in_path.name
        out_name = (name[:-len(in_suffix)] + out_suffix
                    if name.endswith(in_suffix)
                    else in_path.stem + out_suffix)
        if out_dir is None:
            return in_path.parent / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / out_name

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def map_strategies(
        self,
        in_dir: str,
        out_dir: str = "",
        topk: int = 3,
        emit: str = "tokens",
        max_files: int = 0,
    ) -> str:
        """For each *.model.predict.ll in *in_dir*, map predicted steps to
        LLVM analysis tokens and write *.analysis_passes.txt.
        Returns the output directory as a string."""

        self._require_rg()

        if not self.PASSREGISTRY_DEF.exists():
            raise SystemExit(f"PassRegistry.def not found: {self.PASSREGISTRY_DEF}")
        if not self.LLVM_LIB_ROOT.exists():
            raise SystemExit(f"llvm/lib root not found: {self.LLVM_LIB_ROOT}")

        in_dir_p = Path(in_dir)
        if not in_dir_p.is_dir():
            raise SystemExit(f"in_dir is not a directory: {in_dir_p}")

        out_dir_p = Path(out_dir) if out_dir else None
        if emit in ("print", "both", "json") and not self.OPT_BIN.exists():
            raise SystemExit(f"opt binary not found: {self.OPT_BIN}")

        log("Parsing PassRegistry.def ...")
        tok2cls = self._parse_token_to_passclass(self.PASSREGISTRY_DEF)
        cls2tok = self._parse_analysis_class_to_token(self.PASSREGISTRY_DEF)
        log(f"  token->passclass: {len(tok2cls)}  |  analysis class->token: {len(cls2tok)}")

        log("Building transform TF-IDF ranker ...")
        catalog = self._get_transform_catalog()
        if not catalog:
            raise SystemExit("No transform passes parsed from Passes.html")
        vec, X, pass_tokens = self._build_ranker(catalog)
        log(f"  transform catalog size: {len(pass_tokens)}")

        inputs = self._iter_inputs(in_dir_p, ".model.predict.ll")
        if max_files > 0:
            inputs = inputs[:max_files]
        if not inputs:
            raise SystemExit(f"No *.model.predict.ll found under {in_dir_p}")
        log(f"Found input files: {len(inputs)}")

        # Caches
        cache_cls2cpp: Dict[str, List[Path]] = {}
        cache_cls2deps: Dict[str, List[str]] = {}
        cache_dep2atok: Dict[str, str] = {}
        cache_atok_ok: Dict[str, bool] = {}

        probe_ll: Optional[Path] = None
        if emit in ("print", "both", "json"):
            probe_ll = self._make_probe_ir()
            log(f"Prepared opt probe IR: {probe_ll}")

        for idx, ip in enumerate(inputs, start=1):
            raw = ip.read_text(encoding="utf-8", errors="ignore")
            step_blocks = self._extract_steps(raw)
            predicted = self._rank_steps(step_blocks, vec, X, pass_tokens, topk=topk)

            reg_hit = cpp_total = 0
            per_deps: Set[str] = set()
            per_tokens: Set[str] = set()
            per_prints: Set[str] = set()

            for t in predicted:
                tok = self._normalize(t)
                pcls = tok2cls.get(tok, "")
                if not pcls:
                    continue
                reg_hit += 1
                if pcls not in cache_cls2cpp:
                    cache_cls2cpp[pcls] = self._find_impl_files(pcls, self.LLVM_LIB_ROOT)
                files = cache_cls2cpp[pcls]
                cpp_total += len(files)
                if not files:
                    continue
                if pcls not in cache_cls2deps:
                    cache_cls2deps[pcls] = self._infer_deps_from_files(files)
                per_deps.update(cache_cls2deps[pcls])

            for dep in per_deps:
                if dep not in cache_dep2atok:
                    cache_dep2atok[dep] = self._resolve_analysis_token(dep, cls2tok)
                atok = cache_dep2atok[dep]
                if atok:
                    per_tokens.add(atok)

            if emit in ("print", "both", "json"):
                assert probe_ll is not None
                for atok in per_tokens:
                    if atok not in cache_atok_ok:
                        cache_atok_ok[atok] = self._opt_supports(
                            f"print<{atok}>", probe_ll,
                        )
                    if cache_atok_ok[atok]:
                        per_prints.add(f"print<{atok}>")

            op = self._derive_out_path(
                ip, ".model.predict.ll", ".analysis_passes.txt", out_dir_p,
            )

            if emit == "tokens":
                lines = sorted(per_tokens)
                op.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            elif emit == "print":
                lines = sorted(per_prints)
                op.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            elif emit == "both":
                text = "\n".join(sorted(per_tokens)) + "\n\n" + "\n".join(sorted(per_prints))
                op.write_text(text.rstrip() + "\n", encoding="utf-8")
            else:  # json
                payload = {
                    "input": str(ip),
                    "steps": len(step_blocks),
                    "transform_passes": predicted,
                    "registry_hit_tokens": reg_hit,
                    "cppfiles_total": cpp_total,
                    "analysis_dep_types": sorted(per_deps),
                    "analysis_tokens": sorted(per_tokens),
                    "print_passes": sorted(per_prints),
                }
                op.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            log(
                f"[{idx}/{len(inputs)}] {ip.name}: steps={len(step_blocks)} "
                f"transform={len(predicted)} reg_hit={reg_hit} cpp={cpp_total} "
                f"deps={len(per_deps)} tokens={len(per_tokens)} "
                f"print={len(per_prints)} -> {op.name}"
            )

        log("Done.")
        return str(out_dir_p) if out_dir_p else str(in_dir_p)
