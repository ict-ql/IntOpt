import re
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

class StrategyMapping:
    def __init__(self, passregistry_def: str, llvm_lib_root: str, opt_bin: str):
        self.PASSREGISTRY_DEF = Path(passregistry_def)
        self.LLVM_LIB_ROOT = Path(llvm_lib_root)
        self.OPT_BIN = Path(opt_bin)
        self.PASSES_URL = "https://llvm.org/docs/Passes.html"
    
    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
    
    def die(self, msg: str) -> None:
        raise SystemExit(msg)
    
    def normalize(self, text: str) -> str:
        text = (text or "").lower()
        text = text.replace("llvm.", "llvm ")
        text = re.sub(r"[^a-z0-9_\-\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def extract_steps(self, raw: str) -> List[str]:
        STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL | re.IGNORECASE)
        blocks = [b.strip() for b in STEP_RE.findall(raw)]
        return [b for b in blocks if self.normalize(b)]
    
    def parse_step_fields(self, step_block: str) -> Tuple[str, str]:
        TRANS_RE = re.compile(r"\*\*Transformation\*\*:\s*(.+)", re.IGNORECASE)
        CHANGE_RE = re.compile(r"\*\*Change\*\*:\s*(.+)", re.IGNORECASE | re.DOTALL)
        m_t = TRANS_RE.search(step_block)
        transformation = m_t.group(1).strip() if m_t else ""
        m_c = CHANGE_RE.search(step_block)
        change = m_c.group(1).strip() if m_c else ""
        return transformation, change
    
    def fetch_passes_html(self, url: str) -> str:
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
    
    def build_pass_catalog(self, html: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        items: Dict[str, Dict[str, str]] = {}

        for sec in soup.find_all("section"):
            h3 = sec.find("h3")
            code = sec.find("code")
            if not h3 or not code:
                continue

            token_raw = code.get_text(strip=True)
            token = self.normalize(token_raw)
            if not token:
                continue

            scope = self._find_scope(sec)

            h3_text = " ".join(h3.get_text(" ", strip=True).split())
            title = h3_text.replace(token_raw, "").strip().lstrip(":").strip()

            desc_parts = []
            for node in sec.find_all(["p", "li"], recursive=True):
                txt = node.get_text(" ", strip=True)
                if not txt:
                    continue
                if h3_text and txt == h3_text:
                    continue
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
    
    def get_transform_catalog(self, cache_path: str = "", refresh: bool = False) -> List[Dict[str, str]]:
        if cache_path and (not refresh):
            try:
                cached = [json.loads(line) for line in Path(cache_path).read_text(encoding="utf-8").splitlines() if line.strip()]
                if cached and all("pass" in x and "scope" in x for x in cached):
                    out = [p for p in cached if p.get("scope") == "transform"]
                    if out:
                        return out
            except FileNotFoundError:
                pass

        self.log("Fetching Passes.html ...")
        html = self.fetch_passes_html(self.PASSES_URL)
        self.log("Parsing Passes.html ...")
        catalog = self.build_pass_catalog(html)

        if cache_path:
            Path(cache_path).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in catalog) + "\n", encoding="utf-8")
            self.log(f"Cached pass catalog -> {cache_path}")

        return [p for p in catalog if p.get("scope") == "transform"]
    
    def build_step_pass_ranker(self, transform_catalog: List[Dict[str, str]]) -> Tuple[TfidfVectorizer, object, List[str]]:
        DOMAIN_GENERIC_STOPWORDS = {
            "remove", "delete", "eliminate", "cleanup", "clean", "canonicalize", "combine",
            "strengthen", "tighten", "simplify", "merge", "hoist", "reuse", "refine",
            "replace", "keep", "ensure", "enable", "improve", "reduce", "avoid",
            "mark", "add", "set", "use", "make", "perform", "prove",
            "directly", "equivalent", "existing", "associated", "similar", "followed",
            "consecutive", "adjacent", "single", "multiple", "per", "case",
            "write", "writes", "read", "reads", "value", "values",
            "code", "snippet", "snippets",
            "transformation", "change",
        }
        
        pass_tokens = [p["pass"] for p in transform_catalog]
        pass_docs = [
            self.normalize(f'{p["pass"]} {p.get("title","" )} {p.get("desc","" )} {p.get("scope","" )}')
            for p in transform_catalog
        ]
        stopwords = set(ENGLISH_STOP_WORDS) | set(DOMAIN_GENERIC_STOPWORDS)
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, stop_words=list(stopwords))
        X = vec.fit_transform(pass_docs)
        return vec, X, pass_tokens
    
    def map_steps_to_transform_passes(self, step_blocks: List[str], vec: TfidfVectorizer, X, pass_tokens: List[str], topk: int) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for blk in step_blocks:
            transformation, change = self.parse_step_fields(blk)
            transformation_clean = re.sub(r"\([^)]*\)", "", transformation).strip()
            query_raw = f"{transformation_clean}. {change}".strip()

            qv = vec.transform([self.normalize(query_raw)])
            sims = cosine_similarity(qv, X)[0]
            scored = [(pass_tokens[j], float(sims[j])) for j in range(len(pass_tokens))]
            scored.sort(key=lambda x: x[1], reverse=True)
            for t, _ in scored[:topk]:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out
    
    def require_rg(self) -> None:
        try:
            p = subprocess.run(["rg", "--version"], capture_output=True, text=True, check=False)
            if p.returncode != 0:
                raise FileNotFoundError
        except FileNotFoundError:
            self.die("ripgrep (rg) not found. Please install rg or add it to PATH.")
    
    def rg_list_files_fixed(self, needle: str, root: Path, glob: str) -> List[Path]:
        cmd = ["rg", "-l", "-F", "-g", glob, needle, str(root)]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode == 0:
            files = [Path(line.strip()) for line in (p.stdout or "").splitlines() if line.strip()]
            return [f for f in files if f.exists()]
        if p.returncode == 1:
            return []
        return []
    
    def rg_list_files_regex(self, pattern: str, root: Path, glob: str) -> List[Path]:
        cmd = ["rg", "-l", "-g", glob, pattern, str(root)]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode == 0:
            files = [Path(line.strip()) for line in (p.stdout or "").splitlines() if line.strip()]
            return [f for f in files if f.exists()]
        if p.returncode == 1:
            return []
        return []
    
    def parse_passregistry_token_to_passclass(self, passregistry_def: Path) -> Dict[str, str]:
        txt = passregistry_def.read_text(encoding="utf-8", errors="ignore")
        line_re = re.compile(
            r'^\s*[A-Z_]+_PASS\(\s*"([^"]+)"\s*,\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\(',
            re.MULTILINE,
        )
        out: Dict[str, str] = {}
        for m in line_re.finditer(txt):
            token = self.normalize(m.group(1))
            cls = m.group(2)
            if token and cls:
                out[token] = cls
        return out
    
    def parse_analysis_class_to_token(self, passregistry_def: Path) -> Dict[str, str]:
        txt = passregistry_def.read_text(encoding="utf-8", errors="ignore")
        line_re = re.compile(
            r'^\s*[A-Z_]+_(ANALYSIS|ANALYSIS_PASS)\(\s*"([^"]+)"\s*,\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\(',
            re.MULTILINE,
        )
        out: Dict[str, str] = {}
        for m in line_re.finditer(txt):
            tok = self.normalize(m.group(2))
            cls = m.group(3)
            if tok and cls:
                out[cls] = tok
        return out
    
    def resolve_analysis_token(self, dep_type: str, analysis_class2tok: Dict[str, str]) -> str:
        if dep_type in analysis_class2tok:
            return analysis_class2tok[dep_type]
        alias = {
            "DominatorTree": "DominatorTreeAnalysis",
            "PostDominatorTree": "PostDominatorTreeAnalysis",
            "LoopInfo": "LoopAnalysis",
            "ScalarEvolution": "ScalarEvolutionAnalysis",
            "TargetLibraryInfo": "TargetLibraryAnalysis",
            "TargetTransformInfo": "TargetIRAnalysis",
        }
        if dep_type in alias and alias[dep_type] in analysis_class2tok:
            return analysis_class2tok[alias[dep_type]]
        if dep_type.endswith("Info"):
            cand = dep_type[:-4] + "Analysis"
            if cand in analysis_class2tok:
                return analysis_class2tok[cand]
        if not dep_type.endswith("Analysis"):
            cand = dep_type + "Analysis"
            if cand in analysis_class2tok:
                return analysis_class2tok[cand]
        return ""
    
    def find_pass_class_impl_files(self, pass_class: str, root: Path) -> List[Path]:
        pc = re.escape(pass_class)
        patterns = [
            rf"\b(class|struct)\s+{pc}\b",
            rf"\bPassInfoMixin\s*<\s*{pc}\s*>",
            rf"\b{pc}::run\s*\(",
        ]
        files: List[Path] = []
        seen = set()
        for pat in patterns:
            for f in self.rg_list_files_regex(pat, root, glob="*.cpp"):
                s = str(f)
                if s not in seen:
                    seen.add(s)
                    files.append(f)
        return files
    
    def extract_analysis_deps_from_cpp(self, text: str) -> List[str]:
        deps: List[str] = []
        for m in re.finditer(r"addRequired(?:Transitive)?\s*<\s*([A-Za-z0-9_:]*)\s*>", text):
            deps.append(m.group(1))
        for m in re.finditer(r"get(?:Cached)?Result\s*<\s*([A-Za-z0-9_:]*)\s*>", text):
            deps.append(m.group(1))

        filtered: List[str] = []
        seen = set()
        for d in deps:
            if not d or d == "AnalysisT":
                continue
            if "Proxy" in d or "Manager" in d:
                continue
            keep = (
                d.endswith("Analysis")
                or d.endswith("AnalysisPass")
                or d in ("DominatorTree", "LoopInfo", "ScalarEvolution", "AAResults",
                         "TargetLibraryInfo", "TargetTransformInfo", "PostDominatorTree")
                or (d.endswith("Info") and d != "Info")
                or ("Analysis" in d)
            )
            if keep and d not in seen:
                seen.add(d)
                filtered.append(d)
        return filtered
    
    def infer_analysis_deps_from_files(self, files: List[Path]) -> List[str]:
        deps: List[str] = []
        seen = set()
        for p in files:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for d in self.extract_analysis_deps_from_cpp(txt):
                if d not in seen:
                    seen.add(d)
                    deps.append(d)

            if "LoopStandardAnalysisResults" in txt:
                for d in [
                    "DominatorTreeAnalysis",
                    "LoopAnalysis",
                    "ScalarEvolutionAnalysis",
                    "TargetLibraryAnalysis",
                    "TargetIRAnalysis",
                    "AssumptionAnalysis",
                ]:
                    if d not in seen:
                        seen.add(d)
                        deps.append(d)
        return deps
    
    def make_probe_ir(self) -> Path:
        td = tempfile.mkdtemp(prefix="probe_ll_")
        p = Path(td) / "probe.ll"
        p.write_text(
            "source_filename = \"probe.ll\"\n"
            "define void @f() {\n"
            "entry:\n"
            "  ret void\n"
            "}\n",
            encoding="utf-8",
        )
        return p
    
    def opt_supports_pipeline(self, opt_bin: Path, pipeline: str, probe_ll: Path, timeout_sec: int = 6) -> bool:
        cmd = [str(opt_bin), "-disable-output", f"-passes={pipeline}", str(probe_ll)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return False

        text = ((p.stdout or "") + "\n" + (p.stderr or "")).lower()
        if "unknown pass name" in text or "unknown pass" in text:
            return False
        if p.returncode != 0:
            return False
        return True
    
    def iter_inputs(self, in_dir: Path, in_suffix: str) -> List[Path]:
        files: List[Path] = []
        for dp, _, fns in os.walk(in_dir):
            for fn in fns:
                if fn.endswith(in_suffix):
                    files.append(Path(dp) / fn)
        files.sort()
        return files
    
    def derive_out_path(self, in_path: Path, in_suffix: str, out_suffix: str, out_dir: Optional[Path]) -> Path:
        name = in_path.name
        if name.endswith(in_suffix):
            out_name = name[:-len(in_suffix)] + out_suffix
        else:
            out_name = in_path.stem + out_suffix
        if out_dir is None:
            return in_path.parent / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / out_name
    
    def map_strategies_to_passes(self, in_dir: str, out_dir: str = "", topk: int = 3, emit: str = "tokens", max_files: int = 0):
        import json
        
        self.require_rg()

        if not self.PASSREGISTRY_DEF.exists():
            self.die(f"PassRegistry.def not found: {self.PASSREGISTRY_DEF}")
        if not self.LLVM_LIB_ROOT.exists():
            self.die(f"llvm/lib root not found: {self.LLVM_LIB_ROOT}")

        in_dir = Path(in_dir)
        if not in_dir.is_dir():
            self.die(f"--in_dir is not a directory: {in_dir}")

        out_dir = Path(out_dir) if out_dir else None
        if emit in ("print", "both", "json") and (not self.OPT_BIN.exists()):
            self.die(f"opt binary not found: {self.OPT_BIN}")

        self.log("Parsing PassRegistry.def ...")
        token2passclass = self.parse_passregistry_token_to_passclass(self.PASSREGISTRY_DEF)
        analysis_class2tok = self.parse_analysis_class_to_token(self.PASSREGISTRY_DEF)
        self.log(f"  token->passclass: {len(token2passclass)}")
        self.log(f"  analysis class->token: {len(analysis_class2tok)}")

        self.log("Building transform TF-IDF ranker ...")
        transform_catalog = self.get_transform_catalog()
        if not transform_catalog:
            self.die("No transform passes parsed from Passes.html")
        vec, X, pass_tokens = self.build_step_pass_ranker(transform_catalog)
        self.log(f"  transform catalog size: {len(pass_tokens)}")

        inputs = self.iter_inputs(in_dir, ".model.predict.ll")
        if max_files > 0:
            inputs = inputs[:max_files]
        if not inputs:
            self.die(f"No inputs found under {in_dir} with suffix .model.predict.ll")
        self.log(f"Found input files: {len(inputs)}")

        cache_passclass_to_cppfiles: Dict[str, List[Path]] = {}
        cache_passclass_to_deps: Dict[str, List[str]] = {}
        cache_deptype_to_atok: Dict[str, str] = {}
        cache_atok_to_print_ok: Dict[str, bool] = {}

        probe_ll: Optional[Path] = None
        if emit in ("print", "both", "json"):
            probe_ll = self.make_probe_ir()
            self.log(f"Prepared opt probe IR: {probe_ll}")

        for idx, ip in enumerate(inputs, start=1):
            raw = ip.read_text(encoding="utf-8", errors="ignore")
            step_blocks = self.extract_steps(raw)
            predicted = self.map_steps_to_transform_passes(step_blocks, vec, X, pass_tokens, topk=topk)

            reg_hit = 0
            cpp_total = 0

            per_file_deps: Set[str] = set()
            per_file_tokens: Set[str] = set()
            per_file_prints: Set[str] = set()

            for t in predicted:
                tok = self.normalize(t)
                pass_class = token2passclass.get(tok, "")
                if not pass_class:
                    continue
                reg_hit += 1

                if pass_class not in cache_passclass_to_cppfiles:
                    cache_passclass_to_cppfiles[pass_class] = self.find_pass_class_impl_files(pass_class, self.LLVM_LIB_ROOT)
                files = cache_passclass_to_cppfiles[pass_class]
                cpp_total += len(files)
                if not files:
                    continue

                if pass_class not in cache_passclass_to_deps:
                    cache_passclass_to_deps[pass_class] = self.infer_analysis_deps_from_files(files)
                for d in cache_passclass_to_deps[pass_class]:
                    per_file_deps.add(d)

            for dep_type in per_file_deps:
                if dep_type not in cache_deptype_to_atok:
                    cache_deptype_to_atok[dep_type] = self.resolve_analysis_token(dep_type, analysis_class2tok)
                atok = cache_deptype_to_atok[dep_type]
                if atok:
                    per_file_tokens.add(atok)

            if emit in ("print", "both", "json"):
                assert probe_ll is not None
                for atok in per_file_tokens:
                    if atok not in cache_atok_to_print_ok:
                        pipeline = f"print<{atok}>"
                        ok = self.opt_supports_pipeline(self.OPT_BIN, pipeline, probe_ll, timeout_sec=6)
                        cache_atok_to_print_ok[atok] = ok
                    if cache_atok_to_print_ok[atok]:
                        per_file_prints.add(f"print<{atok}>")

            op = self.derive_out_path(ip, ".model.predict.ll", ".analysis_passes.txt", out_dir)

            if emit == "tokens":
                out_lines = sorted(per_file_tokens)
                op.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            elif emit == "print":
                out_lines = sorted(per_file_prints)
                op.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            elif emit == "both":
                tok_lines = sorted(per_file_tokens)
                pr_lines = sorted(per_file_prints)
                text = "\n".join(tok_lines) + "\n\n" + "\n".join(pr_lines)
                if not text.endswith("\n"):
                    text += "\n"
                op.write_text(text, encoding="utf-8")
            else:
                payload = {
                    "input": str(ip),
                    "steps": len(step_blocks),
                    "transform_passes": predicted,
                    "registry_hit_tokens": reg_hit,
                    "cppfiles_total": cpp_total,
                    "analysis_dep_types": sorted(per_file_deps),
                    "analysis_tokens": sorted(per_file_tokens),
                    "print_passes": sorted(per_file_prints),
                }
                op.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            self.log(f"[{idx}/{len(inputs)}] {ip.name}: steps={len(step_blocks)} transform={len(predicted)} "
                    f"reg_hit={reg_hit} cpp={cpp_total} deps={len(per_file_deps)} tokens={len(per_file_tokens)} "
                    f"print={len(per_file_prints)} -> {op.name}")

        self.log("Done.")
        return out_dir
