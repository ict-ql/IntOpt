"""Intrinsic advisor: offline KB construction + online retrieval.

Offline (build_kb):
  1. Parse IntrinsicImpl.inc → extract all intrinsic names
  2. Classify by architecture (x86, aarch64, generic, ...)
  3. Call LLM to get a short description for each intrinsic
  4. Save (name, arch, description) to a JSON file

Online (suggest / batch_suggest):
  1. Summarize the IR's computation patterns via LLM using
     hardware-oriented vocabulary (multiply-add, reduction, SAD, ...)
  2. TF-IDF cosine similarity search against intrinsic descriptions
  3. Return top-k intrinsic suggestions
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from modules.utils import log


# ======================================================================
# CPU feature detection via clang
# ======================================================================

import subprocess


def detect_host_features(clang_bin: str, march: str = "native") -> Tuple[str, str, Set[str]]:
    """Detect CPU features for the given -march target.

    Uses `clang -march=<march> -S -emit-llvm` on a dummy file to extract
    the exact target-cpu and target-features attributes.

    Returns (cpu_name, features_attr_string, feature_set).
    """
    try:
        r = subprocess.run(
            [clang_bin, "-x", "c", "-", "-S", "-emit-llvm",
             "-o", "-", f"-march={march}"],
            input="void f(){}\n",
            capture_output=True, text=True, timeout=30,
        )
        cpu = ""
        features_str = ""
        features = set()

        for line in r.stdout.splitlines():
            if "target-cpu" in line:
                m = re.search(r'"target-cpu"="([^"]+)"', line)
                if m:
                    cpu = m.group(1)
            if "target-features" in line:
                m = re.search(r'"target-features"="([^"]+)"', line)
                if m:
                    features_str = m.group(1)
                    for feat in features_str.split(","):
                        feat = feat.strip()
                        if feat.startswith("+"):
                            features.add(feat[1:])

        return cpu, features_str, features
    except Exception as e:
        log(f"WARN: detect_host_features failed: {e}")
        return "", "", set()


def features_to_attribute_string(cpu: str, features_str: str) -> str:
    """Build the LLVM IR attribute string for target-cpu + target-features.

    Returns a string like:
      "target-cpu"="sapphirerapids" "target-features"="+avx,+avx2,..."
    """
    parts = []
    if cpu:
        parts.append(f'"target-cpu"="{cpu}"')
    if features_str:
        parts.append(f'"target-features"="{features_str}"')
    return " ".join(parts)


# Map feature flags to intrinsic name patterns they enable
# Used to filter intrinsics: if an intrinsic requires a feature not in
# the host's feature set, it gets filtered out.
_FEATURE_TO_INTRINSIC_PREFIX = {
    "amx-int8": ["tdpbssd", "tdpbsud", "tdpbusd", "tdpbuud"],
    "amx-bf16": ["tdpbf16"],
    "amx-fp16": ["tdpfp16"],
    "amx-tile": ["tile", "amx"],
    "avx512f": ["avx512"],
    "avx512bw": ["avx512.mask.pmov", "avx512bw"],
    "avx512vnni": ["vpdpbusd", "vpdpwssd", "vnni"],
    "avx2": ["avx2"],
    "avx": ["avx."],
    "sse4.2": ["sse42", "sse4.2"],
    "sse4.1": ["sse41", "sse4.1"],
    "ssse3": ["ssse3"],
    "sse3": ["sse3"],
    "sse2": ["sse2"],
    "sse": ["sse."],
    "fma": ["fma"],
    "aes": ["aesni", "aes"],
    "sha": ["sha"],
    "pclmul": ["pclmul"],
    "bmi": ["bmi."],
    "bmi2": ["bmi2"],
    "popcnt": ["popcnt"],
    "lzcnt": ["lzcnt"],
}


def intrinsic_requires_unsupported_feature(
    name: str, host_features: Set[str],
) -> bool:
    """Check if an intrinsic requires a feature the host doesn't support.

    Returns True if the intrinsic should be FILTERED OUT."""
    n = name.lower()

    # Generic intrinsics (no arch prefix) are always supported
    parts = name.split(".")
    if len(parts) < 3 or parts[1] not in _KNOWN_TARGETS:
        return False

    # Check AMX specifically (common miss)
    if "amx" in n or "tdp" in n or "tile" in n:
        if "amx-tile" not in host_features:
            return True
        if "tdpbssd" in n or "tdpbsud" in n or "tdpbusd" in n or "tdpbuud" in n:
            if "amx-int8" not in host_features:
                return True
        if "tdpbf16" in n:
            if "amx-bf16" not in host_features:
                return True
        if "tdpfp16" in n:
            if "amx-fp16" not in host_features:
                return True

    # Check AVX-512 variants
    if "avx512" in n and "avx512f" not in host_features:
        return True
    if "avx512vnni" in n and "avx512vnni" not in host_features:
        return True

    # Check AVX2
    if "avx2" in n and "avx2" not in host_features:
        return True

    # Check FMA
    if ".fma" in n and "fma" not in host_features:
        return True

    # Check AES
    if "aesni" in n or "aes" in n:
        if "aes" not in host_features:
            return True

    return False


# ======================================================================
# 1. Parse intrinsic names from IntrinsicImpl.inc
# ======================================================================

# Known target architecture prefixes in LLVM intrinsic names
_KNOWN_TARGETS = {
    "x86", "aarch64", "arm", "amdgcn", "nvvm", "hexagon", "mips",
    "ppc", "riscv", "s390", "ve", "wasm", "bpf", "loongarch",
    "xcore", "r600", "spv", "dx",
}


def parse_intrinsic_names(inc_path: str) -> List[Dict[str, str]]:
    """Parse GET_INTRINSIC_NAME_TABLE from IntrinsicImpl.inc.

    Returns list of {name, arch} dicts.
    arch is 'generic' for target-independent intrinsics."""
    text = Path(inc_path).read_text(encoding="utf-8", errors="ignore")

    # Extract the name table section
    m = re.search(
        r"#ifdef GET_INTRINSIC_NAME_TABLE\s*\n(.*?)#endif",
        text, re.DOTALL,
    )
    if not m:
        raise SystemExit("GET_INTRINSIC_NAME_TABLE not found in " + inc_path)

    entries = []
    for line in m.group(1).splitlines():
        line = line.strip().rstrip(",")
        match = re.match(r'"(llvm\.[^"]+)"', line)
        if not match:
            continue
        name = match.group(1)
        # Classify architecture
        # llvm.<arch>.xxx → arch; llvm.xxx → generic
        parts = name.split(".")
        if len(parts) >= 3 and parts[1] in _KNOWN_TARGETS:
            arch = parts[1]
        else:
            arch = "generic"
        entries.append({"name": name, "arch": arch})

    return entries


def filter_by_arch(
    entries: List[Dict[str, str]],
    archs: Set[str],
) -> List[Dict[str, str]]:
    """Keep only intrinsics for the given architectures + generic."""
    keep = archs | {"generic"}
    return [e for e in entries if e["arch"] in keep]


# ======================================================================
# 2. LLM-based description generation (offline)
# ======================================================================

_DESC_PROMPT_TEMPLATE = """\
You are an LLVM compiler expert. For each LLVM intrinsic below, write a SHORT description \
of what it does and its typical use cases.


Example:
llvm.fma ||| Fused multiply-add in one step; useful in dot products and matrix multiply loops.
llvm.ctpop ||| Counts set bits in an integer; useful in bitwise algorithms and population counting.

Format: one line per intrinsic, exactly:
<name> ||| <short description + use cases>

Intrinsics:
{intrinsic_list}
"""


def generate_descriptions_batch(
    intrinsics: List[Dict[str, str]],
    llm_client,  # LLMClient instance
    model: str = "gpt-5",
    api_mode: str = "auto",
    batch_size: int = 50,
    max_output_tokens: int = 4096,
) -> Dict[str, str]:
    """Call LLM to generate descriptions for intrinsics in batches.

    Returns {intrinsic_name: description}."""
    result: Dict[str, str] = {}
    names = [e["name"] for e in intrinsics]
    total = len(names)

    for i in range(0, total, batch_size):
        batch = names[i:i + batch_size]
        prompt = _DESC_PROMPT_TEMPLATE.format(
            intrinsic_list="\n".join(batch)
        )

        log(f"  Generating descriptions [{i+1}-{min(i+batch_size, total)}/{total}] ...")

        try:
            response = llm_client._call_with_retry(
                prompt_text=prompt,
                model=model,
                max_output_tokens=max_output_tokens,
                temperature=0.3,
                truncation="auto",
                store=False,
                api_mode=api_mode,
                max_retries=2,
                base_backoff=1.0,
            )

            # Parse response: each line is "name ||| description"
            for line in response.splitlines():
                if "|||" not in line:
                    continue
                parts = line.split("|||", 1)
                name = parts[0].strip()
                desc = parts[1].strip()
                # Normalize name (LLM might add @ or quotes)
                name = name.lstrip("@").strip('"').strip("'").strip()
                if name in {n for n in batch}:
                    result[name] = desc
                else:
                    # Fuzzy match: find closest
                    for n in batch:
                        if n in name or name in n:
                            result[n] = desc
                            break

        except Exception as e:
            log(f"  WARN: LLM batch failed: {e}")

        # Small delay between batches
        if i + batch_size < total:
            time.sleep(0.5)

    # Fill missing with a generic description
    for name in names:
        if name not in result:
            result[name] = f"LLVM intrinsic {name}"

    return result


# ======================================================================
# 3. KB persistence (JSON only, no embeddings)
# ======================================================================


def build_kb(
    inc_path: str,
    output_path: str,
    llm_client,
    archs: Set[str] = {"x86", "generic"},
    model: str = "gpt-5",
    api_mode: str = "auto",
    batch_size: int = 50,
    limit: int = 0,
    **kwargs,
) -> str:
    """Build the intrinsic knowledge base (offline), one JSON per arch.

    Output files: <output_dir>/intrinsic.<arch>.json for each arch.
    Returns the output directory."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Parsing intrinsics from {inc_path} ...")
    all_entries = parse_intrinsic_names(inc_path)
    log(f"Total intrinsics: {len(all_entries)}")

    by_arch: Dict[str, List[Dict[str, str]]] = {}
    for e in all_entries:
        if e["arch"] in archs:
            by_arch.setdefault(e["arch"], []).append(e)

    saved_files = []

    for arch, entries in sorted(by_arch.items()):
        log(f"\n=== Building KB for arch: {arch} ({len(entries)} intrinsics) ===")

        if limit > 0:
            entries = entries[:limit]
            log(f"Debug mode: limited to first {limit} intrinsic(s)")

        log("Generating descriptions via LLM ...")
        descriptions = generate_descriptions_batch(
            entries, llm_client,
            model=model, api_mode=api_mode, batch_size=batch_size,
        )
        log(f"Got descriptions for {len(descriptions)} intrinsics")

        kb_entries = []
        for e in entries:
            desc = descriptions.get(e["name"], f"LLVM intrinsic {e['name']}")
            kb_entries.append({
                "name": e["name"],
                "arch": e["arch"],
                "description": desc,
            })

        json_path = output_dir / f"intrinsic.{arch}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(kb_entries, f, ensure_ascii=False, indent=2)

        log(f"Saved: {json_path} ({len(kb_entries)} entries)")
        saved_files.append(str(json_path))

    log(f"\nAll done. Files: {saved_files}")
    return str(output_dir)


# ======================================================================
# 4. Online retrieval: summarize IR → embed → search KB
# ======================================================================

_SUMMARY_PROMPT = """\
Describe the computation patterns in this LLVM IR in 2-3 short sentences.

Only describe patterns that ARE present — never mention what is absent. \
Use these hardware-oriented terms where applicable: \
multiply-add, multiply-accumulate, dot-product, fused multiply-add, \
sum of absolute differences, horizontal add, reduction, accumulate, \
min/max, clamp, compare-and-select, saturating arithmetic, \
absolute value, reciprocal, sqrt, divide, \
bit shift, popcount, leading zero count, trailing zero count, byte swap, \
bitwise AND/OR/XOR, bit extract, bit deposit, \
string compare, byte compare, \
gather, scatter, masked load/store.

State the element data types (i8, i16, i32, i64, float, double) and \
whether there are loops. Do not suggest instructions or analyze feasibility. \
If there are no vectorizable patterns, output only: "no vectorizable patterns"

LLVM IR:
{ir_text}
"""




class IntrinsicAdvisor:
    """Online intrinsic suggestion engine backed by a pre-built KB.

    Matching uses TF-IDF cosine similarity between the IR's
    hardware-oriented computation summary and intrinsic descriptions.
    """

    def __init__(
        self,
        kb_path: str,
        embedding_model_path: str = "",
        host_features: Optional[Set[str]] = None,
        declares_path: str = "",
    ):
        """*kb_path*: directory with intrinsic.<arch>.json files.
        *declares_path*: path to intrinsic_declares.json (declare signatures).
        *host_features*: if provided, filter out unsupported intrinsics.
        *embedding_model_path*: ignored (kept for API compat).
        """
        self.kb_path = kb_path
        self.host_features = host_features
        self.declares_path = declares_path
        self._kb: Optional[List[Dict]] = None
        self._declares: Optional[Dict[str, List[str]]] = None
        # TF-IDF index
        self._tfidf_vec = None
        self._tfidf_matrix = None

    def _load_kb(self):
        if self._kb is not None:
            return

        p = Path(self.kb_path)
        all_entries: List[Dict] = []

        if p.is_dir():
            for json_file in sorted(p.glob("intrinsic.*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                all_entries.extend(entries)
                log(f"Loaded {json_file.name}: {len(entries)} entries")
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                all_entries = json.load(f)
        else:
            raise SystemExit(f"KB path is not a directory or .json file: {p}")

        if not all_entries:
            raise SystemExit(f"No KB entries loaded from {self.kb_path}")

        self._kb = all_entries

        # Build TF-IDF index over intrinsic descriptions
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
        from sklearn.metrics.pairwise import cosine_similarity as _cs

        docs = []
        for e in self._kb:
            # Combine name tokens + description for richer TF-IDF features
            name_tokens = e["name"].replace("llvm.", "").replace(".", " ")
            docs.append(f"{name_tokens} {e.get('description', '')}")

        # Domain stopwords that appear in almost every description
        domain_stops = {
            "used", "useful", "llvm", "intrinsic", "instruction",
            "vector", "packed", "element", "lane", "bit",
            "operation", "operations", "result", "value", "values",
        }
        stops = list(set(ENGLISH_STOP_WORDS) | domain_stops)

        self._tfidf_vec = TfidfVectorizer(
            ngram_range=(1, 3), min_df=1, stop_words=stops,
        )
        self._tfidf_matrix = self._tfidf_vec.fit_transform(docs)

        log(f"Intrinsic KB total: {len(self._kb)} entries, "
            f"TF-IDF matrix: {self._tfidf_matrix.shape}")

    def _load_declares(self):
        """Load intrinsic declare signatures from JSON."""
        if self._declares is not None:
            return
        if not self.declares_path or not Path(self.declares_path).exists():
            if self.declares_path:
                log(f"WARN: declares file not found: {self.declares_path}")
            self._declares = {}
            return
        with open(self.declares_path, "r", encoding="utf-8") as f:
            self._declares = json.load(f)
        log(f"Loaded {len(self._declares)} intrinsic declare signatures")

    def _get_declare(self, intrinsic_name: str) -> str:
        """Look up the declare signature for an intrinsic.

        Returns the declare line, or "OVERLOADED" if the intrinsic is
        valid but overloaded (no fixed signature), or "" if not found."""
        self._load_declares()
        assert self._declares is not None

        def _extract(val) -> str:
            if isinstance(val, str):
                if val == "OVERLOADED":
                    return "OVERLOADED"
                s = val
            elif isinstance(val, list) and val:
                s = max(val, key=lambda x: x.count(","))
            else:
                return ""
            if s and s != "OVERLOADED" and not s.strip().startswith("declare"):
                s = "declare " + s.strip()
            return s

        # Exact match
        if intrinsic_name in self._declares:
            return _extract(self._declares[intrinsic_name])

        # Try progressively shorter base names:
        # llvm.ctlz.i32 → llvm.ctlz (overloaded base)
        parts = intrinsic_name.split(".")
        for i in range(len(parts) - 1, 1, -1):
            base = ".".join(parts[:i])
            if base in self._declares:
                return _extract(self._declares[base])

        return ""



    def _search_kb(
        self, query_text: str, top_k: int,
        arch_filter: Optional[Set[str]] = None,
    ) -> List[Dict[str, str]]:
        """Search KB by TF-IDF cosine similarity."""
        assert self._kb is not None
        assert self._tfidf_vec is not None
        assert self._tfidf_matrix is not None

        # Short-circuit: if the summary says no vectorizable patterns, skip
        if "no vectorizable patterns" in query_text.lower():
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._tfidf_vec.transform([query_text])
        scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        # Filter out compiler-internal / non-compute intrinsics
        _SKIP_PATTERNS = [
            "loop.decrement", "set.loop.iterations", "test.set.loop",
            "test.start.loop", "start.loop.iterations",
            "eh.return", "eh.sjlj", "eh.dwarf", "eh.typeid",
            "coro.", "lifetime.", "invariant.", "dbg.",
            "asan.", "msan.", "tsan.", "hwasan.",
            "memprof.", "instrprof.", "pseudoprobe",
            "donothing", "sideeffect", "experimental",
            ".internal", "annotation", "assume",
            "stacksave", "stackrestore", "stackprotector",
            "get.dynamic.area", "pcmarker", "readcyclecounter",
            "gcread", "gcwrite", "objc.", "swift.",
            "addressofreturnaddress", "returnaddress", "frameaddress",
            "clear_cache", "trap", "debugtrap", "ubsantrap",
            "ptrauth.", "memcpy.inline", "memset.inline",
            "is.constant", "expect", "preserve.struct",
            "preserve.array", "preserve.union",
            "type.test", "type.checked",
        ]
        for i, entry in enumerate(self._kb):
            n = entry["name"].lower()
            if any(p in n for p in _SKIP_PATTERNS):
                scores[i] = -1.0

        # Apply arch + host-feature filters
        if arch_filter:
            keep = arch_filter | {"generic"}
            for i, entry in enumerate(self._kb):
                if entry["arch"] not in keep:
                    scores[i] = -1.0

        if self.host_features is not None:
            for i, entry in enumerate(self._kb):
                if scores[i] < 0:
                    continue
                if intrinsic_requires_unsupported_feature(
                    entry["name"], self.host_features,
                ):
                    scores[i] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            idx = int(idx)
            if scores[idx] <= 0:
                continue
            entry = self._kb[idx]
            results.append({
                "name": entry["name"],
                "arch": entry["arch"],
                "description": entry["description"],
                "score": f"{scores[idx]:.4f}",
            })
        return results

    def suggest(
        self,
        ir_text: str,
        llm_client,
        model: str = "gpt-5",
        api_mode: str = "auto",
        top_k: int = 10,
        arch_filter: Optional[Set[str]] = None,
        cache_dir: str = "",
        cache_key: str = "",
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Suggest intrinsics for a single IR via TF-IDF matching."""
        self._load_kb()

        summary = self._get_or_generate_summary(
            ir_text, llm_client, model, api_mode, cache_dir, cache_key,
        )

        return self._search_kb(summary, top_k, arch_filter)

    def batch_suggest(
        self,
        ir_items: List[Dict[str, str]],
        llm_client,
        model: str = "gpt-5",
        api_mode: str = "auto",
        top_k: int = 10,
        arch_filter: Optional[Set[str]] = None,
        cache_dir: str = "",
        workers: int = 10,
        **kwargs,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Suggest intrinsics for multiple IRs via TF-IDF matching."""
        self._load_kb()

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Phase 1: generate summaries in parallel (with caching)
        summaries: Dict[str, str] = {}
        to_generate: List[Dict[str, str]] = []

        for item in ir_items:
            key = item["key"]
            cached = self._read_cached_summary(cache_dir, key)
            if cached is not None:
                summaries[key] = cached
            else:
                to_generate.append(item)

        if to_generate:
            log(f"Generating {len(to_generate)} IR summaries "
                f"(cached: {len(summaries)}, workers: {workers}) ...")

            def _gen_one(item: Dict[str, str]) -> Tuple[str, str]:
                key = item["key"]
                ir_text = item["ir_text"]
                summary = self._generate_summary(
                    ir_text, llm_client, model, api_mode,
                )
                self._write_cached_summary(cache_dir, key, summary)
                return key, summary

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_gen_one, item): item["key"]
                        for item in to_generate}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    key, summary = fut.result()
                    summaries[key] = summary
                    if done % 20 == 0 or done == len(futs):
                        log(f"  Summaries: [{done}/{len(futs)}]")

        # Phase 2: TF-IDF search for each summary
        results: Dict[str, List[Dict[str, str]]] = {}
        for item in ir_items:
            key = item["key"]
            results[key] = self._search_kb(
                summaries[key], top_k, arch_filter,
            )

        return results

    # ------------------------------------------------------------------
    # Summary generation + caching helpers
    # ------------------------------------------------------------------

    def _generate_summary(
        self, ir_text: str, llm_client, model: str, api_mode: str,
    ) -> str:
        ir_truncated = ir_text[:6000] if len(ir_text) > 6000 else ir_text
        prompt = _SUMMARY_PROMPT.format(ir_text=ir_truncated)
        try:
            summary = llm_client._call_with_retry(
                prompt_text=prompt,
                model=model,
                max_output_tokens=200,
                temperature=0.3,
                truncation="auto",
                store=False,
                api_mode=api_mode,
                max_retries=1,
                base_backoff=1.0,
            )
            return summary if summary and summary.strip() else ir_truncated[:2000]
        except Exception as e:
            log(f"WARN: IR summary failed: {e}")
            return ir_truncated[:2000]

    def _read_cached_summary(self, cache_dir: str, key: str) -> Optional[str]:
        if not cache_dir:
            return None
        p = Path(cache_dir) / f"{key}.summary.txt"
        if p.exists() and p.stat().st_size > 10:
            return p.read_text(encoding="utf-8")
        return None

    def _write_cached_summary(self, cache_dir: str, key: str, summary: str):
        if not cache_dir:
            return
        p = Path(cache_dir) / f"{key}.summary.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(summary, encoding="utf-8")

    def _get_or_generate_summary(
        self, ir_text: str, llm_client, model: str, api_mode: str,
        cache_dir: str = "", cache_key: str = "",
    ) -> str:
        """Get summary from cache or generate it."""
        cached = self._read_cached_summary(cache_dir, cache_key)
        if cached is not None:
            log(f"Using cached summary for {cache_key}")
            return cached
        log("Generating IR summary for intrinsic search ...")
        summary = self._generate_summary(ir_text, llm_client, model, api_mode)
        log(f"IR summary: {summary[:200]}...")
        self._write_cached_summary(cache_dir, cache_key, summary)
        return summary

    def format_suggestions(
        self,
        suggestions: List[Dict[str, str]],
        score_threshold: float = 0.0,
    ) -> str:
        """Format suggestions as text for the refinement prompt.

        *score_threshold*: only include suggestions with score >= this value.
        Signatures are NOT included here — they are attached later in
        the realization prompt by _rewrite_prompts."""
        if not suggestions:
            return ""

        filtered = [
            s for s in suggestions
            if float(s["score"]) >= score_threshold
        ] if score_threshold > 0 else suggestions

        if not filtered:
            return ""

        parts = [
            "Available hardware intrinsics (ranked by relevance):",
            "",
        ]
        for i, s in enumerate(filtered, 1):
            parts.append(
                f"{i}. @{s['name']} [{s['arch']}] (relevance={s['score']})"
            )
            parts.append(f"   {s['description']}")
            parts.append("")

        return "\n".join(parts)

