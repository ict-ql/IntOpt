"""Intrinsic advisor: offline KB construction + online retrieval.

Offline (build_kb):
  1. Parse IntrinsicImpl.inc → extract all intrinsic names
  2. Classify by architecture (x86, aarch64, generic, ...)
  3. Call LLM to get a short description for each intrinsic
  4. Embed descriptions with a sentence-transformer model
  5. Save (name, arch, description, embedding) to a JSON+npy file

Online (query_kb):
  1. Summarize the IR's semantics via LLM
  2. Embed the summary
  3. Cosine-similarity search against the KB
  4. Return top-k intrinsic suggestions
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
# 3. Embedding + KB persistence
# ======================================================================

def embed_texts(
    texts: List[str],
    model_path: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed a list of texts using a sentence-transformer model.

    Returns (N, dim) float32 numpy array."""
    from sentence_transformers import SentenceTransformer

    log(f"Loading embedding model: {model_path}")
    model = SentenceTransformer(model_path)

    log(f"Embedding {len(texts)} texts ...")
    embeddings = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def build_kb(
    inc_path: str,
    output_path: str,
    llm_client,
    embedding_model: str,
    archs: Set[str] = {"x86", "generic"},
    model: str = "gpt-5",
    api_mode: str = "auto",
    batch_size: int = 50,
    limit: int = 0,
) -> str:
    """Build the intrinsic knowledge base (offline), one file per arch.

    Output files: <output_dir>/intrinsic.<arch>.json + .npy for each arch.

    *limit*: if >0, only process the first N intrinsics per arch (for debugging).

    Returns the output directory."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: parse all intrinsic names
    log(f"Parsing intrinsics from {inc_path} ...")
    all_entries = parse_intrinsic_names(inc_path)
    log(f"Total intrinsics: {len(all_entries)}")

    # Group by arch
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

        # Step 2: generate descriptions
        log("Generating descriptions via LLM ...")
        descriptions = generate_descriptions_batch(
            entries, llm_client,
            model=model, api_mode=api_mode, batch_size=batch_size,
        )
        log(f"Got descriptions for {len(descriptions)} intrinsics")

        # Build KB entries
        kb_entries = []
        desc_texts = []
        for e in entries:
            desc = descriptions.get(e["name"], f"LLVM intrinsic {e['name']}")
            kb_entries.append({
                "name": e["name"],
                "arch": e["arch"],
                "description": desc,
            })
            desc_texts.append(f"{e['name']}: {desc}")

        # Step 3: embed
        embeddings = embed_texts(desc_texts, embedding_model)

        # Step 4: save per-arch files
        json_path = output_dir / f"intrinsic.{arch}.json"
        npy_path = output_dir / f"intrinsic.{arch}.npy"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(kb_entries, f, ensure_ascii=False, indent=2)
        np.save(npy_path, embeddings)

        log(f"Saved: {json_path} ({len(kb_entries)} entries) + {npy_path}")
        saved_files.append(str(json_path))

    log(f"\nAll done. Files: {saved_files}")
    return str(output_dir)


# ======================================================================
# 4. Online retrieval: summarize IR → embed → search KB
# ======================================================================

_SUMMARY_PROMPT = """\
Summarize the following LLVM IR in 2-3 sentences. Focus on:
- What computation it performs (e.g., matrix multiply, sorting, hashing)
- Key data types and sizes (float, double, i32, vectors)
- Dominant patterns (loops, reductions, bit manipulation, memory access patterns)

LLVM IR:
{ir_text}
"""


def _intrinsic_tier_boost(name: str) -> float:
    """Return a multiplicative boost factor based on instruction "tier".

    Higher-tier (more powerful) instructions get a larger boost so they
    rank above simple scalar intrinsics at similar cosine similarity.

    Tier 5 (1.5x): AMX tile operations (tdp*, tile*)
    Tier 4 (1.35x): AVX-512 vector ops (avx512*, 512-bit)
    Tier 3 (1.2x): AVX2 / 256-bit vector ops
    Tier 2 (1.1x): SSE / 128-bit vector ops, VNNI
    Tier 1 (1.0x): generic scalar intrinsics
    Tier 0 (0.7x): internal/debug/experimental intrinsics (demoted)
    """
    n = name.lower()

    # Demote: internal codegen forms, debug, experimental, donothing
    if ".internal" in n or "donothing" in n or "experimental" in n:
        return 0.7
    if "dbg." in n or "lifetime." in n or "invariant." in n:
        return 0.7

    # Tier 5: AMX tile operations
    if "tdp" in n or "tile" in n or "amx" in n:
        return 1.5

    # Tier 4: AVX-512 (512-bit vectors)
    if "avx512" in n or ".512" in n or "512" in n:
        return 1.35

    # Tier 3: AVX2 / 256-bit
    if "avx2" in n or ".256" in n:
        return 1.2

    # Tier 2: SSE4 / VNNI / 128-bit SIMD
    if "sse4" in n or "vnni" in n or "vpdp" in n:
        return 1.1
    if "sse" in n or "ssse" in n or ".128" in n:
        return 1.05

    # Tier 1.5: vector reduce, masked ops (target-independent but powerful)
    if "vector.reduce" in n or "masked." in n or "vp." in n:
        return 1.15

    # Tier 1: everything else (generic scalar)
    return 1.0


class IntrinsicAdvisor:
    """Online intrinsic suggestion engine backed by a pre-built KB."""

    def __init__(
        self,
        kb_path: str,
        embedding_model_path: str,
        host_features: Optional[Set[str]] = None,
    ):
        """*kb_path* can be:
          - A directory containing intrinsic.<arch>.json/npy files
          - A single .json file (legacy format)

        *host_features*: if provided, filter out intrinsics that require
        features not in this set.
        """
        self.kb_path = kb_path
        self.embedding_model_path = embedding_model_path
        self.host_features = host_features
        self._kb: Optional[List[Dict]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embed_model = None

    def _load_kb(self):
        if self._kb is not None:
            return

        p = Path(self.kb_path)
        all_entries: List[Dict] = []
        all_embeddings: List[np.ndarray] = []

        if p.is_dir():
            # Load all intrinsic.<arch>.json + .npy pairs in the directory
            for json_file in sorted(p.glob("intrinsic.*.json")):
                npy_file = json_file.with_suffix(".npy")
                if not npy_file.exists():
                    log(f"WARN: no .npy for {json_file.name}, skipping")
                    continue
                with open(json_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                embs = np.load(str(npy_file))
                all_entries.extend(entries)
                all_embeddings.append(embs)
                log(f"Loaded {json_file.name}: {len(entries)} entries")
        elif p.suffix == ".json":
            # Legacy single-file format
            with open(p, "r", encoding="utf-8") as f:
                all_entries = json.load(f)
            all_embeddings.append(np.load(str(p.with_suffix(".npy"))))
        else:
            raise SystemExit(f"KB path is not a directory or .json file: {p}")

        if not all_entries:
            raise SystemExit(f"No KB entries loaded from {self.kb_path}")

        self._kb = all_entries
        self._embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0))
        log(f"Intrinsic KB total: {len(self._kb)} entries, "
            f"embeddings shape={self._embeddings.shape}")

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.embedding_model_path)
        return self._embed_model

    def _embed_query(self, text: str) -> np.ndarray:
        model = self._get_embed_model()
        emb = model.encode([text], normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)[0]

    def suggest(
        self,
        ir_text: str,
        llm_client,
        model: str = "gpt-5",
        api_mode: str = "auto",
        top_k: int = 10,
        arch_filter: Optional[Set[str]] = None,
    ) -> List[Dict[str, str]]:
        """Suggest intrinsics for the given IR.

        Steps:
          1. Summarize IR via LLM
          2. Embed the summary
          3. Cosine similarity search in KB
          4. Return top-k results

        Returns list of {name, arch, description, score}."""
        self._load_kb()
        assert self._kb is not None and self._embeddings is not None

        # Step 1: summarize IR
        # Truncate IR to avoid token limits
        ir_truncated = ir_text[:6000] if len(ir_text) > 6000 else ir_text
        prompt = _SUMMARY_PROMPT.format(ir_text=ir_truncated)

        log("Generating IR summary for intrinsic search ...")
        try:
            summary = llm_client._call_with_retry(
                prompt_text=prompt,
                model=model,
                max_output_tokens=512,
                temperature=0.3,
                truncation="auto",
                store=False,
                api_mode=api_mode,
                max_retries=1,
                base_backoff=1.0,
            )
        except Exception as e:
            log(f"WARN: IR summary failed: {e}")
            summary = ir_truncated[:2000]

        log(f"IR summary: {summary[:200]}...")

        # Step 2: embed
        query_emb = self._embed_query(summary)

        # Step 3: cosine similarity (embeddings are already normalized)
        scores = self._embeddings @ query_emb  # (N,)

        # Optional arch filter
        if arch_filter:
            keep = arch_filter | {"generic"}
            mask = np.array([
                e["arch"] in keep for e in self._kb
            ], dtype=bool)
            scores = np.where(mask, scores, -1.0)

        # Filter out intrinsics requiring unsupported host features
        if self.host_features is not None:
            for i, entry in enumerate(self._kb):
                if scores[i] < 0:
                    continue
                if intrinsic_requires_unsupported_feature(entry["name"], self.host_features):
                    scores[i] = -1.0

        # Boost advanced instructions: AMX > AVX-512 > AVX2 > SSE > generic scalar
        # This ensures tile/vector intrinsics rank above simple scalar ones
        for i, entry in enumerate(self._kb):
            if scores[i] < 0:
                continue
            scores[i] *= _intrinsic_tier_boost(entry["name"])

        # Step 4: top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            if scores[idx] < 0:
                continue
            entry = self._kb[idx]
            results.append({
                "name": entry["name"],
                "arch": entry["arch"],
                "description": entry["description"],
                "score": f"{scores[idx]:.4f}",
            })

        return results

    def format_suggestions(self, suggestions: List[Dict[str, str]]) -> str:
        """Format suggestions as text for the refinement prompt."""
        if not suggestions:
            return ""

        parts = [
            "The following LLVM intrinsics may be relevant to this code "
            "(retrieved from the intrinsic knowledge base by semantic similarity):",
            "",
        ]
        for i, s in enumerate(suggestions, 1):
            parts.append(
                f"{i}. @{s['name']} [{s['arch']}] (similarity={s['score']})"
            )
            parts.append(f"   {s['description']}")
            parts.append("")

        parts.append(
            "Consider whether any of these intrinsics could replace or "
            "improve existing operations. If you use a target-specific "
            "intrinsic, add the appropriate target-features attribute."
        )
        return "\n".join(parts)

