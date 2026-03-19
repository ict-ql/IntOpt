"""Post-process LLM-generated IR: normalise markdown fences, strip
redundant IR attributes, eliminate trivial bitcasts and SSA copies."""

import html
import re
from pathlib import Path
from typing import Dict, Set

from modules.utils import log

# ---------------------------------------------------------------------------
# SSA rewrite engine
# ---------------------------------------------------------------------------

_SSA_TOKEN_BOUNDARY = r"[A-Za-z0-9$._-]"


def _resolve_chains(repl: Dict[str, str]) -> Dict[str, str]:
    """Collapse chains: %a->%b, %b->%c  =>  %a->%c  (cycle-safe)."""
    def _resolve(v: str) -> str:
        seen: Set[str] = set()
        while v in repl and v not in seen:
            seen.add(v)
            v = repl[v]
        return v
    return {k: _resolve(v) for k, v in repl.items()}


def _token_sub(line: str, lhs: str, rhs: str) -> str:
    pat = re.compile(
        rf"(?<!{_SSA_TOKEN_BOUNDARY}){re.escape(lhs)}(?!{_SSA_TOKEN_BOUNDARY})"
    )
    return pat.sub(rhs, line)


def _apply_rewrites(ir_text: str, repl: Dict[str, str], delete_idx: Set[int]) -> str:
    if not repl and not delete_idx:
        return ir_text
    lines = ir_text.splitlines(keepends=True)
    repl = _resolve_chains(repl)
    if repl:
        for i, line in enumerate(lines):
            if i in delete_idx:
                continue
            for lhs, rhs in repl.items():
                line = _token_sub(line, lhs, rhs)
            lines[i] = line
    return "".join(line for i, line in enumerate(lines) if i not in delete_idx)


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def _delete_redundant_info(ctx: str) -> str:
    return ctx.replace("dso_local local_unnamed_addr", "dso_local")


_LLVM_FENCE_RE = re.compile(
    r"```llvm[ \t]*\r?\n(.*?)\r?\n```[ \t]*", flags=re.DOTALL,
)

def _replace_llvm_fences(md: str) -> str:
    """Convert ```llvm ... ``` markdown fences to <code>…</code> tags."""
    def _repl(m: re.Match) -> str:
        return f"<code>{html.escape(m.group(1), quote=False)}</code>"
    return _LLVM_FENCE_RE.sub(_repl, md)


_BITCAST_RE = re.compile(
    r"""(?mx) ^
    \s*(?P<lhs>%[-a-zA-Z$._0-9]+)\s*=\s*bitcast
    \s+(?P<src_ptr>ptr(?:\s+addrspace\(\d+\))?(?:\s+[^,%\n]+)?)
    \s+(?P<src>%[-a-zA-Z$._0-9]+)
    \s+to\s+(?P<dst_ptr>ptr(?:\s+addrspace\(\d+\))?(?:\s+[^,\n]+)?)
    \s*(?:,.*)?$"""
)

_ADDRSPACE_RE = re.compile(r"addrspace\((\d+)\)")

def _addrspace_of(ptr_like: str) -> str:
    m = _ADDRSPACE_RE.search(ptr_like)
    return m.group(1) if m else ""


def _delete_bitcast(ir_text: str) -> str:
    """Eliminate ptr-to-ptr bitcasts (same addrspace) via SSA rename."""
    lines = ir_text.splitlines(keepends=True)
    repl: Dict[str, str] = {}
    delete_idx: Set[int] = set()
    for i, line in enumerate(lines):
        m = _BITCAST_RE.match(line)
        if not m:
            continue
        if _addrspace_of(m.group("src_ptr")) != _addrspace_of(m.group("dst_ptr")):
            continue
        repl[m.group("lhs")] = m.group("src")
        delete_idx.add(i)
    return _apply_rewrites("".join(lines), repl, delete_idx)


_COPY_RE = re.compile(
    r"""(?mx) ^
    \s*(?P<lhs>%[-a-zA-Z$._0-9]+)\s*=\s*(?P<rhs>%[-a-zA-Z$._0-9]+)
    \s*(?:,.*)?$"""
)

def _delete_ssa_copies(ir_text: str) -> str:
    """Eliminate trivial SSA copies like  %a = %b."""
    lines = ir_text.splitlines(keepends=True)
    repl: Dict[str, str] = {}
    delete_idx: Set[int] = set()
    for i, line in enumerate(lines):
        m = _COPY_RE.match(line)
        if not m:
            continue
        lhs, rhs = m.group("lhs"), m.group("rhs")
        if lhs == rhs:
            delete_idx.add(i)
        else:
            repl[lhs] = rhs
            delete_idx.add(i)
    return _apply_rewrites("".join(lines), repl, delete_idx)


# Match  %var = <integer_literal>  (possibly negative)
_CONST_ASSIGN_RE = re.compile(
    r"""(?mx) ^
    \s*(?P<lhs>%[-a-zA-Z$._0-9]+)\s*=\s*(?P<val>-?\d+)
    \s*(?:,.*)?$"""
)

def _propagate_constants(ir_text: str) -> str:
    """Propagate constant assignments like  %one = 72340172838076673.

    Replaces all uses of %one with the literal value and removes the
    assignment line."""
    lines = ir_text.splitlines(keepends=True)
    repl: Dict[str, str] = {}
    delete_idx: Set[int] = set()
    for i, line in enumerate(lines):
        m = _CONST_ASSIGN_RE.match(line)
        if not m:
            continue
        repl[m.group("lhs")] = m.group("val")
        delete_idx.add(i)
    return _apply_rewrites("".join(lines), repl, delete_idx)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PostProcessor:
    """Apply IR-level cleanups to *.model.predict.ll files in-place."""

    def run(self, in_dir: str) -> str:
        """Process all prediction files under *in_dir*.  Returns the directory."""
        in_dir_p = Path(in_dir)
        files = sorted(in_dir_p.rglob("*.model.predict.ll"))
        log(f"[PostProcess] {len(files)} files to process")

        for f in files:
            ctx = f.read_text(encoding="utf-8")
            ctx = _replace_llvm_fences(ctx)
            ctx = _delete_redundant_info(ctx)
            ctx = _delete_bitcast(ctx)
            ctx = _delete_ssa_copies(ctx)
            ctx = _propagate_constants(ctx)
            f.write_text(ctx, encoding="utf-8")

        log(f"[PostProcess] Done. Output: {in_dir_p}")
        return str(in_dir_p)
