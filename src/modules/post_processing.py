import argparse
import random
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List
import re
from typing import Dict, Set
import html

def delete_redundant_info(ctx):
    ctx = ctx.replace("dso_local local_unnamed_addr", "dso_local")
    return ctx

_SSA_TOKEN_BOUNDARY = r"[A-Za-z0-9$._-]"

def _resolve_chains(repl: Dict[str, str]) -> Dict[str, str]:
    """Collapse chains: %a->%b, %b->%c => %a->%c (cycle-safe)."""
    def resolve(v: str) -> str:
        seen = set()
        while v in repl and v not in seen:
            seen.add(v)
            v = repl[v]
        return v
    return {k: resolve(v) for k, v in repl.items()}

def _token_sub(line: str, lhs: str, rhs: str) -> str:
    """Replace SSA name lhs with rhs only when lhs is a whole token."""
    pat = re.compile(rf"(?<!{_SSA_TOKEN_BOUNDARY}){re.escape(lhs)}(?!{_SSA_TOKEN_BOUNDARY})")
    return pat.sub(rhs, line)

def apply_rewrites(ir_text: str, repl: Dict[str, str], delete_idx: Set[int]) -> str:
    """
    Apply SSA renames and delete selected lines.
    - repl: mapping like {'%a':'%b'}
    - delete_idx: line indices to delete (0-based)
    """
    if not repl and not delete_idx:
        return ir_text

    lines = ir_text.splitlines(keepends=True)

    # Collapse chains for stable rewriting
    repl = _resolve_chains(repl)

    # Substitute uses (skip lines that will be deleted)
    if repl:
        for i, line in enumerate(lines):
            if i in delete_idx:
                continue
            for lhs, rhs in repl.items():
                line = _token_sub(line, lhs, rhs)
            lines[i] = line

    # Drop deleted lines
    return "".join(line for i, line in enumerate(lines) if i not in delete_idx)

# Match: %lhs = bitcast <ptr-like> %src to <ptr-like>
# ptr-like includes: "ptr", "ptr i8", "ptr addrspace(1)", "ptr addrspace(1) i8", etc.
_BITCAST_PTRLIKE_PTRLIKE_RE = re.compile(
    r"""(?mx) ^
    \s*(?P<lhs>%[-a-zA-Z$._0-9]+)\s*=\s*bitcast
    \s+(?P<src_ptr>ptr(?:\s+addrspace\(\d+\))?(?:\s+[^,%\n]+)?)  # allow multi-token types like "<4 x float>"
    \s+(?P<src>%[-a-zA-Z$._0-9]+)
    \s+to\s+(?P<dst_ptr>ptr(?:\s+addrspace\(\d+\))?(?:\s+[^,\n]+)?)  # allow multi-token types
    \s*(?:,.*)?$
    """
)

_COPY_ASSIGN_RE = re.compile(
    r"""(?mx) ^
    \s*(?P<lhs>%[-a-zA-Z$._0-9]+)\s*=\s*(?P<rhs>%[-a-zA-Z$._0-9]+)
    \s*(?:,.*)?$
    """
)

_ADDRSPACE_RE = re.compile(r"addrspace\((\d+)\)")


LLVM_FENCE_RE = re.compile(
    r"```llvm[ \t]*\r?\n"      # opening fence: ```llvm + optional spaces + newline
    r"(.*?)"                   # code content (non-greedy)
    r"\r?\n```[ \t]*",         # closing fence: newline + ```
    flags=re.DOTALL
)

def replace_llvm_fences(md: str) -> str:
    def repl(m: re.Match) -> str:
        code = m.group(1)
        code_escaped = html.escape(code, quote=False)
        return f"<code>{code_escaped}</code>"
    return LLVM_FENCE_RE.sub(repl, md)

def _addrspace_of(ptr_like: str) -> str:
    m = _ADDRSPACE_RE.search(ptr_like)
    return m.group(1) if m else ""

def delete_bitcast(ir_text: str) -> str:
    """
    Eliminate pointer-to-pointer bitcasts by SSA renaming + deleting the bitcast line.

    Examples:
      %a = bitcast ptr %b to ptr i8
      %vecp = bitcast ptr %0 to ptr <4 x float>

    Conservative rule:
    - Only eliminate when addrspace matches (both none, or same number).
    """
    lines = ir_text.splitlines(keepends=True)

    repl: Dict[str, str] = {}
    delete_idx: Set[int] = set()

    for i, line in enumerate(lines):
        m = _BITCAST_PTRLIKE_PTRLIKE_RE.match(line)
        if not m:
            continue
        lhs = m.group("lhs")
        src = m.group("src")
        src_ptr = m.group("src_ptr")
        dst_ptr = m.group("dst_ptr")

        if _addrspace_of(src_ptr) != _addrspace_of(dst_ptr):
            continue

        repl[lhs] = src
        delete_idx.add(i)

    return apply_rewrites("".join(lines), repl, delete_idx)

def delete_ssa_copies(ir_text: str) -> str:
    """
    Eliminate pseudo-IR SSA copy lines like:
      %cnt.next = %cnt.sel
    by replacing uses of %cnt.next with %cnt.sel and removing the line.
    """
    lines = ir_text.splitlines(keepends=True)

    repl: Dict[str, str] = {}
    delete_idx: Set[int] = set()

    for i, line in enumerate(lines):
        m = _COPY_ASSIGN_RE.match(line)
        if not m:
            continue
        lhs = m.group("lhs")
        rhs = m.group("rhs")
        if lhs == rhs:
            delete_idx.add(i)
            continue
        repl[lhs] = rhs
        delete_idx.add(i)

    return apply_rewrites("".join(lines), repl, delete_idx)


class PostProcessor:
    def __init__(self):
        pass

    def post_process(self, in_dir: str) -> str:
        """对LLM生成的*.model.predict.ll文件进行后处理（原地修改）"""
        in_dir = Path(in_dir)
        model_pred_ll_files = sorted(in_dir.rglob("*.model.predict.ll"))
        print(f"[PostProcess] process files: {len(model_pred_ll_files)}")

        for ll_file in model_pred_ll_files:
            with open(ll_file, "r") as f:
                ctx = f.read()
            ctx = replace_llvm_fences(ctx)
            ctx = delete_redundant_info(ctx)
            ctx = delete_bitcast(ctx)
            ctx = delete_ssa_copies(ctx)

            with open(ll_file, "w") as f:
                f.write(ctx)

        print(f"[PostProcess] Done. Output: {in_dir}")
        return str(in_dir)