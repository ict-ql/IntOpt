"""Post-process LLM-generated IR: normalise markdown fences, strip
redundant IR attributes, eliminate trivial bitcasts and SSA copies,
and fix common LLM-generated LLVM IR syntax errors."""

import html
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
# LLM IR syntax fixups
# ---------------------------------------------------------------------------

# Pattern 1: Vector/scalar constant assigned to SSA variable
#   %name = <N x ty> <...>          -> inline the constant at use sites
#   %name = zeroinitializer          -> inline
#   %name = <ty> value               -> inline (e.g. %x = i8 0, %p = ptr %q)
#
# This regex matches lines like:
#   %foo = <16 x i32> <i32 0, i32 1, ...>
#   %foo = zeroinitializer
#   %foo = i8 0
#   %foo = ptr %bar
#   %foo = <i64 0, 1, 2, ...>       (missing vector type prefix)
_VEC_CONST_ASSIGN_RE = re.compile(
    r"""(?mx) ^
    [ \t]*(?P<lhs>%[-a-zA-Z$._0-9]+)\s*=\s*
    (?P<val>
        zeroinitializer                                     # bare zeroinitializer
      | <\d+\s+x\s+[^>]+>\s*<[^>]*>                        # <N x ty> <...>
      | <\d+\s+x\s+[^>]+>\s+zeroinitializer                # <N x ty> zeroinitializer
      | <(?:i\d+|float|double|ptr)\s+[^>]*>                 # <i64 0, 1, 2, ...> (missing type prefix)
      | (?:i\d+|float|double|half|bfloat)\s+-?[\d.eE+inf]+  # i8 0, float 1.0, etc.
      | ptr\s+%[-a-zA-Z$._0-9]+                             # ptr %name (alias)
    )
    \s*$"""
)


def _inline_constant_assignments(ir_text: str) -> str:
    """Replace  %var = <constant>  with inline usage and delete the line.

    Handles vector constants, zeroinitializer, scalar typed constants,
    and ptr aliases that LLMs incorrectly generate as SSA assignments.
    Also handles multi-line vector constants that span continuation lines."""
    # Phase 0: join multi-line vector constant assignments into single lines.
    # Pattern: a line starting with  %name = <...  that doesn't close with >
    # followed by continuation lines until we see the closing >.
    joined_lines: List[str] = []
    lines_raw = ir_text.splitlines(keepends=True)
    i = 0
    while i < len(lines_raw):
        line = lines_raw[i]
        stripped = line.lstrip()
        # Detect  %name = <Nxi type> <...  or  %name = <type val, val, ...
        if (re.match(r"%[-a-zA-Z$._0-9]+\s*=\s*<", stripped)
                and stripped.count("<") > stripped.count(">")
                and not stripped.lstrip().startswith(";")):
            # Accumulate continuation lines
            combined = line.rstrip("\n")
            i += 1
            while i < len(lines_raw):
                cont = lines_raw[i].rstrip("\n")
                combined += " " + cont.strip()
                i += 1
                if ">" in cont and combined.count(">") >= combined.count("<"):
                    break
            joined_lines.append(combined + "\n")
        else:
            joined_lines.append(line)
            i += 1

    ir_text = "".join(joined_lines)

    # Phase 1: match and replace single-line constant assignments
    lines = ir_text.splitlines(keepends=True)
    repl: Dict[str, str] = {}
    delete_idx: Set[int] = set()
    for i, line in enumerate(lines):
        m = _VEC_CONST_ASSIGN_RE.match(line)
        if not m:
            continue
        lhs = m.group("lhs")
        val = m.group("val").strip()
        # For ptr aliases like  %begin = ptr %0  ->  replace %begin with %0
        if val.startswith("ptr "):
            val = val[4:].strip()
        repl[lhs] = val
        delete_idx.add(i)
    if not repl:
        return ir_text
    return _apply_rewrites("".join(lines), repl, delete_idx)


# Pattern 2: Inline constant expressions used as instruction operands
#   select i1 %c, i32 (add i32 %a, %b), i32 %d
#   getelementptr ... i64 (zext i32 %i to i64)
# These need to be split into separate instructions.
# This is complex to fix generically, so we handle the most common cases.

_INLINE_CONSTEXPR_RE = re.compile(
    r"""(?P<ty>i\d+|float|double|ptr)\s+\(
        (?P<op>add|sub|mul|shl|lshr|ashr|and|or|xor|zext|sext|trunc|bitcast)
        \s+(?P<body>[^)]+)\)""",
    re.VERBOSE,
)


def _fix_inline_constexprs(ir_text: str) -> str:
    """Split inline constant expressions into separate instructions.

    Converts patterns like:
      %r = select i1 %c, i32 (add i32 %a, %b), i32 %d
    Into:
      %_fixup_0 = add i32 %a, %b
      %r = select i1 %c, i32 %_fixup_0, i32 %d
    """
    lines = ir_text.splitlines(keepends=True)
    counter = [0]
    changed = True
    max_iters = 20  # safety limit
    while changed and max_iters > 0:
        changed = False
        max_iters -= 1
        new_lines: List[str] = []
        for line in lines:
            m = _INLINE_CONSTEXPR_RE.search(line)
            if m and not line.lstrip().startswith(";"):
                ty = m.group("ty")
                op = m.group("op")
                body = m.group("body").strip()
                tmp = f"%_fixup_{counter[0]}"
                counter[0] += 1
                # Build the extracted instruction
                if op in ("zext", "sext", "trunc", "bitcast"):
                    extracted = f"  {tmp} = {op} {body}\n"
                else:
                    extracted = f"  {tmp} = {op} {ty} {body}\n"
                replacement = f"{ty} {tmp}"
                new_line = line[:m.start()] + replacement + line[m.end():]
                new_lines.append(extracted)
                new_lines.append(new_line)
                changed = True
            else:
                new_lines.append(line)
        lines = new_lines
    return "".join(lines)


# Pattern 3: ptr* (legacy typed pointer syntax)
#   bitcast ptr %x to ptr*  ->  remove the bitcast entirely

def _fix_legacy_ptr_star(ir_text: str) -> str:
    """Remove ptr* references (legacy typed pointer syntax).

    Lines like  %p = bitcast ptr %x to ptr*  become SSA renames."""
    lines = ir_text.splitlines(keepends=True)
    repl: Dict[str, str] = {}
    delete_idx: Set[int] = set()
    ptr_star_re = re.compile(
        r"""(?mx) ^
        \s*(?P<lhs>%[-a-zA-Z$._0-9]+)\s*=\s*bitcast\s+ptr\s+
        (?P<src>%[-a-zA-Z$._0-9]+)\s+to\s+ptr\*\s*$"""
    )
    for i, line in enumerate(lines):
        m = ptr_star_re.match(line)
        if m:
            repl[m.group("lhs")] = m.group("src")
            delete_idx.add(i)
    if not repl:
        return ir_text
    return _apply_rewrites("".join(lines), repl, delete_idx)


# Pattern 4: dereferenceable_or_null / dereferenceable on return type position
#   define ... noalias dereferenceable_or_null(24) ptr @func(...)
# These attributes are not valid on return values in modern LLVM IR.

_DEREF_RETURN_RE = re.compile(
    r"(define\s[^@]*?)\s*dereferenceable(?:_or_null)?\(\d+\)\s*(ptr\s)",
)

_DEREF_RETURN_AFTER_RE = re.compile(
    r"(define\s[^@]*?ptr)\s+dereferenceable(?:_or_null)?\(\d+\)\s*(@)",
)


def _fix_return_deref_attrs(ir_text: str) -> str:
    """Remove dereferenceable(_or_null) from return type position.

    Handles both  define ... dereferenceable_or_null(N) ptr @func
    and           define ... ptr dereferenceable_or_null(N) @func"""
    ir_text = _DEREF_RETURN_RE.sub(r"\1 \2", ir_text)
    ir_text = _DEREF_RETURN_AFTER_RE.sub(r"\1 \2", ir_text)
    return ir_text


# Pattern 5: Missing terminator before label
# If a non-terminator instruction is followed by a label, insert
# an unconditional br to that label.

_TERMINATOR_RE = re.compile(
    r"^\s*(br\s|ret\s|switch\s|unreachable|invoke\s|resume\s|"
    r"indirectbr\s|callbr\s|catchswitch\s|catchret\s|cleanupret\s)",
)
_LABEL_RE = re.compile(r"^([-a-zA-Z$._0-9]+):\s*(?:;.*)?$")


def _fix_missing_terminators(ir_text: str) -> str:
    """Insert br instructions before labels that lack a terminator.

    Detects cases where a non-terminator instruction is immediately
    followed by a label (possibly with blank lines in between) and
    inserts  br label %<label>."""
    lines = ir_text.splitlines(keepends=False)
    result: List[str] = []
    for i, line in enumerate(lines):
        label_m = _LABEL_RE.match(line)
        if label_m and i > 0:
            # Walk backwards past blank lines and comments to find last real line
            prev = ""
            for j in range(i - 1, -1, -1):
                candidate = lines[j].strip()
                if candidate and not candidate.startswith(";"):
                    prev = candidate
                    break
            # Insert br if prev is a real instruction but not a terminator/label/brace
            if (prev and not prev.endswith(":")
                    and not _TERMINATOR_RE.match(prev)
                    and not prev.startswith("}")
                    and not prev.startswith("define ")
                    and not prev.startswith("declare ")):
                label_name = label_m.group(1)
                result.append(f"  br label %{label_name}")
        result.append(line)
    return "\n".join(result) + ("\n" if ir_text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PostProcessor:
    """Apply IR-level cleanups to *.model.predict.ll files in-place."""

    def run(self, in_dir: str, pattern: str = "*.model.predict.ll") -> str:
        """Process all matching files under *in_dir*.  Returns the directory.

        *pattern*: glob pattern for files to process.
        Default is '*.model.predict.ll' (pipeline mode).
        Use '*.optimized.ll' for verify/perf_test mode."""
        in_dir_p = Path(in_dir)
        files = sorted(in_dir_p.rglob(pattern))
        log(f"[PostProcess] {len(files)} files matching '{pattern}'")

        for f in files:
            ctx = f.read_text(encoding="utf-8")
            ctx = _replace_llvm_fences(ctx)
            ctx = _delete_redundant_info(ctx)
            ctx = _delete_bitcast(ctx)
            ctx = _delete_ssa_copies(ctx)
            ctx = _propagate_constants(ctx)
            # LLM IR syntax fixups
            ctx = _inline_constant_assignments(ctx)
            ctx = _fix_inline_constexprs(ctx)
            ctx = _fix_legacy_ptr_star(ctx)
            ctx = _fix_return_deref_attrs(ctx)
            ctx = _fix_missing_terminators(ctx)
            f.write_text(ctx, encoding="utf-8")

        log(f"[PostProcess] Done. Output: {in_dir_p}")
        return str(in_dir_p)
