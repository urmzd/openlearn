"""Solution markers with ids and the compiling stubber (DESIGN 5.2).

A marker line holds only comment punctuation and the marker:

    /* SOLUTION-BEGIN L9.1 */      # SOLUTION-BEGIN M04.1      // SOLUTION-END

In course references a region wraps a whole function body. `stub()` replaces
each region whose id is empty or equals `want` with a body that compiles:

    Python  raise NotImplementedError("L9.1")
    Rust    todo!("L9.1")
    Go      panic("todo: L9.1")        (imports left unused become `_` imports)
    C       tl_status -> tl_set_last_error(..); return TL_EUNSUPPORTED;
            pointer   -> ...; return NULL;      void -> abort();
            any other -> ...; return (T){0};   (works for scalars and structs)

The C stub calls tl_set_last_error only when the file includes a tinyllm
header, so non-tinyllm C (primers) stubs still compile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BEGIN = "SOLUTION-BEGIN"
END = "SOLUTION-END"
# Same shape as MARKER_RE in practice/bin/ol.
MARKER_RE = re.compile(
    r"^[^A-Za-z0-9]*("
    + BEGIN
    + r"( [A-Za-z][A-Za-z0-9.+-]*)?|"
    + END
    + r")[^A-Za-z0-9]*$"
)

LANG_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".c": "c",
    ".h": "c",
    ".rs": "rust",
    ".go": "go",
}


def lang_of(path: str) -> str | None:
    for ext, lang in LANG_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return None


@dataclass
class Region:
    begin: int  # 0-based line index of the BEGIN marker
    end: int  # 0-based line index of the END marker
    id: str  # "" when the marker carries no id


class MarkerError(ValueError):
    pass


def _lines(text: str) -> list[str]:
    """Split on newline only. str.splitlines also breaks on U+2028, U+2029,
    U+0085 and other separators, which can sit inside a string literal: a
    stub built from those pieces would not compile."""
    out = text.split("\n")
    if out and out[-1] == "":
        out.pop()
    return [x[:-1] if x.endswith("\r") else x for x in out]


def _lines_keepends(text: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def regions(text: str) -> list[Region]:
    out: list[Region] = []
    open_at: int | None = None
    open_id = ""
    for i, line in enumerate(_lines(text)):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1).startswith(BEGIN):
            if open_at is not None:
                raise MarkerError(
                    f"line {i + 1}: {BEGIN} inside an open region (opened line {open_at + 1})"
                )
            open_at, open_id = i, (m.group(2) or "").strip()
        else:
            if open_at is None:
                raise MarkerError(f"line {i + 1}: {END} without a {BEGIN}")
            out.append(Region(open_at, i, open_id))
            open_at = None
    if open_at is not None:
        raise MarkerError(f"line {open_at + 1}: {BEGIN} is never closed")
    return out


def has_markers(text: str) -> bool:
    return any(MARKER_RE.match(line) for line in _lines(text))


_FUNC_DEF = {
    "python": re.compile(r"^\s*(async\s+)?def\s", re.M),
    "rust": re.compile(r"\bfn\s+\w+[^;{]*\{", re.S),
    "go": re.compile(r"^func\s", re.M),
    "c": re.compile(r"^[A-Za-z_][\w \t\*]*\([^;{]*\)\s*\{", re.M),
}


def defines_functions(text: str, path: str) -> bool:
    """Whether a file has function bodies. A unit without any (a crate root
    that only declares modules, a package doc file) is glue: it needs no
    markers and its stub is itself."""
    pat = _FUNC_DEF.get(lang_of(path) or "")
    return bool(pat and pat.search(text))


def drop_markers(text: str) -> str:
    """The reference as the learner would have written it."""
    keep = [
        line
        for line in _lines_keepends(text)
        if not MARKER_RE.match(line.rstrip("\r\n"))
    ]
    return "".join(keep)


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


# ---------------------------------------------------------------------------
# C signature analysis

_C_DROP = re.compile(
    r"\b(static|inline|extern|__inline__|_Noreturn|register)\b|__attribute__\s*\(\(.*?\)\)"
)


def _c_signature(lines: list[str], begin: int) -> str:
    """The text from the previous declaration boundary to the `{` before begin."""
    buf: list[str] = []
    j = begin - 1
    while j >= 0:
        s = lines[j].strip()
        if j < begin - 1 and (
            not s or s.endswith((";", "}", "*/")) or s.startswith("#")
        ):
            break
        buf.insert(0, lines[j])
        j -= 1
    text = " ".join(buf)
    text = re.sub(r"/\*.*?\*/", " ", text)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def c_return_type(lines: list[str], begin: int) -> str:
    sig = _c_signature(lines, begin)
    head = sig.split("{")[0]
    m = re.match(r"^\s*(.*?)\b([A-Za-z_]\w*)\s*\(", head, re.S)
    if not m:
        return "int"
    rtype = _C_DROP.sub(" ", m.group(1))
    return " ".join(rtype.split()) or "int"


def _c_body(rtype: str, mid: str, pad: str, tinyllm: bool) -> list[str]:
    set_err = [f'{pad}tl_set_last_error("unimplemented: {mid}");'] if tinyllm else []
    if "*" in rtype:
        return set_err + [f"{pad}return NULL;"]
    if rtype == "void":
        return [f"{pad}abort();"]
    if rtype == "tl_status":
        return set_err + [f"{pad}return TL_EUNSUPPORTED;"]
    return set_err + [f"{pad}return ({rtype}){{0}};"]


# ---------------------------------------------------------------------------
# Go unused imports

_GO_IMPORT_BLOCK = re.compile(r"^import\s*\(\s*\n(.*?)^\)", re.S | re.M)
_GO_IMPORT_ONE = re.compile(
    r'^import[ \t]+(?:([A-Za-z_.][\w]*)[ \t]+)?"([^"]+)"[ \t]*$', re.M
)
_GO_SPEC = re.compile(r'^(\s*)(?:([A-Za-z_.][\w]*)\s+)?"([^"]+)"(.*)$')


def _go_pkg_name(path: str) -> str:
    parts = path.split("/")
    last = parts[-1]
    if re.fullmatch(r"v[0-9]+", last) and len(parts) > 1:
        last = parts[-2]
    return re.sub(r"[^A-Za-z0-9_]", "_", last.split(".")[0])


def _go_strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return re.sub(r'"(?:\\.|[^"\\])*"|`[^`]*`', '""', src)


def go_blank_unused_imports(src: str) -> str:
    """Rewrite imports the stubbed code no longer uses as `_` imports, which
    always compile (the package is still linked, so init order is unchanged)."""
    m = _GO_IMPORT_BLOCK.search(src)
    spans: list[tuple[int, int]] = []
    if m:
        spans.append((m.start(), m.end()))
    for one in _GO_IMPORT_ONE.finditer(src):
        spans.append((one.start(), one.end()))
    if not spans:
        return src
    body = src
    for a, b in sorted(spans, reverse=True):
        body = body[:a] + " " * (b - a) + body[b:]
    used_text = _go_strip_comments(body)

    def fix_spec(line: str) -> str:
        s = _GO_SPEC.match(line)
        if not s:
            return line
        lead, alias, path, rest = s.groups()
        if alias in ("_", "."):
            return line
        name = alias or _go_pkg_name(path)
        if re.search(r"\b" + re.escape(name) + r"\.", used_text):
            return line
        return f'{lead}_ "{path}"{rest}'

    out = src
    if m:
        block = m.group(1)
        new_block = "\n".join(fix_spec(x) for x in block.split("\n"))
        out = out[: m.start(1)] + new_block + out[m.end(1) :]
    out = _GO_IMPORT_ONE.sub(
        lambda one: (
            "import "
            + fix_spec(f'{one.group(1) + " " if one.group(1) else ""}"{one.group(2)}"')
        ),
        out,
    )
    return out


# ---------------------------------------------------------------------------
# the stubber


def stub(text: str, path: str, want: str | None = None) -> str:
    """Stub every region whose id is empty or equals `want` (None: every region)."""
    lang = lang_of(path)
    if lang is None:
        raise MarkerError(f"{path}: no stub rule for this file type")
    lines = _lines(text)
    regs = regions(text)
    tinyllm = bool(re.search(r'^\s*#\s*include\s*[<"]tinyllm', text, re.M))
    out: list[str] = []
    cursor = 0
    need_stdlib = need_stddef = False
    for r in regs:
        if want is not None and r.id not in ("", want):
            continue
        out.extend(lines[cursor : r.begin])
        mid = r.id or want or "unimplemented"
        # Indent like the body the region held (C markers often sit at column 0).
        body = [x for x in lines[r.begin + 1 : r.end] if x.strip()]
        pad = _indent(body[0]) if body else _indent(lines[r.begin])
        if lang == "python":
            out.append(f'{pad}raise NotImplementedError("{mid}")')
        elif lang == "rust":
            out.append(f'{pad}todo!("{mid}")')
        elif lang == "go":
            out.append(f'{pad}panic("todo: {mid}")')
        else:
            rtype = c_return_type(lines, r.begin)
            body = _c_body(rtype, mid, pad, tinyllm)
            need_stdlib |= any("abort()" in b for b in body)
            need_stddef |= any("NULL" in b for b in body)
            out.extend(body)
        cursor = r.end + 1
    out.extend(lines[cursor:])
    # Remaining (other-id) marker lines are dropped so the stub reads cleanly.
    result = "\n".join(x for x in out if not MARKER_RE.match(x)) + (
        "\n" if text.endswith("\n") else ""
    )
    if lang == "go":
        result = go_blank_unused_imports(result)
    if lang == "c":
        pre = []
        if need_stdlib and not re.search(r"#\s*include\s*<stdlib\.h>", result):
            pre.append("#include <stdlib.h> /* ol stub: abort */")
        if need_stddef and not re.search(r"#\s*include\s*<std(def|lib|io)\.h>", result):
            pre.append("#include <stddef.h> /* ol stub: NULL */")
        if pre:
            # After the file's own header comment, so it still opens the file.
            at = _after_header_comment(result)
            result = result[:at] + "\n".join(pre) + "\n" + result[at:]
    return result


def _after_header_comment(text: str) -> int:
    """Offset just past a leading `/* ... */` or run of `//` lines, else 0."""
    if text.startswith("/*"):
        end = text.find("*/")
        if end < 0:
            return 0
        nl = text.find("\n", end)
        return len(text) if nl < 0 else nl + 1
    at = 0
    while text.startswith("//", at):
        nl = text.find("\n", at)
        if nl < 0:
            return len(text)
        at = nl + 1
    return at


# ---------------------------------------------------------------------------
# lint: markers wrap whole function bodies only (verify check 5)


def _brace_delta(s: str) -> int:
    s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
    s = re.sub(r"'(?:\\.|[^'\\])'", "''", s)
    s = re.sub(r"//[^\n]*", "", s)
    return s.count("{") - s.count("}")


def lint(text: str, path: str) -> list[str]:
    lang = lang_of(path)
    try:
        regs = regions(text)
    except MarkerError as e:
        return [f"{path}: {e}"]
    lines = _lines(text)
    errs: list[str] = []

    def prev_nonblank(i: int) -> int:
        i -= 1
        while i >= 0 and not lines[i].strip():
            i -= 1
        return i

    def next_nonblank(i: int) -> int:
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        return i

    for r in regs:
        where = f"{path}:{r.begin + 1}"
        body = lines[r.begin + 1 : r.end]
        if lang in ("c", "rust", "go"):
            p = prev_nonblank(r.begin)
            n = next_nonblank(r.end)
            if p < 0 or not lines[p].rstrip().endswith("{"):
                errs.append(
                    f"{where}: region must start right after a function's opening brace"
                )
                continue
            sig = " ".join(lines[max(0, p - 8) : p + 1])
            kw = {"rust": r"\bfn\b", "go": r"\bfunc\b", "c": r"\)\s*\{\s*$"}[lang]
            if not re.search(kw, sig if lang != "c" else lines[p].rstrip()):
                errs.append(
                    f"{where}: the brace before the region does not open a function"
                )
            if n >= len(lines) or not lines[n].strip().startswith("}"):
                errs.append(
                    f"{where}: region must end right before the function's closing brace"
                )
            if sum(_brace_delta(x) for x in body) != 0:
                errs.append(
                    f"{where}: braces inside the region do not balance (it must be the whole body)"
                )
        elif lang == "python":
            pad = len(_indent(lines[r.begin]))
            p = prev_nonblank(r.begin)
            # Skip a docstring between the def and the region.
            if p >= 0 and lines[p].strip().endswith(('"""', "'''")):
                q = p
                quote = lines[p].strip()[-3:]
                if lines[q].strip().count(quote) < 2 or lines[q].strip() == quote:
                    q -= 1
                    while q >= 0 and quote not in lines[q]:
                        q -= 1
                p = prev_nonblank(q)
            header_ok = False
            k = p
            while (
                k >= 0 and k > p - 40
            ):  # a ruff-formatted signature puts one parameter per line
                s = lines[k].strip()
                if re.match(r"(async\s+)?def\s", s) and len(_indent(lines[k])) < pad:
                    header_ok = lines[p].rstrip().endswith(":")
                    break
                k -= 1
            if not header_ok:
                errs.append(f"{where}: region must be the body of a def")
            n = next_nonblank(r.end)
            if n < len(lines) and len(_indent(lines[n])) >= pad:
                errs.append(
                    f"{where}: code after the region is still inside the function body"
                )
            for x in body:
                if x.strip() and len(_indent(x)) < pad:
                    errs.append(
                        f"{where}: a line in the region is less indented than the marker"
                    )
                    break
    return errs
