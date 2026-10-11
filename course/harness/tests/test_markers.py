import shutil
import subprocess
from pathlib import Path

import pytest

from olcourse import markers

C_SRC = """\
#include "tinyllm/abi.h"
typedef struct { int a; } pair;

tl_status tl_x(int n) {
/* SOLUTION-BEGIN rt.01 */
    return n ? TL_OK : TL_EINVAL;
/* SOLUTION-END */
}

static inline const char *
tl_name(void) {
/* SOLUTION-BEGIN rt.01 */
    return "x";
/* SOLUTION-END */
}

void tl_drop(void *p) {
/* SOLUTION-BEGIN rt.01 */
    (void)p;
/* SOLUTION-END */
}

pair tl_pair(void) {
/* SOLUTION-BEGIN rt.01 */
    pair p = {1};
    return p;
/* SOLUTION-END */
}

uint32_t tl_version(void) {
/* SOLUTION-BEGIN rt.01 */
    return 1;
/* SOLUTION-END */
}
"""

ABI_H = """\
#include <stdint.h>
typedef int32_t tl_status;
enum { TL_OK = 0, TL_EINVAL = 1, TL_EUNSUPPORTED = 9 };
void tl_set_last_error(const char *msg);
"""


def test_regions_and_ids():
    regs = markers.regions(C_SRC)
    assert [r.id for r in regs] == ["rt.01"] * 5
    assert markers.MARKER_RE.match("    # SOLUTION-BEGIN S-M07a")
    assert markers.MARKER_RE.match("// SOLUTION-END")
    assert not markers.MARKER_RE.match("# the SOLUTION-BEGIN marker wraps bodies")


def test_unbalanced_markers():
    with pytest.raises(markers.MarkerError):
        markers.regions("# SOLUTION-BEGIN a.01\n# SOLUTION-BEGIN a.01\n")
    with pytest.raises(markers.MarkerError):
        markers.regions("# SOLUTION-END\n")


def test_c_stub_bodies_and_compile(tmp_path):
    out = markers.stub(C_SRC, "c/src/x.c", "rt.01")
    assert "return TL_EUNSUPPORTED;" in out
    assert "return NULL;" in out and "#include <stddef.h>" in out
    assert "abort();" in out and "#include <stdlib.h>" in out
    assert "return (pair){0};" in out
    assert "return (uint32_t){0};" in out
    assert out.count('tl_set_last_error("unimplemented: rt.01");') == 4
    (tmp_path / "tinyllm").mkdir()
    (tmp_path / "tinyllm" / "abi.h").write_text(ABI_H)
    (tmp_path / "x.c").write_text(out)
    r = subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-I",
            str(tmp_path),
            "-fsyntax-only",
            str(tmp_path / "x.c"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_c_stub_without_tinyllm_header_skips_error_slot():
    src = "int add(int a, int b) {\n/* SOLUTION-BEGIN lang.03 */\n  return a + b;\n/* SOLUTION-END */\n}\n"
    out = markers.stub(src, "primers/lang.03/add.c", "lang.03")
    assert "tl_set_last_error" not in out and "return (int){0};" in out


def test_want_keeps_other_ids():
    src = "def a():\n    # SOLUTION-BEGIN M01.1\n    return 1\n    # SOLUTION-END\n\n\ndef b():\n    # SOLUTION-BEGIN M01.2\n    return 2\n    # SOLUTION-END\n"
    out = markers.stub(src, "python/x.py", "M01.2")
    assert (
        "return 1" in out
        and 'raise NotImplementedError("M01.2")' in out
        and "SOLUTION" not in out
    )


def test_python_and_rust_stubs():
    py = 'def f(x):\n    """doc"""\n    # SOLUTION-BEGIN M04.1\n    return x\n    # SOLUTION-END\n'
    assert (
        markers.stub(py, "python/f.py", "M04.1")
        == 'def f(x):\n    """doc"""\n    raise NotImplementedError("M04.1")\n'
    )
    rs = "pub fn f() -> u32 {\n    // SOLUTION-BEGIN ds.05\n    1\n    // SOLUTION-END\n}\n"
    assert '    todo!("ds.05")' in markers.stub(rs, "rust/f.rs", "ds.05")


GO_SRC = """\
package demo

import (
\t"errors"
\t"math"
\tstr "strings"
)

var ErrX = errors.New("x")

func F(x float64) float64 {
\t// SOLUTION-BEGIN dur.01
\treturn math.Sqrt(x) + float64(len(str.TrimSpace(" a ")))
\t// SOLUTION-END
}
"""


def test_go_stub_blanks_unused_imports(tmp_path):
    out = markers.stub(GO_SRC, "go/demo/f.go", "dur.01")
    assert '\t"errors"' in out and '\t_ "math"' in out and '\t_ "strings"' in out
    assert 'panic("todo: dur.01")' in out
    if shutil.which("go") is None:
        pytest.skip("go not installed")
    (tmp_path / "go.mod").write_text("module demo\n\ngo 1.22\n")
    (tmp_path / "f.go").write_text(out)
    r = subprocess.run(
        ["go", "build", "./..."],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "GOWORK": "off", "GOFLAGS": ""},
    )
    assert r.returncode == 0, r.stderr


def test_lint_whole_bodies_only():
    good = "def f():\n    # SOLUTION-BEGIN M01.1\n    return 1\n    # SOLUTION-END\n"
    assert markers.lint(good, "f.py") == []
    partial = "def f():\n    x = 1\n    # SOLUTION-BEGIN M01.1\n    return x\n    # SOLUTION-END\n"
    assert any("body of a def" in e for e in markers.lint(partial, "f.py"))
    c_partial = "int f(void) {\n  int x = 1;\n/* SOLUTION-BEGIN rt.01 */\n  return x;\n/* SOLUTION-END */\n}\n"
    assert any("opening brace" in e for e in markers.lint(c_partial, "f.c"))
    c_tail = "int f(void) {\n/* SOLUTION-BEGIN rt.01 */\n  int x = 1;\n/* SOLUTION-END */\n  return x;\n}\n"
    assert any("closing brace" in e for e in markers.lint(c_tail, "f.c"))
    assert markers.lint(C_SRC, "x.c") == []


def test_drop_markers():
    assert "SOLUTION" not in markers.drop_markers(C_SRC)
    assert "return n ? TL_OK : TL_EINVAL;" in markers.drop_markers(C_SRC)


def test_glue_units_need_no_markers():
    assert not markers.defines_functions("pub mod acc;\npub mod mean;\n", "lib.rs")
    assert markers.defines_functions("pub fn f() -> u8 { 1 }\n", "lib.rs")


def test_bash_strip_solution_honors_ids(tmp_path):
    ss = Path(__file__).resolve().parents[3] / "practice" / "bin" / "ol"
    text = ss.read_text()
    fn = text[text.index("strip_solution() {") : text.index("has_markers() {")]
    src = tmp_path / "x.py"
    src.write_text(
        "def a():\n    # SOLUTION-BEGIN M01.1\n    return 1\n    # SOLUTION-END\n"
        "def b():\n    # SOLUTION-BEGIN\n    return 2\n    # SOLUTION-END\n"
    )
    script = f"MARKER_BEGIN=SOLUTION-BEGIN; MARKER_END=SOLUTION-END\n{fn}\nstrip_solution {src} M01.1\nstrip_solution {src} other\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    want_first = "def a():\n    # TODO: your implementation goes here.\ndef b():\n    # TODO: your implementation goes here.\n"
    want_second = (
        "def a():\n    return 1\ndef b():\n    # TODO: your implementation goes here.\n"
    )
    assert r.stdout == want_first + want_second, r.stdout
