"""The C overlay's sanitizer builds (DESIGN 5.4, 4.4): ASan and UBSan in the
test build, ThreadSanitizer as a third build for modules that declare
`sanitize = ["thread"]`, and the counting allocator in all of them."""

# The planted race keeps the total right (the atomic add stays), so the ASan
# build's hand_example still passes and only ThreadSanitizer can fail the
# check: every worker also writes one plain shared variable.
RACY = """
int64_t race_last; /* external linkage: the stores cannot be dropped */
static void *race_work(void *p) {
    race_arg *a = p;
    for (int64_t i = 0; i < a->per; i++) {
        atomic_fetch_add_explicit(a->total, 1, memory_order_relaxed);
        race_last = i;
    }
    return NULL;
}
"""


def _start_abi(ss):
    ss("start", "rt.90", rc=0)
    ss.implement("c/src/runtime/abi.c")
    ss("check", "rt.90", rc=0)


def test_asan_and_ubsan_fail_the_check(ss):
    ss.init()
    _start_abi(ss)
    ss("start", "rt.91", rc=0)
    ss.implement("c/src/runtime/demo.c")
    demo = ss.learner / "c/src/runtime/demo.c"
    good = demo.read_text()
    demo.write_text(
        good.replace(
            "for (int64_t i = 0; i < n; i++) acc += x[i];",
            "for (int64_t i = 0; i <= n; i++) acc += x[i];",
        )
    )
    out = ss("check", "rt.91", rc=1).out
    assert "AddressSanitizer" in out and "buffer-overflow" in out
    demo.write_text(
        good.replace(
            "    *out = (float)acc;",
            # volatile: gcc folds `big - big` to 0 and drops a dead overflow
            # before UBSan sees it, so the overflowing sum must be used.
            "    volatile int big = 2147483647 - (int)n;\n"
            "    int over = big + (int)(acc > 0) * 8;\n"
            "    *out = (float)acc + 0.0f * (float)over;",
        )
    )
    out = ss("check", "rt.91", rc=1).out
    assert "runtime error" in out and "overflow" in out


def test_thread_sanitizer_build_catches_a_race(ss):
    ss.add_extras("sanitizers")
    ss.init()
    _start_abi(ss)
    ss("start", "rt.92", rc=0)
    ss.implement("c/src/runtime/race.c")
    out = ss("check", "rt.92", rc=0).out
    assert "PASS rt.92" in out
    race = ss.learner / "c/src/runtime/race.c"
    text = race.read_text()
    start = text.index("static void *race_work")
    end = text.index("tl_status tl_demo_count_parallel")
    race.write_text(text[:start] + RACY.lstrip() + "\n" + text[end:])
    out = ss("check", "rt.92", rc=1).out
    assert "ThreadSanitizer: data race" in out
    out = ss("check", "rt.92", rc=None, env={"OL_TSAN": "0"}).out
    assert "ThreadSanitizer" not in out  # the TSan build is the one that sees it
