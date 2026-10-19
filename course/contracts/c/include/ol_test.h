/* ol_test.h: the single-header C test kit used by the solution's C tests.
 *
 * Frozen helper. Usage:
 *
 *   #include "tinyllm.h"      // optional; enables the counting allocator
 *   #include "ol_test.h"
 *
 *   OL_TEST(hand_example) {
 *       // WHY: ...   KIND: unit
 *       float y = 0;
 *       OL_EQ(tl_demo_sum_f32(x, 3, &y), TL_OK);
 *       OL_CLOSE(y, 6.0, 1e-6, 1e-7);
 *   }
 *
 *   int main(void) { return OL_RUN_ALL(); }
 *
 * Tests in several .c files link into one binary: the case table is a weak
 * global shared by every translation unit. OL_RUN_ALL() runs every case (or
 * the comma-separated names in $OL_ONLY), prints one line per case, and
 * returns the exit code: 0 when every case passed and at least one ran.
 *
 * Counting allocator: built with -DOL_COUNTING_ALLOC=1 (the C test build sets it)
 * and with tinyllm/abi.h included first, OL_RUN_ALL installs an allocator
 * through tl_set_allocator that counts live blocks, and fails any case that
 * returns with more blocks live than it started with. This is the leak check
 * on every platform (Apple clang has no LeakSanitizer). ol_alloc_fail_after(n)
 * makes the n-th allocation from now return NULL, for fault tests.
 */
#ifndef OL_TEST_H
#define OL_TEST_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef OL_MAX_CASES
#define OL_MAX_CASES 1024
#endif

typedef struct {
    const char *name;
    void (*fn)(void);
    const char *file;
} ol_case;

__attribute__((weak)) ol_case ol__cases[OL_MAX_CASES];
__attribute__((weak)) int ol__ncases;
__attribute__((weak)) int ol__failed;
__attribute__((weak)) long ol__live_allocs;
__attribute__((weak)) long ol__fail_after = -1;

static inline void ol__register(const char *name, void (*fn)(void), const char *file) {
    if (ol__ncases >= OL_MAX_CASES) {
        fprintf(stderr, "ol_test.h: more than %d cases; raise OL_MAX_CASES\n", OL_MAX_CASES);
        abort();
    }
    ol__cases[ol__ncases].name = name;
    ol__cases[ol__ncases].fn = fn;
    ol__cases[ol__ncases].file = file;
    ol__ncases++;
}

#define OL_TEST(name)                                                              \
    static void name(void);                                                        \
    __attribute__((constructor)) static void ol__reg_##name(void) {                \
        ol__register(#name, name, __FILE__);                                       \
    }                                                                              \
    static void name(void)

static inline int ol__fail(const char *file, int line, const char *what) {
    fprintf(stdout, "    %s:%d: %s\n", file, line, what);
    ol__failed = 1;
    return 0;
}

static inline int ol__eq_i(long long a, long long b, const char *ea, const char *eb, const char *file, int line) {
    if (a == b) return 1;
    char buf[512];
    snprintf(buf, sizeof buf, "OL_EQ(%s, %s): %lld != %lld", ea, eb, a, b);
    return ol__fail(file, line, buf);
}

static inline int ol__eq_d(double a, double b, const char *ea, const char *eb, const char *file, int line) {
    if (a == b) return 1;
    char buf[512];
    snprintf(buf, sizeof buf, "OL_EQ(%s, %s): %.17g != %.17g", ea, eb, a, b);
    return ol__fail(file, line, buf);
}

static inline double ol__abs(double x) { return x < 0 ? -x : x; }

static inline int ol__close(double a, double b, double rtol, double atol, const char *ea, const char *eb,
                     const char *file, int line) {
    double diff = ol__abs(a - b), allowed = atol + rtol * ol__abs(b);
    if (diff <= allowed) return 1;
    char buf[512];
    snprintf(buf, sizeof buf, "OL_CLOSE(%s, %s): %.17g vs %.17g, |diff| %.3e > allowed %.3e", ea, eb, a, b,
             diff, allowed);
    return ol__fail(file, line, buf);
}

/* Each assertion returns from the test on failure. */
#define OL_EQ(a, b)                                                                                 \
    do {                                                                                            \
        if (!_Generic((a) + (b), float: ol__eq_d, double: ol__eq_d, long double: ol__eq_d,       \
                      default: ol__eq_i)((a), (b), #a, #b, __FILE__, __LINE__))                     \
            return;                                                                                 \
    } while (0)
#define OL_TRUE(c)                                                                                  \
    do {                                                                                            \
        if (!(c)) {                                                                                 \
            ol__fail(__FILE__, __LINE__, "OL_TRUE(" #c ")");                                        \
            return;                                                                                 \
        }                                                                                           \
    } while (0)
#define OL_CLOSE(a, b, rtol, atol)                                                                  \
    do {                                                                                            \
        if (!ol__close((double)(a), (double)(b), (rtol), (atol), #a, #b, __FILE__, __LINE__))     \
            return;                                                                                 \
    } while (0)
#define OL_FAIL(msg)                                                                                \
    do {                                                                                            \
        ol__fail(__FILE__, __LINE__, (msg));                                                        \
        return;                                                                                     \
    } while (0)

static inline void ol_alloc_fail_after(long n) { ol__fail_after = n; }
static inline long ol_live_allocs(void) { return ol__live_allocs; }

#if defined(OL_COUNTING_ALLOC) && defined(TL_ABI_VERSION)
#define OL__COUNTING 1
static inline void *ol__alloc(void *user, size_t n, size_t align) {
    (void)user;
    if (ol__fail_after == 0) return NULL;
    if (ol__fail_after > 0) ol__fail_after--;
    if (align < sizeof(void *)) align = sizeof(void *);
    size_t size = n ? n : 1;
    size = (size + align - 1) / align * align;
    void *p = aligned_alloc(align, size);
    if (p) ol__live_allocs++;
    return p;
}
static inline void ol__free(void *user, void *p) {
    (void)user;
    if (p) {
        ol__live_allocs--;
        free(p);
    }
}
#else
#define OL__COUNTING 0
#endif

static inline int ol__selected(const char *name) {
    const char *only = getenv("OL_ONLY");
    if (!only || !*only) return 1;
    size_t n = strlen(name);
    for (const char *p = only; *p;) {
        const char *q = strchr(p, ',');
        size_t len = q ? (size_t)(q - p) : strlen(p);
        if (len == n && strncmp(p, name, n) == 0) return 1;
        if (!q) break;
        p = q + 1;
    }
    return 0;
}

static inline int ol_run_all(void) {
    int counting = 0;
#if OL__COUNTING
    tl_allocator a = {ol__alloc, ol__free, NULL};
    tl_status st = tl_set_allocator(&a);
    counting = st == TL_OK;
    if (!counting) fprintf(stdout, "note: counting allocator unavailable (tl_set_allocator returned %d)\n", (int)st);
#endif
    int ran = 0, failed = 0;
    for (int i = 0; i < ol__ncases; i++) {
        if (!ol__selected(ol__cases[i].name)) continue;
        long before = ol__live_allocs;
        ol__failed = 0;
        ol__fail_after = -1;
        ol__cases[i].fn();
        ran++;
        if (counting && !ol__failed && ol__live_allocs != before) {
            fprintf(stdout, "    leak: %ld allocation(s) still live after the test\n", ol__live_allocs - before);
            ol__failed = 1;
        }
        fprintf(stdout, "%s %s\n", ol__failed ? "FAIL" : "ok  ", ol__cases[i].name);
        failed += ol__failed;
    }
#if OL__COUNTING
    if (counting) tl_set_allocator(NULL);
#endif
    if (ran == 0) {
        fprintf(stdout, "no test ran%s\n", getenv("OL_ONLY") ? " (OL_ONLY matched nothing)" : "");
        return 1;
    }
    fprintf(stdout, "%d passed, %d failed\n", ran - failed, failed);
    fflush(stdout);
    return failed ? 1 : 0;
}

#define OL_RUN_ALL() ol_run_all()

#endif /* OL_TEST_H */
