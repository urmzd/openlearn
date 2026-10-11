/* Course tests for rt.02: c/src/runtime/arena.c against tinyllm/arena.h,
 * built with ASan and UBSan and the counting allocator (-DOL_COUNTING_ALLOC=1:
 * every hook call is counted, a test that ends with a live block fails, and
 * ol_alloc_fail_after(n) makes the n-th call from now return NULL).
 *
 * Offsets in these tests are measured from the first allocation, which the
 * default alignment puts on a 64-byte boundary, so they do not depend on
 * where the hook placed a block.
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

static uintptr_t addr(const void *p) { return (uintptr_t)p; }

static tl_arena_stats stats(const tl_arena *a) {
    tl_arena_stats s;
    memset(&s, 0xAB, sizeof s);
    tl_arena_stats_get(a, &s);
    return s;
}

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example in a 256-byte block. a = 10 bytes at
     *      the default alignment 64; b = 4 bytes aligned to 64 lands 64 past
     *      a (54 bytes of padding); a mark; c = 100 bytes aligned to 8 lands
     *      at 72; rewinding to the mark and asking again returns c's address.
     *      bytes_used grows by padding + size: +58, then +104, and goes back
     *      to its value at the mark; high_water keeps the peak.
     * KIND: unit, smoke
     * CATCHES: s01, s11
     * CHAPTER: rt.02 section 3 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(256, &ar), TL_OK);
    char *a = tl_arena_alloc(ar, 10, 0);
    OL_TRUE(a != NULL && addr(a) % 64 == 0);
    size_t u0 = stats(ar).bytes_used;
    OL_TRUE(u0 >= 10 && u0 <= 10 + 63);
    char *b = tl_arena_alloc(ar, 4, 64);
    OL_EQ(b - a, 64);
    OL_EQ(stats(ar).bytes_used, u0 + 58);
    tl_arena_mark m = tl_arena_mark_get(ar);
    char *c = tl_arena_alloc(ar, 100, 8);
    OL_EQ(c - a, 72);
    OL_EQ(stats(ar).bytes_used, u0 + 58 + 104);
    tl_arena_reset_to(ar, m);
    OL_EQ(stats(ar).bytes_used, u0 + 58);
    OL_EQ(stats(ar).high_water, u0 + 58 + 104);
    char *c2 = tl_arena_alloc(ar, 100, 8);
    OL_TRUE(c2 == c);
    tl_arena_stats s = stats(ar);
    OL_EQ(s.n_blocks, 1);
    OL_EQ(s.bytes_reserved, 256);
    tl_arena_destroy(ar);
}

OL_TEST(every_power_of_two_alignment) {
    /* WHY: kernels ask for 16 (NEON), 32 (AVX), 64 (a cache line), and 4096
     *      (a page) after allocations of odd sizes; each pointer must be a
     *      multiple of its alignment, and 0 means 64.
     * KIND: unit
     * CATCHES: s02, s03
     * CHAPTER: rt.02 section 2.2 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(1 << 16, &ar), TL_OK);
    for (size_t al = 1; al <= 4096; al *= 2) {
        OL_TRUE(tl_arena_alloc(ar, 3, 1) != NULL); /* knock the offset off any boundary */
        void *p = tl_arena_alloc(ar, 7, al);
        OL_TRUE(p != NULL);
        OL_EQ(addr(p) % al, 0);
    }
    OL_TRUE(tl_arena_alloc(ar, 3, 1) != NULL);
    OL_EQ(addr(tl_arena_alloc(ar, 1, 0)) % 64, 0);
    tl_arena_destroy(ar);
}

OL_TEST(bad_requests_change_nothing) {
    /* WHY: n = 0, an alignment that is not a power of two, and one above
     *      4096 return NULL with a message, and the arena is exactly as
     *      before: the next good request gets the address it would have had.
     * KIND: boundary
     * CATCHES: s04, m002
     * CHAPTER: rt.02 section 4 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(1024, &ar), TL_OK);
    char *a = tl_arena_alloc(ar, 16, 16);
    tl_arena_stats before = stats(ar);
    tl_set_last_error("");
    OL_TRUE(tl_arena_alloc(ar, 0, 16) == NULL);
    OL_TRUE(strstr(tl_last_error(), "tl_arena_alloc") != NULL);
    tl_set_last_error("");
    OL_TRUE(tl_arena_alloc(ar, 8, 3) == NULL);
    OL_TRUE(strstr(tl_last_error(), "tl_arena_alloc") != NULL);
    OL_TRUE(tl_arena_alloc(ar, 8, 8192) == NULL);
    tl_arena_stats after = stats(ar);
    OL_EQ(after.bytes_used, before.bytes_used);
    OL_EQ(after.n_blocks, before.n_blocks);
    OL_TRUE(tl_arena_alloc(ar, 16, 16) == a + 16);
    OL_EQ(tl_arena_create(64, NULL), TL_EINVAL);
    tl_arena_destroy(ar);
    tl_arena_destroy(NULL);
}

OL_TEST(nested_marks) {
    /* WHY: an attention kernel takes a mark per call and its online-softmax
     *      helper takes another inside it. Rewinding the inner mark releases
     *      only the inner allocations; rewinding the outer one releases both,
     *      and allocation resumes at the same addresses, across three blocks.
     * KIND: unit
     * CATCHES: s05, s08, s09
     * CHAPTER: rt.02 section 2.3 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(256, &ar), TL_OK); /* each step below needs a new block */
    (void)tl_arena_alloc(ar, 100, 0);
    tl_arena_mark outer = tl_arena_mark_get(ar);
    char *x = tl_arena_alloc(ar, 200, 0);
    tl_arena_mark inner = tl_arena_mark_get(ar);
    char *y = tl_arena_alloc(ar, 300, 0);
    tl_arena_reset_to(ar, inner);
    OL_TRUE(tl_arena_alloc(ar, 300, 0) == y);
    tl_arena_reset_to(ar, inner);
    tl_arena_reset_to(ar, outer);
    OL_TRUE(tl_arena_alloc(ar, 200, 0) == x);
    tl_arena_destroy(ar);
}

OL_TEST(a_later_mark_is_ignored) {
    /* WHY: marks nest like a stack. Rewinding to a mark taken after the
     *      current position would "release" memory that was never handed
     *      out and later hand the same bytes out twice; the contract says
     *      ignore it and set the error slot.
     * KIND: boundary
     * CATCHES: s07
     * CHAPTER: rt.02 section 5, Pitfalls */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(128, &ar), TL_OK);
    tl_arena_mark start = tl_arena_mark_get(ar);
    (void)tl_arena_alloc(ar, 100, 0);
    (void)tl_arena_alloc(ar, 100, 0); /* a second block */
    tl_arena_mark late = tl_arena_mark_get(ar);
    tl_arena_reset_to(ar, start);
    char *p = tl_arena_alloc(ar, 50, 0);
    size_t used = stats(ar).bytes_used;
    tl_set_last_error("");
    tl_arena_reset_to(ar, late);
    OL_TRUE(strstr(tl_last_error(), "tl_arena_reset_to") != NULL);
    OL_EQ(stats(ar).bytes_used, used);
    char *q = tl_arena_alloc(ar, 10, 1);
    OL_TRUE(q == p + 50);
    tl_arena_destroy(ar);
}

OL_TEST(blocks_are_kept_and_reused) {
    /* WHY: the point of an arena: after a rewind, the blocks it already has
     *      serve the next round, so no block is freed and none is asked for
     *      again. Two 100-byte requests in 128-byte blocks need two blocks,
     *      the second time too, and get the same addresses.
     * KIND: unit
     * CATCHES: s05, s08, s09
     * CHAPTER: rt.02 section 2.4 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(128, &ar), TL_OK);
    tl_arena_mark start = tl_arena_mark_get(ar);
    char *a = tl_arena_alloc(ar, 100, 0), *b = tl_arena_alloc(ar, 100, 0);
    OL_EQ(stats(ar).n_blocks, 2);
    OL_EQ(stats(ar).bytes_reserved, 256);
    long live = ol_live_allocs();
    tl_arena_reset_to(ar, start);
    OL_EQ(ol_live_allocs(), live);
    OL_TRUE(tl_arena_alloc(ar, 100, 0) == a);
    OL_TRUE(tl_arena_alloc(ar, 100, 0) == b);
    OL_EQ(stats(ar).n_blocks, 2);
    OL_EQ(ol_live_allocs(), live);
    tl_arena_destroy(ar);
}

OL_TEST(exact_fit_uses_the_last_byte) {
    /* WHY: a request that exactly fills what is left of a block belongs in
     *      that block. An off-by-one in the fit check (< for <=) wastes the
     *      tail and asks the hook for a block the step does not need.
     * KIND: boundary
     * CATCHES: m001
     * CHAPTER: rt.02 section 3 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(128, &ar), TL_OK);
    char *a = tl_arena_alloc(ar, 64, 0);
    char *b = tl_arena_alloc(ar, 64, 0);
    OL_TRUE(a != NULL && b == a + 64);
    OL_EQ(stats(ar).n_blocks, 1);
    tl_arena_destroy(ar);
}

OL_TEST(oversized_request_gets_its_own_block) {
    /* WHY: a request larger than the block size (a long prompt's attention
     *      tile) must still succeed, in a block of its own, aligned, and
     *      fully writable (ASan reports a byte past the block).
     * KIND: boundary
     * CATCHES: s10
     * CHAPTER: rt.02 section 4 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(128, &ar), TL_OK);
    unsigned char *p = tl_arena_alloc(ar, 1000, 256);
    OL_TRUE(p != NULL && addr(p) % 256 == 0);
    memset(p, 0x5A, 1000);
    OL_TRUE(stats(ar).bytes_reserved >= 1000);
    unsigned char *q = tl_arena_alloc(ar, 20, 0);
    OL_TRUE(q != NULL);
    memset(q, 0x33, 20);
    for (int i = 0; i < 1000; i++) OL_EQ(p[i], 0x5A);
    tl_arena_destroy(ar);
}

/* Model-based random operations: a stack of marks, live allocations filled
 * with a byte pattern, and the stats the contract promises. */
enum { MAXLIVE = 256, MAXMARKS = 32 };
typedef struct {
    unsigned char *p;
    size_t n;
    unsigned char tag;
} live_t;

static int random_ops(ol_gen *g, int size) {
    tl_arena *ar = NULL;
    if (tl_arena_create((size_t)ol_gen_int(g, 64, 512), &ar) != TL_OK) return 0;
    live_t live[MAXLIVE];
    int nlive = 0;
    struct {
        tl_arena_mark m;
        int nlive;
        size_t used;
    } marks[MAXMARKS];
    int nmarks = 0, ok = 1;
    size_t peak = 0;
    for (int step = 0; step < 20 + size && ok; step++) {
        int op = (int)ol_gen_int(g, 0, 9);
        if (op < 6 && nlive < MAXLIVE) {
            size_t n = (size_t)ol_gen_int(g, 1, 700), al = (size_t)1 << ol_gen_int(g, 0, 8);
            unsigned char *p = tl_arena_alloc(ar, n, al);
            if (p == NULL || addr(p) % al != 0) ok = 0;
            else {
                for (int i = 0; i < nlive; i++) /* no overlap with a live allocation */
                    if (p < live[i].p + live[i].n && live[i].p < p + n) ok = 0;
                unsigned char tag = (unsigned char)ol_gen_int(g, 1, 255);
                memset(p, tag, n);
                live[nlive++] = (live_t){p, n, tag};
            }
        } else if (op < 8 && nmarks < MAXMARKS) {
            marks[nmarks].m = tl_arena_mark_get(ar);
            marks[nmarks].nlive = nlive;
            marks[nmarks].used = 0;
            tl_arena_stats s;
            tl_arena_stats_get(ar, &s);
            marks[nmarks++].used = s.bytes_used;
        } else if (nmarks > 0) {
            int k = (int)ol_gen_int(g, 0, nmarks - 1); /* any earlier mark */
            tl_arena_reset_to(ar, marks[k].m);
            nlive = marks[k].nlive;
            tl_arena_stats s;
            tl_arena_stats_get(ar, &s);
            if (s.bytes_used != marks[k].used) ok = 0;
            nmarks = k;
        }
        tl_arena_stats s;
        tl_arena_stats_get(ar, &s);
        if (s.bytes_used > peak) peak = s.bytes_used;
        size_t sum = 0;
        for (int i = 0; i < nlive; i++) sum += live[i].n;
        if (s.bytes_used < sum || s.high_water != peak || s.bytes_used > s.bytes_reserved) ok = 0;
    }
    for (int i = 0; i < nlive && ok; i++) /* nothing live was overwritten */
        for (size_t j = 0; j < live[i].n; j++)
            if (live[i].p[j] != live[i].tag) {
                ok = 0;
                break;
            }
    tl_arena_destroy(ar);
    return ok;
}

OL_TEST(random_operations_against_a_model) {
    /* WHY: 300 seeded runs of random allocations, marks, and rewinds to any
     *      earlier mark. Every live allocation stays aligned, disjoint from
     *      the others, and unchanged; bytes_used returns exactly to its value
     *      at the mark, never falls below the live bytes, and high_water is
     *      the running maximum.
     * KIND: property
     * CATCHES: s01, s02, s05, s06, s11, m003
     * CHAPTER: rt.02 section 2.3 */
    OL_CHECK_PROP(random_ops, 300, 200);
}

OL_TEST(zero_hook_calls_after_warmup) {
    /* WHY: the reason the engine uses an arena: once the blocks exist, a
     *      step (mark, allocate the scratch, rewind) never reaches the
     *      allocator hook, so it cannot fail and costs no malloc. After
     *      three warmup steps every hook call is made to fail; 1000 more
     *      steps must still succeed.
     * KIND: fault
     * CATCHES: s05, s06, s08, s09, s11
     * CHAPTER: rt.02 section 1 */
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(4096, &ar), TL_OK);
    static const size_t sizes[] = {1000, 3000, 64, 5000, 200, 4000};
    for (int step = 0; step < 1003; step++) {
        if (step == 3) ol_alloc_fail_after(0);
        tl_arena_mark m = tl_arena_mark_get(ar);
        for (int i = 0; i < 6; i++) {
            if (tl_arena_alloc(ar, sizes[i], 64) == NULL) {
                ol_alloc_fail_after(-1);
                tl_arena_destroy(ar);
                OL_FAIL("an allocation after warmup reached the hook");
            }
        }
        tl_arena_reset_to(ar, m);
    }
    ol_alloc_fail_after(-1);
    tl_arena_stats s = stats(ar);
    OL_EQ(s.bytes_used, 0);
    OL_TRUE(s.high_water >= 13264);
    tl_arena_destroy(ar);
}

OL_TEST(hook_failure_leaves_the_arena_usable) {
    /* WHY: every path that asks the hook for memory can fail. For each n,
     *      fail the n-th hook call in a create-and-grow sequence: create
     *      returns TL_ENOMEM with nothing allocated, a failed alloc returns
     *      NULL and leaves the arena as it was, and destroy then frees
     *      everything (the counting allocator reports any leak).
     * KIND: fault
     * CATCHES: s12, s13, m004
     * CHAPTER: rt.02 section 4 */
    for (long n = 0; n < 40; n++) {
        ol_alloc_fail_after(n);
        tl_arena *ar = NULL;
        tl_status st = tl_arena_create(64, &ar);
        if (st != TL_OK) {
            OL_EQ(st, TL_ENOMEM);
            OL_TRUE(ar == NULL);
            continue;
        }
        int failed = 0;
        for (int i = 0; i < 30; i++) {
            tl_arena_stats before = stats(ar);
            void *p = tl_arena_alloc(ar, 48, 0); /* one block each time */
            if (p == NULL) {
                failed = 1;
                tl_arena_stats after = stats(ar);
                OL_EQ(after.bytes_used, before.bytes_used);
                OL_EQ(after.n_blocks, before.n_blocks);
            } else {
                memset(p, 1, 48);
            }
        }
        ol_alloc_fail_after(-1);
        /* Half the time destroy at once, half after a successful request,
         * so both "never got a block" and "recovered" end states are freed. */
        if (n % 2 == 1) OL_TRUE(tl_arena_alloc(ar, 48, 0) != NULL || !failed);
        tl_arena_destroy(ar);
    }
    ol_alloc_fail_after(-1);
}

int main(void) { return OL_RUN_ALL(); }


/* ABI support tests, combined here so the C harness links one main. */
/* A hook that counts its own calls through the `user` pointer, so a test can
 * see which allocator served a request. */
typedef struct {
    long allocs, frees;
} hook_calls;

static void *counting_alloc(void *user, size_t n, size_t align) {
    hook_calls *h = user;
    h->allocs++;
    if (align < sizeof(void *)) align = sizeof(void *);
    size_t size = (n + align - 1) / align * align;
    return aligned_alloc(align, size);
}

static void counting_free(void *user, void *p) {
    hook_calls *h = user;
    h->frees++;
    free(p);
}

/* Put the harness's counting allocator back after a test replaced the hook,
 * so the leak check keeps working for the tests that follow. */
static void restore_harness_hook(void) {
#if OL__COUNTING
    tl_allocator a = {ol__alloc, ol__free, NULL};
    tl_set_allocator(&a);
#else
    tl_set_allocator(NULL);
#endif
}

OL_TEST(abi_version_matches_header) {
    /* WHY: C modules compare the implementation with TL_ABI_VERSION; the support
     *      routines must report the header version exactly.
     * KIND: unit, smoke
     * CHAPTER: rt.02 section 4 */
    OL_EQ(tl_abi_version(), TL_ABI_VERSION);
    OL_EQ(tl_abi_version(), 1);
}

OL_TEST(status_str_names_every_code) {
    /* WHY: every status has its constant's name, so a log line or a Python
     *      exception says TL_EINVAL instead of 1. The table is the one the
     *      chapter's worked example walks.
     * KIND: unit
     * CHAPTER: rt.02 section 3 */
    static const char *names[] = {"TL_OK", "TL_EINVAL", "TL_ENOMEM", "TL_ESHAPE", "TL_EDTYPE", "TL_EFULL",
                                  "TL_ENOTFOUND", "TL_EFORMAT", "TL_EBUSY", "TL_EUNSUPPORTED", "TL_EIO"};
    for (int32_t s = 0; s <= TL_EIO; s++) {
        const char *got = tl_status_str(s);
        OL_TRUE(got != NULL);
        if (strcmp(got, names[s]) != 0) OL_FAIL(names[s]);
    }
}

OL_TEST(status_str_unknown_codes) {
    /* WHY: tl_status is an integer enum, so a C caller can
     *      hand in any value: a negative one, or a code from a newer ABI. An
     *      array indexed by s would read out of bounds; the answer must be
     *      TL_UNKNOWN, never NULL and never a crash.
     * KIND: boundary
     * CHAPTER: rt.02 section 5, Pitfalls */
    OL_TRUE(strcmp(tl_status_str(11), "TL_UNKNOWN") == 0);
    OL_TRUE(strcmp(tl_status_str(-1), "TL_UNKNOWN") == 0);
    OL_TRUE(strcmp(tl_status_str(INT32_MAX), "TL_UNKNOWN") == 0);
    OL_TRUE(strcmp(tl_status_str(INT32_MIN), "TL_UNKNOWN") == 0);
}

OL_TEST(error_slot_copies_and_truncates) {
    /* WHY: the error slot is shared support code, and every C unit
     *      relies on it: it copies the message (the caller's buffer may be a
     *      stack array that dies on return), treats NULL as "", and truncates
     *      to TL_LAST_ERROR_CAP - 1 bytes instead of overflowing.
     * KIND: boundary, regression
     * CHAPTER: rt.02 section 2 */
    char buf[16] = "from a buffer";
    tl_set_last_error(buf);
    memset(buf, 'X', sizeof buf - 1);
    OL_TRUE(strcmp(tl_last_error(), "from a buffer") == 0);
    tl_set_last_error(NULL);
    OL_TRUE(tl_last_error() != NULL && tl_last_error()[0] == '\0');
    char big[TL_LAST_ERROR_CAP + 50];
    memset(big, 'a', sizeof big - 1);
    big[sizeof big - 1] = '\0';
    tl_set_last_error(big);
    OL_EQ((long long)strlen(tl_last_error()), TL_LAST_ERROR_CAP - 1);
}

OL_TEST(alloc_goes_through_the_hook) {
    /* WHY: the harness counts live blocks through the hook it installed with
     *      your tl_set_allocator. A tl_alloc or tl_free that bypasses the hook
     *      makes every leak check in the course blind, and freeing hook memory
     *      with free() corrupts any allocator that is not malloc.
     * KIND: fault
     * CHAPTER: rt.02 section 2 */
#if OL__COUNTING
    long before = ol_live_allocs();
    void *p = tl_alloc(10, 64);
    OL_TRUE(p != NULL);
    OL_EQ(ol_live_allocs(), before + 1);
    tl_free(p);
    OL_EQ(ol_live_allocs(), before);
#endif
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    OL_EQ(tl_set_allocator(&mine), TL_OK);
    void *q = tl_alloc(24, 16);
    tl_free(q);
    long allocs = calls.allocs, frees = calls.frees;
    restore_harness_hook();
    OL_TRUE(q != NULL);
    OL_EQ(allocs, 1);
    OL_EQ(frees, 1);
}

OL_TEST(alloc_alignment_by_hand) {
    /* WHY: the chapter's worked example: a request for 10 bytes at alignment
     *      64 returns an address divisible by 64. Kernels rely on it for
     *      vector loads; the default hook must round the size up to a
     *      multiple of the alignment, which aligned_alloc requires.
     * KIND: unit
     * CHAPTER: rt.02 section 3 */
    tl_set_allocator(NULL); /* the default hook, the one your code supplies */
    int ok = 1;
    for (size_t align = 1; align <= 4096; align *= 2) {
        void *p = tl_alloc(10, align);
        if (p == NULL || (uintptr_t)p % align != 0) ok = 0;
        if (p) memset(p, 0xAB, 10); /* ASan: the 10 bytes are really ours */
        tl_free(p);
    }
    restore_harness_hook();
    OL_TRUE(ok);
}

OL_TEST(set_allocator_null_restores_the_default) {
    /* WHY: tl_set_allocator(NULL) puts the default hook back. A library that
     *      keeps the last hook after NULL keeps calling a hook whose owner may
     *      be gone (a test's counters on its stack, a freed arena).
     * KIND: unit
     * CHAPTER: rt.02 section 2 */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    OL_EQ(tl_set_allocator(&mine), TL_OK);
    OL_EQ(tl_set_allocator(NULL), TL_OK);
    void *p = tl_alloc(32, 16);
    tl_free(p);
    long allocs = calls.allocs, frees = calls.frees;
    restore_harness_hook();
    OL_TRUE(p != NULL);
    OL_EQ(allocs, 0);
    OL_EQ(frees, 0);
}

OL_TEST(alloc_rejects_zero_and_bad_alignment) {
    /* WHY: tl_alloc(0, ...) and a non power of two alignment are caller bugs
     *      the contract defines: NULL with the error slot set, and the hook
     *      never called (a hook may not accept them either).
     * KIND: boundary
     * CHAPTER: rt.02 section 4 */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    OL_EQ(tl_set_allocator(&mine), TL_OK);
    tl_set_last_error(NULL);
    void *a = tl_alloc(0, 8);
    int err_a = tl_last_error()[0] != '\0';
    tl_set_last_error(NULL);
    void *b = tl_alloc(8, 3);
    int err_b = tl_last_error()[0] != '\0';
    void *c = tl_alloc(8, 0);
    long hook_used = calls.allocs;
    restore_harness_hook();
    OL_TRUE(a == NULL && b == NULL && c == NULL);
    OL_TRUE(err_a && err_b);
    OL_EQ(hook_used, 0);
}

OL_TEST(alloc_failure_sets_the_error) {
    /* WHY: when the hook returns NULL, tl_alloc returns NULL and says why;
     *      every constructor turns that into TL_ENOMEM. Later modules exercise this
     *      fault path with fail-after-n tests.
     * KIND: fault
     * CHAPTER: rt.02 section 4 */
#if OL__COUNTING
    tl_set_last_error(NULL);
    ol_alloc_fail_after(0);
    void *p = tl_alloc(16, 16);
    OL_TRUE(p == NULL);
    OL_TRUE(strstr(tl_last_error(), "tl_alloc") != NULL);
#endif
}

OL_TEST(set_allocator_rejects_half_a_hook) {
    /* WHY: a hook with alloc but no free would leak every block it hands
     *      out; the contract says TL_EINVAL, and the old hook stays installed.
     * KIND: boundary
     * CHAPTER: rt.02 section 4 */
    tl_allocator half = {counting_alloc, NULL, NULL};
    OL_EQ(tl_set_allocator(&half), TL_EINVAL);
    OL_TRUE(strstr(tl_last_error(), "tl_set_allocator") != NULL);
    tl_allocator other_half = {NULL, counting_free, NULL};
    OL_EQ(tl_set_allocator(&other_half), TL_EINVAL);
#if OL__COUNTING
    long before = ol_live_allocs();
    void *p = tl_alloc(8, 8); /* still the harness hook */
    OL_EQ(ol_live_allocs(), before + 1);
    tl_free(p);
#endif
}

OL_TEST(set_allocator_copies_the_struct) {
    /* WHY: callers build the tl_allocator on the stack and return. Keeping a
     *      pointer to it reads a dead stack frame on the next tl_alloc; the
     *      hook must be copied when it is installed.
     * KIND: fault
     * CHAPTER: rt.02 section 5, Pitfalls */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    OL_EQ(tl_set_allocator(&mine), TL_OK);
    mine.alloc = NULL; /* the caller reuses its struct */
    mine.free = NULL;
    mine.user = NULL;
    void *p = tl_alloc(8, 8);
    tl_free(p);
    long allocs = calls.allocs;
    restore_harness_hook();
    OL_TRUE(p != NULL);
    OL_EQ(allocs, 1);
}

OL_TEST(free_null_is_a_no_op) {
    /* WHY: cleanup paths call tl_free on pointers that may never have been
     *      allocated; like free(NULL), it must do nothing (and must not hand
     *      NULL to a hook that does not expect it).
     * KIND: boundary
     * CHAPTER: rt.02 section 4 */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    OL_EQ(tl_set_allocator(&mine), TL_OK);
    tl_free(NULL);
    long frees = calls.frees;
    restore_harness_hook();
    OL_EQ(frees, 0);
}
