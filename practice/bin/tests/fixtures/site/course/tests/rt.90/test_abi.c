#include <string.h>

#include "tinyllm.h"
#include "ol_test.h"

OL_TEST(abi_version_matches_header) {
    /* WHY: a C consumer can compare the library version with the header's
     *      version before using the interface.
     * KIND: unit */
    OL_EQ(tl_abi_version(), TL_ABI_VERSION);
}

OL_TEST(status_strings) {
    /* WHY: every status has a readable name and an unknown code still gets one.
     * KIND: unit */
    OL_TRUE(tl_status_str(TL_OK) != NULL);
    OL_TRUE(strcmp(tl_status_str(TL_OK), "ok") == 0);
    OL_TRUE(strcmp(tl_status_str(12345), "unknown status") == 0);
}

OL_TEST(alloc_goes_through_the_hook) {
    /* WHY: the counting allocator only sees allocations that use tl_alloc, and
     *      64-byte alignment is what the kernels rely on.
     * KIND: fault */
    long before = ol_live_allocs();
    void *p = tl_alloc(10, 64);
    OL_TRUE(p != NULL);
    OL_EQ((long long)((uintptr_t)p % 64), 0);
    OL_EQ(ol_live_allocs(), before + 1);
    tl_free(p);
    OL_EQ(ol_live_allocs(), before);
}

OL_TEST(set_allocator_rejects_half_a_hook) {
    /* WHY: a hook with alloc but no free would leak every block it hands out.
     * KIND: boundary */
    tl_allocator half = {0};
    OL_EQ(tl_set_allocator(&half), TL_EINVAL);
    OL_TRUE(tl_last_error() != NULL);
}

int main(void) { return OL_RUN_ALL(); }
