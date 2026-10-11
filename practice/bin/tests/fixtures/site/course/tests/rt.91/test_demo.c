#include "tinyllm.h"
#include "ol_test.h"

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example: 1 + 2 + 3 = 6.
     * KIND: unit */
    float x[3] = {1.0f, 2.0f, 3.0f}, y = 0.0f;
    OL_EQ(tl_demo_sum_f32(x, 3, &y), TL_OK);
    OL_CLOSE(y, 6.0, 1e-6, 1e-7);
}

OL_TEST(null_out_is_einval) {
    /* WHY: a NULL out pointer is a caller bug that must be reported, not a crash.
     * KIND: boundary */
    float x[1] = {1.0f};
    OL_EQ(tl_demo_sum_f32(x, 1, NULL), TL_EINVAL);
}

OL_TEST(create_destroy_no_leak) {
    /* WHY: every block create takes, destroy gives back (the counting allocator checks).
     * KIND: fault */
    tl_demo_buf *b = NULL;
    OL_EQ(tl_demo_buf_create(8, &b), TL_OK);
    OL_TRUE(tl_demo_buf_data(b) != NULL);
    OL_CLOSE(tl_demo_buf_data(b)[7], 0.0, 0, 0);
    tl_demo_buf_destroy(b);
}

OL_TEST(alloc_failure_is_clean) {
    /* WHY: when the second allocation fails, the first must be freed: a
     *      half-built object is the classic constructor leak.
     * KIND: fault */
    tl_demo_buf *b = NULL;
    ol_alloc_fail_after(1);
    OL_EQ(tl_demo_buf_create(8, &b), TL_ENOMEM);
}

int main(void) { return OL_RUN_ALL(); }
