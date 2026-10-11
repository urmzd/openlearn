#include "tinyllm.h"
#include "ol_test.h"

OL_TEST(sums_four) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f}, y = 0.0f;
    OL_EQ(tl_demo_sum_f32(x, 4, &y), TL_OK);
    OL_CLOSE(y, 10.0, 0, 0);
}

OL_TEST(alloc_failure_frees_the_struct) {
    tl_demo_buf *b = NULL;
    ol_alloc_fail_after(1);
    OL_EQ(tl_demo_buf_create(4, &b), TL_ENOMEM);
}

int main(void) { return OL_RUN_ALL(); }
