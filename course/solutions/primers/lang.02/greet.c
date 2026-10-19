/* lang.02 primer: given in the chapter; you write the Makefile. */
#include <stdio.h>

#include "greet.h"

int greet(char *buf, size_t cap, const char *name) {
    return snprintf(buf, cap, "hello, %s", name);
}
