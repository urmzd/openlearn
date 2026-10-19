/* lang.02 primer: given in the chapter; you write the Makefile.
   Exit codes: 0 ok, 1 runtime error, 2 usage error. */
#include <stdio.h>

#include "greet.h"

int main(int argc, char **argv) {
    char buf[64];
    if (argc != 2) {
        fprintf(stderr, "usage: %s NAME\n", argv[0]);
        return 2;
    }
    int n = greet(buf, sizeof buf, argv[1]);
    if (n < 0 || (size_t)n >= sizeof buf) {
        fprintf(stderr, "%s: name too long\n", argv[0]);
        return 1;
    }
    puts(buf);
    return 0;
}
