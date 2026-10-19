/* tinyllm/failpoint.h: named failpoints for C units.
 *
 * Header only, so a unit includes it like any contract header and no unit
 * owns it. The spec is the one the Go, Rust, and Python kits read:
 *
 *   TL_FAILPOINTS="rt/kv/after-export=crash;rt/arena/grow=error;x=3*sleep(20ms)"
 *
 *   if (tl_failpoint("rt/kv/after-export")) return TL_EIO;   // error action fired
 *
 * Actions: crash (_exit(137), what a SIGKILL leaves), error (the call returns
 * 1; the unit decides which status that is), sleep(Nms), off. `N*` fires only
 * on the Nth evaluation of that name in this process. Without TL_FAILPOINTS
 * the call is one getenv and returns 0. Counts are per translation unit. */
#ifndef TINYLLM_FAILPOINT_H
#define TINYLLM_FAILPOINT_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TL_FAILPOINT_MAX 32

static inline int tl_failpoint(const char *name) {
    static char names[TL_FAILPOINT_MAX][64];
    static long counts[TL_FAILPOINT_MAX];
    const char *spec = getenv("TL_FAILPOINTS");
    if (spec == NULL || name == NULL) return 0;
    size_t nl = strlen(name);
    const char *p = spec;
    while (*p) {
        const char *end = strchr(p, ';');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        while (len && *p == ' ') p++, len--;
        if (len > nl && strncmp(p, name, nl) == 0 && p[nl] == '=') {
            const char *a = p + nl + 1;
            size_t alen = len - nl - 1;
            long nth = 0;
            while (alen && *a >= '0' && *a <= '9') nth = nth * 10 + (*a - '0'), a++, alen--;
            if (nth) {
                if (!alen || *a != '*') return 0; /* malformed: never fire */
                a++, alen--;
            }
            int slot = -1;
            for (int i = 0; i < TL_FAILPOINT_MAX; i++) {
                if (names[i][0] == '\0' || strcmp(names[i], name) == 0) {
                    if (names[i][0] == '\0' && nl < sizeof names[i]) memcpy(names[i], name, nl + 1);
                    slot = i;
                    break;
                }
            }
            long c = slot >= 0 ? ++counts[slot] : 1;
            if (nth && c != nth) return 0;
            if (alen >= 5 && strncmp(a, "crash", 5) == 0) _exit(137);
            if (alen >= 5 && strncmp(a, "error", 5) == 0) return 1;
            if (alen >= 6 && strncmp(a, "sleep(", 6) == 0) {
                long ms = strtol(a + 6, NULL, 10);
                struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
                nanosleep(&ts, NULL);
                return 0;
            }
            return 0; /* off, or an action C does not have */
        }
        if (!end) break;
        p = end + 1;
    }
    return 0;
}

#endif /* TINYLLM_FAILPOINT_H */
