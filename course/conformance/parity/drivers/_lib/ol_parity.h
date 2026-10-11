/* ol_parity.h: the parity driver kit for C (course/DESIGN.md 5.8).
 *
 * `ol parity` feeds a driver one JSON object per stdin line (one case) and
 * reads one JSON value per stdout line back. A C driver reads the fields it
 * needs with these scanners, calls the unit under test, and prints its
 * answer with printf. The scanners understand exactly what the suites send:
 * numbers, arrays of numbers, and strings without escapes.
 *
 *   char line[1 << 20];
 *   while (ssp_line(line, sizeof line)) {
 *       int64_t n = ssp_int(line, "n", 0);
 *       double xs[64]; int64_t k = ssp_doubles(line, "xs", xs, 64);
 *       printf("%.9g\n", ...);
 *       fflush(stdout);
 *   }
 */
#ifndef OL_PARITY_H
#define OL_PARITY_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int ssp_line(char *buf, size_t cap) {
    if (!fgets(buf, (int)cap, stdin)) return 0;
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
    return 1;
}

/* Pointer just after `"key":` (and any spaces), or NULL. */
static inline const char *ssp_find(const char *line, const char *key) {
    size_t k = strlen(key);
    for (const char *p = line; (p = strchr(p, '"')) != NULL; p++) {
        if (strncmp(p + 1, key, k) == 0 && p[k + 1] == '"') {
            const char *q = p + k + 2;
            while (*q == ' ') q++;
            if (*q == ':') {
                q++;
                while (*q == ' ') q++;
                return q;
            }
        }
    }
    return NULL;
}

static inline double ssp_double(const char *line, const char *key, double dflt) {
    const char *p = ssp_find(line, key);
    return p ? strtod(p, NULL) : dflt;
}

static inline int64_t ssp_int(const char *line, const char *key, int64_t dflt) {
    const char *p = ssp_find(line, key);
    return p ? (int64_t)strtoll(p, NULL, 10) : dflt;
}

static inline uint64_t ssp_u64(const char *line, const char *key, uint64_t dflt) {
    const char *p = ssp_find(line, key);
    return p ? (uint64_t)strtoull(p, NULL, 10) : dflt;
}

/* Fills out[0..cap) from a JSON array of numbers; returns the count (-1 if absent). */
static inline int64_t ssp_doubles(const char *line, const char *key, double *out, int64_t cap) {
    const char *p = ssp_find(line, key);
    if (!p || *p != '[') return -1;
    p++;
    int64_t n = 0;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') break;
        char *end;
        double v = strtod(p, &end);
        if (end == p) return -1;
        if (n < cap) out[n] = v;
        n++;
        p = end;
    }
    return n;
}

static inline int64_t ssp_floats(const char *line, const char *key, float *out, int64_t cap) {
    double tmp[4096];
    int64_t n = ssp_doubles(line, key, tmp, cap < 4096 ? cap : 4096);
    for (int64_t i = 0; i < n && i < cap; i++) out[i] = (float)tmp[i];
    return n;
}

static inline void ssp_print_floats(const float *x, int64_t n) {
    putchar('[');
    for (int64_t i = 0; i < n; i++) printf(i ? ",%.9g" : "%.9g", (double)x[i]);
    puts("]");
    fflush(stdout);
}

#endif
