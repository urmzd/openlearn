/* http.c - the part you write. */
#include "http.h"

#include <stdlib.h>
#include <string.h>

enum { ST_REQUEST_LINE = 0, ST_HEADERS, ST_DONE, ST_ERROR };

static int ci_equal(const char *a, const char *b) {
  while (*a && *b) {
    char ca = *a, cb = *b;
    if (ca >= 'A' && ca <= 'Z') ca += 32;
    if (cb >= 'A' && cb <= 'Z') cb += 32;
    if (ca != cb) return 0;
    a++;
    b++;
  }
  return *a == '\0' && *b == '\0';
}

/* Copy at most cap-1 bytes and terminate. Returns 0 if the source did not
 * fit, so callers can reject rather than silently truncate. */
static int copy_bounded(char *dst, size_t cap, const char *src, size_t len) {
  if (len >= cap) return 0;
  memcpy(dst, src, len);
  dst[len] = '\0';
  return 1;
}

void http_init(HttpParser *p) {
  memset(p, 0, sizeof *p);
  p->state = ST_REQUEST_LINE;
  p->content_length = -1;
}

/* Parse one complete request line, without its terminator. */
static int parse_request_line(HttpParser *p, const char *s, size_t len) {
  /* METHOD SP PATH SP HTTP/1.N */
  const char *sp1 = memchr(s, ' ', len);
  if (!sp1) return 0;
  size_t method_len = (size_t)(sp1 - s);
  if (method_len == 0) return 0;

  const char *rest = sp1 + 1;
  size_t rest_len = len - method_len - 1;
  const char *sp2 = memchr(rest, ' ', rest_len);
  if (!sp2) return 0;
  size_t path_len = (size_t)(sp2 - rest);
  if (path_len == 0) return 0;

  const char *ver = sp2 + 1;
  size_t ver_len = rest_len - path_len - 1;

  /* Exactly "HTTP/1.N". Anything else is rejected rather than guessed at. */
  if (ver_len != 8 || memcmp(ver, "HTTP/1.", 7) != 0) return 0;
  if (ver[7] < '0' || ver[7] > '9') return 0;

  if (!copy_bounded(p->method, sizeof p->method, s, method_len)) return 0;
  if (!copy_bounded(p->path, sizeof p->path, rest, path_len)) return 0;
  p->version_minor = ver[7] - '0';
  return 1;
}

/* Parse one header line, without its terminator. */
static int parse_header_line(HttpParser *p, const char *s, size_t len) {
  const char *colon = memchr(s, ':', len);
  if (!colon) return 0;

  size_t name_len = (size_t)(colon - s);
  if (name_len == 0) return 0;

  /* No space is permitted between the name and the colon. This is not
   * pedantry: "Foo : bar" being accepted by one parser and rejected by
   * another is exactly how request smuggling gets through a proxy. */
  if (s[name_len - 1] == ' ' || s[name_len - 1] == '\t') return 0;

  const char *val = colon + 1;
  size_t val_len = len - name_len - 1;

  /* Optional whitespace either side of the value is stripped. */
  while (val_len > 0 && (*val == ' ' || *val == '\t')) { val++; val_len--; }
  while (val_len > 0 && (val[val_len - 1] == ' ' || val[val_len - 1] == '\t')) val_len--;

  if (p->header_count >= HTTP_MAX_HEADERS) return 0;
  HttpHeader *h = &p->headers[p->header_count];
  if (!copy_bounded(h->name, sizeof h->name, s, name_len)) return 0;
  if (!copy_bounded(h->value, sizeof h->value, val, val_len)) return 0;
  p->header_count++;

  if (ci_equal(h->name, "Content-Length")) {
    /* Reject anything that is not purely digits. strtol would happily accept
     * "12abc", a leading "+", or a negative, and a body length the parser and
     * the server disagree about is a smuggling primitive. */
    if (h->value[0] == '\0') return 0;
    long n = 0;
    for (const char *c = h->value; *c; c++) {
      if (*c < '0' || *c > '9') return 0;
      if (n > (long)((1L << 40) / 10)) return 0; /* absurdly large: refuse */
      n = n * 10 + (*c - '0');
    }
    /* A second, different Content-Length is a conflict, not an update. */
    if (p->content_length >= 0 && p->content_length != n) return 0;
    p->content_length = n;
  }
  return 1;
}

/* Hand one complete line (terminator already stripped) to the right parser. */
static int handle_line(HttpParser *p, const char *s, size_t len) {
  if (p->state == ST_REQUEST_LINE) {
    /* Leading blank lines before the request line are tolerated, as RFC 9112
     * recommends for robustness against stray CRLFs from old clients. */
    if (len == 0) return 1;
    if (!parse_request_line(p, s, len)) return 0;
    p->state = ST_HEADERS;
    return 1;
  }

  if (len == 0) { /* the blank line that ends the headers */
    p->state = ST_DONE;
    return 1;
  }
  return parse_header_line(p, s, len);
}

HttpStatus http_feed(HttpParser *p, const char *data, size_t len, size_t *consumed) {
  size_t i = 0;
  *consumed = 0;

  if (p->state == ST_ERROR) return HTTP_ERROR;
  if (p->state == ST_DONE) return HTTP_DONE;

  for (; i < len; i++) {
    char c = data[i];

    if (c == '\n') {
      /* A line is complete. Strip one trailing CR, so that both CRLF and a
       * bare LF work: the spec says CRLF, and real clients send both. */
      size_t n = p->line_len;
      if (n > 0 && p->line[n - 1] == '\r') n--;

      if (!handle_line(p, p->line, n)) {
        p->state = ST_ERROR;
        *consumed = i + 1;
        return HTTP_ERROR;
      }
      p->line_len = 0;

      if (p->state == ST_DONE) {
        /* Consume the terminator, and no more: whatever follows is body and
         * belongs to the caller. */
        *consumed = i + 1;
        return HTTP_DONE;
      }
      continue;
    }

    /* An over-long line is refused rather than truncated. Truncating would
     * turn a header the client sent into a different, shorter header that the
     * application then trusts. */
    if (p->line_len >= sizeof p->line) {
      p->state = ST_ERROR;
      *consumed = i;
      return HTTP_ERROR;
    }
    p->line[p->line_len++] = c;
  }

  /* Ran out of input mid-line. The partial line stays in p->line, which is
   * exactly what makes the next call able to resume. */
  *consumed = len;
  return HTTP_INCOMPLETE;
}

const char *http_header(const HttpParser *p, const char *name) {
  for (size_t i = 0; i < p->header_count; i++) {
    if (ci_equal(p->headers[i].name, name)) return p->headers[i].value;
  }
  return NULL;
}
