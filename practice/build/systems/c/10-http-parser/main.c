/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0;
#define CHECK(cond, what)                                                      \
  do {                                                                         \
    checks++;                                                                  \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/* Feed a whole buffer in one call. */
static HttpStatus feed_all(HttpParser *p, const char *s, size_t *consumed) {
  return http_feed(p, s, strlen(s), consumed);
}

static const char *SIMPLE =
    "GET /index.html HTTP/1.1\r\n"
    "Host: example.com\r\n"
    "User-Agent: test/1.0\r\n"
    "Content-Length: 42\r\n"
    "\r\n";

static void test_parses_a_whole_request(void) {
  HttpParser p;
  size_t consumed = 0;
  http_init(&p);

  CHECK(feed_all(&p, SIMPLE, &consumed) == HTTP_DONE, "a complete request parses");
  CHECK(consumed == strlen(SIMPLE), "the whole header block was consumed");
  CHECK(strcmp(p.method, "GET") == 0, "method");
  CHECK(strcmp(p.path, "/index.html") == 0, "path");
  CHECK(p.version_minor == 1, "minor version");
  CHECK(p.header_count == 3, "three headers");
  CHECK(p.content_length == 42, "content length parsed");

  CHECK(strcmp(http_header(&p, "Host"), "example.com") == 0, "header lookup");
  CHECK(strcmp(http_header(&p, "host"), "example.com") == 0, "lookup is case-insensitive");
  CHECK(strcmp(http_header(&p, "HOST"), "example.com") == 0, "in both directions");
  CHECK(http_header(&p, "Missing") == NULL, "absent header returns NULL");
}

/* The point of the exercise: the same input split every possible way must give
 * the same answer. One byte at a time is the cruellest split and the one a
 * real network will eventually produce. */
static void test_every_possible_split(void) {
  size_t total = strlen(SIMPLE);

  for (size_t split = 0; split <= total; split++) {
    HttpParser p;
    size_t consumed = 0;
    http_init(&p);

    HttpStatus st = http_feed(&p, SIMPLE, split, &consumed);
    if (st != HTTP_DONE) {
      CHECK(st == HTTP_INCOMPLETE, "a partial request is incomplete, not an error");
      st = http_feed(&p, SIMPLE + consumed, total - consumed, &consumed);
    }
    CHECK(st == HTTP_DONE, "the request parses however it is split");
    CHECK(strcmp(p.method, "GET") == 0, "method survives the split");
    CHECK(strcmp(p.path, "/index.html") == 0, "path survives the split");
    CHECK(p.header_count == 3, "headers survive the split");
    CHECK(p.content_length == 42, "content length survives the split");
  }

  /* And now one byte at a time, all the way through. */
  HttpParser p;
  http_init(&p);
  HttpStatus st = HTTP_INCOMPLETE;
  for (size_t i = 0; i < total; i++) {
    size_t consumed = 0;
    st = http_feed(&p, SIMPLE + i, 1, &consumed);
    if (st == HTTP_DONE) {
      CHECK(i == total - 1, "it completes exactly at the final byte");
      break;
    }
    CHECK(st == HTTP_INCOMPLETE, "every intermediate byte leaves it incomplete");
  }
  CHECK(st == HTTP_DONE, "byte-at-a-time feeding completes");
  CHECK(strcmp(http_header(&p, "User-Agent"), "test/1.0") == 0, "and the values are intact");
}

/* consumed must stop at the end of the headers so the caller can find the
 * body. Over-consuming eats the first bytes of the payload. */
static void test_body_is_left_for_the_caller(void) {
  const char *req =
      "POST /submit HTTP/1.0\r\n"
      "Content-Length: 5\r\n"
      "\r\n"
      "HELLO";
  HttpParser p;
  size_t consumed = 0;
  http_init(&p);

  CHECK(feed_all(&p, req, &consumed) == HTTP_DONE, "request parses");
  CHECK(p.version_minor == 0, "HTTP/1.0 is recognised");
  CHECK(p.content_length == 5, "content length");
  CHECK(strcmp(req + consumed, "HELLO") == 0, "consumed stops exactly at the body");
}

static void test_whitespace_and_line_endings(void) {
  HttpParser p;
  size_t consumed = 0;

  /* Bare LF, which plenty of real clients send. */
  http_init(&p);
  CHECK(feed_all(&p, "GET / HTTP/1.1\nHost: a.com\n\n", &consumed) == HTTP_DONE,
        "bare LF line endings are accepted");
  CHECK(strcmp(http_header(&p, "Host"), "a.com") == 0, "and parse correctly");

  /* Value whitespace is stripped on both sides. */
  http_init(&p);
  CHECK(feed_all(&p, "GET / HTTP/1.1\r\nX-Pad:   spaced   \r\n\r\n", &consumed) == HTTP_DONE,
        "padded values parse");
  CHECK(strcmp(http_header(&p, "X-Pad"), "spaced") == 0, "surrounding whitespace is stripped");

  /* An empty value is legal. */
  http_init(&p);
  CHECK(feed_all(&p, "GET / HTTP/1.1\r\nX-Empty:\r\n\r\n", &consumed) == HTTP_DONE,
        "an empty header value is legal");
  CHECK(strcmp(http_header(&p, "X-Empty"), "") == 0, "and reads back as empty");

  /* No headers at all. */
  http_init(&p);
  CHECK(feed_all(&p, "GET / HTTP/1.1\r\n\r\n", &consumed) == HTTP_DONE,
        "a request with no headers parses");
  CHECK(p.header_count == 0, "and has no headers");
  CHECK(p.content_length == -1, "with content_length reported absent");
}

/* Malformed input must be rejected, not guessed at. Each of these is a real
 * parser-differential bug class: if two parsers in a chain disagree about
 * whether a request is valid, an attacker can smuggle a second request past
 * the first one. */
static void test_malformed_requests_are_rejected(void) {
  const char *bad[] = {
      "GET\r\n\r\n",                              /* no path or version */
      "GET /\r\n\r\n",                            /* no version */
      "GET / HTTP/9.9x\r\n\r\n",                  /* malformed version */
      "GET / FTP/1.1\r\n\r\n",                    /* wrong protocol */
      " / HTTP/1.1\r\n\r\n",                      /* empty method */
      "GET  HTTP/1.1\r\n\r\n",                    /* empty path */
      "GET / HTTP/1.1\r\nNoColonHere\r\n\r\n",    /* header without a colon */
      "GET / HTTP/1.1\r\n: novalue\r\n\r\n",      /* header with an empty name */
      "GET / HTTP/1.1\r\nFoo : bar\r\n\r\n",      /* space before the colon */
      "GET / HTTP/1.1\r\nContent-Length: abc\r\n\r\n",   /* non-numeric length */
      "GET / HTTP/1.1\r\nContent-Length: 12abc\r\n\r\n", /* trailing junk */
      "GET / HTTP/1.1\r\nContent-Length: -5\r\n\r\n",    /* negative length */
      "GET / HTTP/1.1\r\nContent-Length: \r\n\r\n",      /* empty length */
      "GET / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n", /* conflicting */
  };

  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
    HttpParser p;
    size_t consumed = 0;
    http_init(&p);
    CHECK(feed_all(&p, bad[i], &consumed) == HTTP_ERROR, "malformed input is rejected");

    /* Once in error, it stays in error rather than recovering into a
     * half-parsed state. */
    CHECK(feed_all(&p, "\r\n", &consumed) == HTTP_ERROR, "the error state is sticky");
  }

  /* A duplicate Content-Length that agrees is not a conflict. */
  HttpParser p;
  size_t consumed = 0;
  http_init(&p);
  CHECK(feed_all(&p, "GET / HTTP/1.1\r\nContent-Length: 7\r\nContent-Length: 7\r\n\r\n",
                 &consumed) == HTTP_DONE,
        "an identical repeated Content-Length is accepted");
  CHECK(p.content_length == 7, "with the agreed value");
}

/* Over-long input must be refused rather than truncated. Truncating turns a
 * header the client sent into a shorter one the application then trusts. */
static void test_oversized_input_is_refused_not_truncated(void) {
  HttpParser p;
  size_t consumed = 0;
  char *huge = malloc(HTTP_MAX_LINE + 512);

  /* A path far longer than the line buffer. */
  int n = snprintf(huge, HTTP_MAX_LINE + 512, "GET /");
  memset(huge + n, 'a', HTTP_MAX_LINE + 100);
  strcpy(huge + n + HTTP_MAX_LINE + 100, " HTTP/1.1\r\n\r\n");

  http_init(&p);
  CHECK(feed_all(&p, huge, &consumed) == HTTP_ERROR, "an over-long line is refused");

  /* A header value longer than the value field. */
  n = snprintf(huge, HTTP_MAX_LINE + 512, "GET / HTTP/1.1\r\nX-Big: ");
  memset(huge + n, 'b', 400);
  strcpy(huge + n + 400, "\r\n\r\n");
  http_init(&p);
  CHECK(feed_all(&p, huge, &consumed) == HTTP_ERROR, "an over-long header value is refused");

  free(huge);

  /* More headers than there are slots. */
  char many[4096];
  size_t off = (size_t)snprintf(many, sizeof many, "GET / HTTP/1.1\r\n");
  for (int i = 0; i < HTTP_MAX_HEADERS + 5; i++) {
    off += (size_t)snprintf(many + off, sizeof many - off, "X-%d: v\r\n", i);
  }
  snprintf(many + off, sizeof many - off, "\r\n");
  http_init(&p);
  CHECK(feed_all(&p, many, &consumed) == HTTP_ERROR, "too many headers is refused");
}

/* A parser must be reusable: a server handles many requests on one connection
 * and re-initialising must fully clear the previous one. */
static void test_reuse_after_init(void) {
  HttpParser p;
  size_t consumed = 0;

  http_init(&p);
  CHECK(feed_all(&p, SIMPLE, &consumed) == HTTP_DONE, "first request parses");
  CHECK(p.header_count == 3, "first request has three headers");

  http_init(&p);
  CHECK(feed_all(&p, "PUT /other HTTP/1.1\r\nA: b\r\n\r\n", &consumed) == HTTP_DONE,
        "second request parses");
  CHECK(strcmp(p.method, "PUT") == 0, "the method was replaced");
  CHECK(strcmp(p.path, "/other") == 0, "the path was replaced");
  CHECK(p.header_count == 1, "headers from the previous request are gone");
  CHECK(p.content_length == -1, "and so is the previous content length");
  CHECK(http_header(&p, "Host") == NULL, "the old headers are truly unreachable");
}

int main(void) {
  test_parses_a_whole_request();
  test_every_possible_split();
  test_body_is_left_for_the_caller();
  test_whitespace_and_line_endings();
  test_malformed_requests_are_rejected();
  test_oversized_input_is_refused_not_truncated();
  test_reuse_after_init();
  printf("ok  c/10-http-parser  %d checks passed\n", checks);
  return 0;
}
