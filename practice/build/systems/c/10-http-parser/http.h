/* http.h - an incremental HTTP/1.1 request parser.
 *
 * The interface. http.c implements it and main.c tests it.
 *
 * "Incremental" is the whole exercise. Bytes arrive from a socket in whatever
 * chunks the network felt like, with no relationship to message boundaries, so
 * the parser must be able to stop mid-header and resume when more arrives.
 * That is why it is a state machine and not a series of calls to strtok.
 */
#ifndef HTTP_H
#define HTTP_H

#include <stddef.h>

#define HTTP_MAX_HEADERS 32
#define HTTP_MAX_LINE 2048

typedef enum {
  HTTP_INCOMPLETE = 0, /* need more bytes */
  HTTP_DONE,           /* headers fully parsed */
  HTTP_ERROR           /* malformed, and unrecoverable */
} HttpStatus;

typedef struct {
  char name[64];
  char value[256];
} HttpHeader;

typedef struct {
  char method[16];
  char path[512];
  int version_minor; /* the 1 in HTTP/1.1 */

  HttpHeader headers[HTTP_MAX_HEADERS];
  size_t header_count;

  long content_length; /* -1 when absent */

  /* Internals. Do not read these from outside the parser. */
  int state;
  char line[HTTP_MAX_LINE];
  size_t line_len;
} HttpParser;

/* Reset to the start state, ready for a new request. */
void http_init(HttpParser *p);

/* Feed `len` bytes. May be called repeatedly with any split of the input,
 * including one byte at a time.
 *
 * Returns HTTP_INCOMPLETE if more is needed, HTTP_DONE once the blank line
 * after the headers has been seen, or HTTP_ERROR. Once it returns HTTP_DONE or
 * HTTP_ERROR, further feeding is undefined until http_init is called again.
 *
 * `consumed` receives how many bytes were used, so the caller can find the
 * start of the body after HTTP_DONE. */
HttpStatus http_feed(HttpParser *p, const char *data, size_t len, size_t *consumed);

/* Case-insensitive header lookup. NULL when absent. */
const char *http_header(const HttpParser *p, const char *name);

#endif /* HTTP_H */
