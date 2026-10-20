# HTTP request parser

Every server starts by parsing bytes off a socket, and this is where "just split on `\r\n`" stops working. A socket hands you whatever the network felt like delivering, so the chunk you were given may end in the middle of a header name. This section builds an HTTP/1.1 request-header parser that accepts input in arbitrary chunks, carries just enough state to resume mid-line, and allocates nothing.

It is also a section about security. An HTTP parser that is generous about what it accepts is a parser that disagrees with some other parser in the chain, and every such disagreement is a way to smuggle a second request inside the first. You will see why over-long input must be refused rather than truncated, why `strtol` is the wrong tool for `Content-Length`, and how to tell a leniency that is safe from one that is not. The wire format itself is explained in [HTTP/1.1, JSON, and Server-Sent Events from the wire up](../../01-language-primers/05-http-and-sse.md) in *Programming and Algorithms*; the bounded-copy and overflow habits come from the earlier sections of this chapter.

> **Learning objectives**
> - Parse a line-oriented protocol incrementally, so that every split of the input produces the same result.
> - Bound every field with a fixed buffer and refuse, never truncate, input that does not fit.
> - Identify parser-differential ambiguities (whitespace before a colon, conflicting `Content-Length`) and reject them.
> - Validate a numeric header without `strtol` and without integer overflow.
> - Report exactly where the header block ends so the caller can find the body.

## The interface

[`http.h`](../../../../../practice/build/systems/c/10-http-parser/http.h) declares the interface, [`http.c`](../../../../../practice/build/systems/c/10-http-parser/http.c) is the solution, and [`main.c`](../../../../../practice/build/systems/c/10-http-parser/main.c) is the test suite.

| Function | Does |
|----------|------|
| `http_init(p)` | Reset for a new request |
| `http_feed(p, data, len, &consumed)` | Feed bytes; `HTTP_INCOMPLETE`, `HTTP_DONE`, or `HTTP_ERROR` |
| `http_header(p, name)` | Case-insensitive lookup, `NULL` when absent |

The constraints are strict. Buffers are fixed-size throughout (method 16 bytes, path 512, header name 64, value 256, at most 32 headers, lines up to 2048 bytes), and nothing is allocated. `consumed` must stop at the end of the header block so the caller can find the body. Over-long input is an error, never truncated. And once in error, the parser stays in error.

To build and run the tests from this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c http.c -o http && ./http
# ok  c/10-http-parser  695 checks passed
```

## Incremental input

Incremental parsing is the whole exercise. A socket hands you 3 bytes, then 1400, then 12, with no relationship between those boundaries and the message's. So the parser cannot hold a pointer into the caller's buffer across calls, and must carry enough state to resume mid-line. `test_every_possible_split` feeds the same request at every one of its split points and then one byte at a time, which is the test that fails for anything built around `strtok` or a single pass over a complete buffer.

## Strictness as a security property

Refuse over-long input; never truncate it. Truncating a 3,000-byte path into a 512-byte buffer does not produce an error; it produces a *different request* that the application then trusts. That is a security bug, not a robustness one, and the same reasoning applies to header names, values, and the header count.

Strictness here is a security property, not pedantry. Every rejected case in `test_malformed_requests_are_rejected` is a real parser-differential class. When a proxy and a backend disagree about whether `Foo : bar` is a header, or about which of two `Content-Length` values wins, an attacker can hide a second request inside the first. That is request smuggling, and the defence is that a parser which is unsure must refuse rather than guess.

`strtol` is the wrong tool for `Content-Length`. It accepts `"12abc"`, `"+5"`, leading whitespace, and negatives, and reports success for all of them. A body length the parser and the server disagree about is precisely the smuggling primitive above. Validate that the field is digits and nothing else.

Some leniency is still right. The parser accepts a bare LF where the spec says CRLF. Robustness against real clients requires it, and stripping at most one trailing `\r` costs one line. This is a place where being liberal is *safe*, because both readings produce the same message. Contrast `Foo : bar`, where being liberal changes what the message means, and is therefore not safe. That distinction is the entire art of the robustness principle, and the reason it has fallen out of favour.

Finally, the error state has to be sticky. Once a request is malformed, there is no resynchronisation point: you cannot know where the next valid line starts, and guessing lets an attacker choose. The connection is finished. The tests check that feeding more bytes after an error still reports an error.

## Deriving the design

Seven observations take you from the protocol to the code.

The unit of meaning is a line. The request line and each header line end at `\n`; the header block ends at an empty line. So the parser's job splits in two: assemble lines from arbitrary chunks, then interpret complete lines.

Assembling lines across calls needs a private buffer. The caller's buffer is gone after `http_feed` returns, so bytes of an unfinished line are copied into `p->line`, and the next call appends to it. That buffer, its length, and a state are all the parser has to remember.

There are three meaningful states: before the request line, among headers, and done. There is also an error state that every path can fall into and none can leave.

Byte-at-a-time is the simplest correct loop. Each byte is either `\n` (a line is complete: interpret it) or something to append. Then every split of the input produces exactly the same sequence of complete lines, which is the property the every-split test checks.

Every field has a fixed buffer, so every copy is checked. A copy that does not fit is an error, because truncation would produce a different request than the client sent.

Parse exactly what is allowed and reject the rest. The version must be exactly `HTTP/1.N`; there must be no space before a header's colon; `Content-Length` must be all digits and must not conflict with an earlier one. Anything ambiguous is rejected, because ambiguity between parsers is the attack.

Stop at the blank line. Whatever follows is body, and belongs to the caller, so `consumed` points just past the blank line's `\n`.

## Building it step by step

The bounded copy every field goes through:

```c
/* Copy and terminate, or report that it did not fit. Never truncates. */
static int copy_bounded(char *dst, size_t cap, const char *src, size_t len) {
  if (len >= cap) return 0;   /* >= : one byte is needed for the NUL */
  memcpy(dst, src, len);
  dst[len] = '\0';
  return 1;
}
```

The feed loop, which is the incremental part:

```c
HttpStatus http_feed(HttpParser *p, const char *data, size_t len, size_t *consumed) {
  size_t i = 0;
  *consumed = 0;

  /* Sticky terminal states. */
  if (p->state == ST_ERROR) return HTTP_ERROR;
  if (p->state == ST_DONE) return HTTP_DONE;

  for (; i < len; i++) {
    char c = data[i];

    if (c == '\n') {
      /* A line is complete. Strip one trailing CR, so CRLF and bare LF both
       * work and both mean the same thing. */
      size_t n = p->line_len;
      if (n > 0 && p->line[n - 1] == '\r') n--;

      if (!handle_line(p, p->line, n)) {
        p->state = ST_ERROR;
        *consumed = i + 1;
        return HTTP_ERROR;
      }
      p->line_len = 0;

      if (p->state == ST_DONE) {
        /* Consume the terminator and no more: the rest is body. */
        *consumed = i + 1;
        return HTTP_DONE;
      }
      continue;
    }

    /* Refuse an over-long line rather than truncating it. */
    if (p->line_len >= sizeof p->line) {
      p->state = ST_ERROR;
      *consumed = i;
      return HTTP_ERROR;
    }
    p->line[p->line_len++] = c;
  }

  /* Out of input mid-line: the partial line stays in p->line, which is what
   * lets the next call resume. */
  *consumed = len;
  return HTTP_INCOMPLETE;
}
```

Dispatching a complete line by state:

```c
static int handle_line(HttpParser *p, const char *s, size_t len) {
  if (p->state == ST_REQUEST_LINE) {
    /* Blank lines before the request line are tolerated (RFC 9112 suggests
     * it, for stray CRLFs left by old clients). */
    if (len == 0) return 1;
    if (!parse_request_line(p, s, len)) return 0;
    p->state = ST_HEADERS;
    return 1;
  }

  if (len == 0) {          /* the blank line that ends the headers */
    p->state = ST_DONE;
    return 1;
  }
  return parse_header_line(p, s, len);
}
```

The request line is split on the first two spaces with `memchr`, which works
on a length-delimited buffer and never runs past it:

```c
  /* METHOD SP PATH SP HTTP/1.N, each part non-empty */
  const char *sp1 = memchr(s, ' ', len);
  /* ... method = s[0..sp1), path = up to the second space ... */

  /* Exactly "HTTP/1.N". Anything else is rejected rather than guessed at. */
  if (ver_len != 8 || memcmp(ver, "HTTP/1.", 7) != 0) return 0;
  if (ver[7] < '0' || ver[7] > '9') return 0;

  if (!copy_bounded(p->method, sizeof p->method, s, method_len)) return 0;
  if (!copy_bounded(p->path, sizeof p->path, rest, path_len)) return 0;
  p->version_minor = ver[7] - '0';
```

Header lines, where the smuggling defences live:

```c
  const char *colon = memchr(s, ':', len);
  if (!colon) return 0;                    /* no colon: not a header */

  size_t name_len = (size_t)(colon - s);
  if (name_len == 0) return 0;             /* empty name */

  /* No whitespace between name and colon: "Foo : bar" accepted by one
   * parser and rejected by another is how smuggling gets through. */
  if (s[name_len - 1] == ' ' || s[name_len - 1] == '\t') return 0;

  /* ... strip optional whitespace around the value ... */

  if (p->header_count >= HTTP_MAX_HEADERS) return 0;   /* refuse, never drop */
  HttpHeader *h = &p->headers[p->header_count];
  if (!copy_bounded(h->name, sizeof h->name, s, name_len)) return 0;
  if (!copy_bounded(h->value, sizeof h->value, val, val_len)) return 0;
  p->header_count++;

  if (ci_equal(h->name, "Content-Length")) {
    /* Digits only: strtol would accept "12abc", "+5", or "-5". */
    if (h->value[0] == '\0') return 0;
    long n = 0;
    for (const char *c = h->value; *c; c++) {
      if (*c < '0' || *c > '9') return 0;
      if (n > (long)((1L << 40) / 10)) return 0;   /* absurdly large: refuse */
      n = n * 10 + (*c - '0');
    }
    /* A second, different Content-Length is a conflict, not an update. */
    if (p->content_length >= 0 && p->content_length != n) return 0;
    p->content_length = n;
  }
```

The guard refuses a value once it passes `2^40 / 10` before the next digit, so
`n` never exceeds about `2^40` and `n * 10 + digit` stays far below `LONG_MAX`
where `long` is 64 bits (Linux, macOS). On a platform with a 32-bit `long`
(64-bit Windows) `1L << 40` would itself be undefined; a portable version would
compute in `long long` or compare against `LONG_MAX / 10`. An identical repeated
`Content-Length` is accepted, a different one is an error.

The full solution is in [`http.c`](../../../../../practice/build/systems/c/10-http-parser/http.c).

## Why this design

The parser does O(1) work per input byte: each byte is appended to the line buffer once and scanned a constant number of times when its line completes. Memory is the fixed `HttpParser` struct, about 13 KB, with no allocation.

Copying lines is chosen over pointing into the caller's buffer. Zero-copy parsers (such as picohttpparser) are faster but require the caller to keep the whole header block contiguous and re-feed it. Copying into `p->line` is what makes arbitrary chunking work with no cooperation from the caller.

A line-level state machine is chosen over a character-level one. A full character-level machine (method, space, path, ...) avoids the line buffer, but has many more states and is much harder to read. Since the header block is bounded anyway, the line buffer costs nothing that matters.

Strict is chosen over lenient. Lenient parsing accepts more clients, but every leniency that changes meaning is a disagreement an attacker can use. The solution is lenient only where both readings agree: CRLF versus LF, leading blank lines, and whitespace around values.

The accepted trade is scope: there is no obsolete line folding, no chunked encoding, and no limits configurable at run time. This is a header parser, not a full HTTP stack.

## What can go wrong

The rejected inputs are the point of the exercise, so it is worth seeing them together.

| Input or mistake | Why it is refused or wrong |
|------------------|----------------------------|
| A field longer than its buffer | Truncation would create a different request; refused instead |
| `Foo : bar` | Parsers disagree on the name; a smuggling vector |
| Two different `Content-Length` values | Proxy and backend may pick different ones; refused (identical repeats are accepted) |
| `Content-Length: 12abc`, `+5`, `-5` | Not digits only; `strtol` would accept them |
| A version other than exactly `HTTP/1.N` | Rejected rather than guessed at |
| More than 32 headers | Refused, never silently dropped |
| More bytes after an error | Still an error: there is no safe resynchronisation point |
| Holding pointers into the caller's buffer | They are invalid after `http_feed` returns |

> **Warning:** The `Content-Length` overflow guard uses `1L << 40`, which is correct where `long` is 64 bits (Linux, macOS). On a platform with a 32-bit `long`, such as 64-bit Windows, the shift is itself undefined. A portable version would compute in `long long` or compare against `LONG_MAX / 10`.

## Where this leads

The next layer is chunked transfer encoding, a second state machine layered on the first and where a great many CVEs live, particularly around the interaction between `Transfer-Encoding` and `Content-Length` when both are present. The correct behaviour is to reject the request outright, and working out why is more instructive than any amount of reading about it. The streaming side of HTTP, Server-Sent Events, is covered in [HTTP/1.1, JSON, and Server-Sent Events from the wire up](../../01-language-primers/05-http-and-sse.md).

## Key terms

- **Incremental parser**: a parser that accepts input in arbitrary chunks and resumes where the previous chunk ended.
- **Request smuggling**: an attack that exploits two parsers disagreeing about message boundaries to hide one request inside another.
- **Parser differential**: any input that two conforming-looking parsers interpret differently.
- **Sticky error state**: an error that, once entered, every later call reports, because the stream cannot be safely resynchronised.
- **Robustness principle**: "be liberal in what you accept"; safe only where every reading produces the same message.

## Summary

- The parser assembles lines byte by byte into a private buffer, so any split of the input yields the same lines.
- Every field is copied with a bounded copy that fails rather than truncates.
- Ambiguous input (space before a colon, conflicting or non-digit `Content-Length`, a malformed version) is rejected, because ambiguity is the smuggling vector.
- Leniency is allowed only where both readings agree: bare LF, leading blank lines, whitespace around values.
- `consumed` stops just past the blank line so the caller can find the body, and errors are sticky.

## Review questions

1. Why does `test_every_possible_split` pass for this parser by construction?

   <details><summary>Answer</summary>

   The loop handles one byte at a time and keeps its only cross-call state
   (`state`, `line`, `line_len`) in the parser. Whatever the chunk boundaries,
   the same bytes are appended in the same order and lines complete at the same
   `\n`s, so `handle_line` sees an identical sequence of lines.

   </details>

2. Why does `copy_bounded` fail on `len >= cap` rather than `len > cap`?

   <details><summary>Answer</summary>

   The destination needs room for the bytes plus a terminating NUL, so a string
   of `cap` bytes does not fit in a `cap`-byte buffer. `len == cap` must fail
   too.

   </details>

3. Why is `"Foo : bar"` rejected but `"Foo:   bar  "` accepted?

   <details><summary>Answer</summary>

   Whitespace around the value is optional in the grammar and both parsers in a
   chain will strip it, so accepting it changes nothing. Whitespace before the
   colon is not allowed; some parsers would read the name as `"Foo "` and some as
   `"Foo"`, and that disagreement is a request-smuggling vector, so the only safe
   answer is to refuse.

   </details>

4. After `HTTP_DONE`, what does `consumed` tell the caller, and why does the loop
   return immediately instead of continuing?

   <details><summary>Answer</summary>

   It is the index just past the blank line's `\n`, so `data + consumed` is the
   first byte of the body. Continuing would treat body bytes as header text and
   could mis-parse or reject a valid request.

   </details>

5. Why reject two different `Content-Length` values instead of taking the last
   one?

   <details><summary>Answer</summary>

   A proxy that takes the first and a backend that takes the last would disagree
   about where the body ends, and the leftover bytes would be read as a second,
   attacker-chosen request. Refusing the conflict removes the disagreement. Two
   identical values carry no ambiguity, so they are accepted.

   </details>
