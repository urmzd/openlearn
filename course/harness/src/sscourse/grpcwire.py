"""A unary gRPC client over cleartext HTTP/2, in the standard library.

The disaggregated milestone check (disagg.py) calls an engine's
`tl.engine.v1.EngineControl` (Prefill, Info) and a decode engine's
`tl.kv.v1.KvTransferService.Release` directly, as the gateway does (gw.05).
The harness stays stdlib plus PyYAML, so this module carries the little of
gRPC those calls need instead of a grpcio dependency:

- HTTP/2 with prior knowledge (RFC 9113): the preface, SETTINGS and PING
  acknowledgements, one stream per call, DATA in frames of at most 16 KiB,
  connection and stream WINDOW_UPDATEs for what arrives.
- HPACK (RFC 7541): requests are sent as literals without indexing or
  Huffman coding; responses are decoded in full (static and dynamic table,
  Huffman strings), since servers use all of it.
- gRPC over HTTP/2: the 5-byte length prefix, `grpc-status` and
  `grpc-message` from the trailers (or a trailers-only response).
- The protobuf wire format (varint, fixed32, fixed64, length-delimited) to
  encode and decode the few messages, by field number.

Every call opens a fresh connection and closes it (calls are rare: a handful
per milestone step).
"""

from __future__ import annotations

import socket
import struct
import urllib.parse
from dataclasses import dataclass

# gRPC status codes by number (grpc/doc/statuscodes.md).
CODES = (
    "OK CANCELLED UNKNOWN INVALID_ARGUMENT DEADLINE_EXCEEDED NOT_FOUND "
    "ALREADY_EXISTS PERMISSION_DENIED RESOURCE_EXHAUSTED FAILED_PRECONDITION "
    "ABORTED OUT_OF_RANGE UNIMPLEMENTED INTERNAL UNAVAILABLE DATA_LOSS "
    "UNAUTHENTICATED"
).split()


def code_name(code: int) -> str:
    return CODES[code] if 0 <= code < len(CODES) else f"code {code}"


class RpcError(Exception):
    """A call that did not end with grpc-status 0. `code` is the gRPC status
    (UNAVAILABLE, 14, when the connection itself failed)."""

    def __init__(self, code: int, message: str):
        super().__init__(f"{code_name(code)}: {message}")
        self.code = code
        self.message = message

    @property
    def name(self) -> str:
        return code_name(self.code)


# ---------------------------------------------------------------------------
# protobuf wire format

VARINT, FIXED64, LEN, FIXED32 = 0, 1, 2, 5


def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64  # int32/int64 negatives are ten-byte two's complement
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7
        if shift > 63:
            raise ValueError("varint longer than 10 bytes")


def field_varint(num: int, value: int) -> bytes:
    if not value:
        return b""  # proto3: a default scalar is not on the wire
    return _varint(num << 3 | VARINT) + _varint(int(value))


def field_float(num: int, value: float) -> bytes:
    if not value:
        return b""
    return _varint(num << 3 | FIXED32) + struct.pack("<f", float(value))


def field_bytes(num: int, value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode()
    if not value:
        return b""
    return _varint(num << 3 | LEN) + _varint(len(value)) + value


def field_message(num: int, encoded: bytes) -> bytes:
    """An embedded message, present even when empty."""
    return _varint(num << 3 | LEN) + _varint(len(encoded)) + encoded


def field_packed(num: int, values) -> bytes:
    body = b"".join(_varint(int(v)) for v in values)
    return field_bytes(num, body) if body else b""


def decode(buf: bytes) -> dict[int, list]:
    """Field number -> the values in wire order: ints for varint and fixed
    fields (fixed32 as its raw u32), bytes for length-delimited ones."""
    out: dict[int, list] = {}
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == VARINT:
            v, i = _read_varint(buf, i)
        elif wt == FIXED64:
            v = struct.unpack_from("<Q", buf, i)[0]
            i += 8
        elif wt == FIXED32:
            v = struct.unpack_from("<I", buf, i)[0]
            i += 4
        elif wt == LEN:
            n, i = _read_varint(buf, i)
            if i + n > len(buf):
                raise ValueError("truncated length-delimited field")
            v = bytes(buf[i : i + n])
            i += n
        else:
            raise ValueError(f"unsupported wire type {wt}")
        out.setdefault(num, []).append(v)
    return out


def one(fields: dict, num: int, default=0):
    vals = fields.get(num)
    return vals[-1] if vals else default


def as_int32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def as_float(v: int) -> float:
    return struct.unpack("<f", struct.pack("<I", v))[0]


def as_double(v: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", v))[0]


def packed_varints(fields: dict, num: int) -> list[int]:
    """A repeated scalar: packed (one LEN value) or unpacked (many varints)."""
    out: list[int] = []
    for v in fields.get(num, []):
        if isinstance(v, bytes):
            i = 0
            while i < len(v):
                x, i = _read_varint(v, i)
                out.append(x)
        else:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# HPACK

_STATIC_NAMES = (
    ":authority :method :method :path :path :scheme :scheme :status :status "
    ":status :status :status :status :status accept-charset accept-encoding "
    "accept-language accept-ranges accept access-control-allow-origin age "
    "allow authorization cache-control content-disposition content-encoding "
    "content-language content-length content-location content-range "
    "content-type cookie date etag expect expires from host if-match "
    "if-modified-since if-none-match if-range if-unmodified-since "
    "last-modified link location max-forwards proxy-authenticate "
    "proxy-authorization range referer refresh retry-after server set-cookie "
    "strict-transport-security transfer-encoding user-agent vary via "
    "www-authenticate"
).split()
_STATIC_VALUES = {
    2: "GET",
    3: "POST",
    4: "/",
    5: "/index.html",
    6: "http",
    7: "https",
    8: "200",
    9: "204",
    10: "206",
    11: "304",
    12: "400",
    13: "404",
    14: "500",
    16: "gzip, deflate",
}
STATIC = [(n, _STATIC_VALUES.get(i + 1, "")) for i, n in enumerate(_STATIC_NAMES)]

# RFC 7541 Appendix B: the Huffman code of each byte, "<hex code>/<bits>".
_HUFFMAN = (
    "1ff8/13 7fffd8/23 fffffe2/28 fffffe3/28 fffffe4/28 fffffe5/28 "
    "fffffe6/28 fffffe7/28 fffffe8/28 ffffea/24 3ffffffc/30 fffffe9/28 "
    "fffffea/28 3ffffffd/30 fffffeb/28 fffffec/28 fffffed/28 fffffee/28 "
    "fffffef/28 ffffff0/28 ffffff1/28 ffffff2/28 3ffffffe/30 ffffff3/28 "
    "ffffff4/28 ffffff5/28 ffffff6/28 ffffff7/28 ffffff8/28 ffffff9/28 "
    "ffffffa/28 ffffffb/28 14/6 3f8/10 3f9/10 ffa/12 1ff9/13 15/6 f8/8 "
    "7fa/11 3fa/10 3fb/10 f9/8 7fb/11 fa/8 16/6 17/6 18/6 0/5 1/5 2/5 19/6 "
    "1a/6 1b/6 1c/6 1d/6 1e/6 1f/6 5c/7 fb/8 7ffc/15 20/6 ffb/12 3fc/10 "
    "1ffa/13 21/6 5d/7 5e/7 5f/7 60/7 61/7 62/7 63/7 64/7 65/7 66/7 67/7 "
    "68/7 69/7 6a/7 6b/7 6c/7 6d/7 6e/7 6f/7 70/7 71/7 72/7 fc/8 73/7 fd/8 "
    "1ffb/13 7fff0/19 1ffc/13 3ffc/14 22/6 7ffd/15 3/5 23/6 4/5 24/6 5/5 "
    "25/6 26/6 27/6 6/5 74/7 75/7 28/6 29/6 2a/6 7/5 2b/6 76/7 2c/6 8/5 "
    "9/5 2d/6 77/7 78/7 79/7 7a/7 7b/7 7ffe/15 7fc/11 3ffd/14 1ffd/13 "
    "ffffffc/28 fffe6/20 3fffd2/22 fffe7/20 fffe8/20 3fffd3/22 3fffd4/22 "
    "3fffd5/22 7fffd9/23 3fffd6/22 7fffda/23 7fffdb/23 7fffdc/23 7fffdd/23 "
    "7fffde/23 ffffeb/24 7fffdf/23 ffffec/24 ffffed/24 3fffd7/22 7fffe0/23 "
    "ffffee/24 7fffe1/23 7fffe2/23 7fffe3/23 7fffe4/23 1fffdc/21 3fffd8/22 "
    "7fffe5/23 3fffd9/22 7fffe6/23 7fffe7/23 ffffef/24 3fffda/22 1fffdd/21 "
    "fffe9/20 3fffdb/22 3fffdc/22 7fffe8/23 7fffe9/23 1fffde/21 7fffea/23 "
    "3fffdd/22 3fffde/22 fffff0/24 1fffdf/21 3fffdf/22 7fffeb/23 7fffec/23 "
    "1fffe0/21 1fffe1/21 3fffe0/22 1fffe2/21 7fffed/23 3fffe1/22 7fffee/23 "
    "7fffef/23 fffea/20 3fffe2/22 3fffe3/22 3fffe4/22 7ffff0/23 3fffe5/22 "
    "3fffe6/22 7ffff1/23 3ffffe0/26 3ffffe1/26 fffeb/20 7fff1/19 3fffe7/22 "
    "7ffff2/23 3fffe8/22 1ffffec/25 3ffffe2/26 3ffffe3/26 3ffffe4/26 "
    "7ffffde/27 7ffffdf/27 3ffffe5/26 fffff1/24 1ffffed/25 7fff2/19 "
    "1fffe3/21 3ffffe6/26 7ffffe0/27 7ffffe1/27 3ffffe7/26 7ffffe2/27 "
    "fffff2/24 1fffe4/21 1fffe5/21 3ffffe8/26 3ffffe9/26 ffffffd/28 "
    "7ffffe3/27 7ffffe4/27 7ffffe5/27 fffec/20 fffff3/24 fffed/20 "
    "1fffe6/21 3fffe9/22 1fffe7/21 1fffe8/21 7ffff3/23 3fffea/22 3fffeb/22 "
    "1ffffee/25 1ffffef/25 fffff4/24 fffff5/24 3ffffea/26 7ffff4/23 "
    "3ffffeb/26 7ffffe6/27 3ffffec/26 3ffffed/26 7ffffe7/27 7ffffe8/27 "
    "7ffffe9/27 7ffffea/27 7ffffeb/27 ffffffe/28 7ffffec/27 7ffffed/27 "
    "7ffffee/27 7ffffef/27 7fffff0/27 3ffffee/26 "
).split()
_HUFF_DECODE = {
    (int(c, 16), int(n)): sym
    for sym, (c, n) in enumerate(x.split("/") for x in _HUFFMAN)
}


def huffman_decode(data: bytes) -> bytes:
    out = bytearray()
    code = nbits = 0
    for byte in data:
        for k in range(7, -1, -1):
            code = code << 1 | (byte >> k) & 1
            nbits += 1
            sym = _HUFF_DECODE.get((code, nbits))
            if sym is not None:
                out.append(sym)
                code = nbits = 0
            elif nbits > 30:
                raise ValueError("invalid Huffman code")
    # The padding is the most significant bits of EOS (all ones), < 8 bits.
    if nbits > 7 or code != (1 << nbits) - 1:
        raise ValueError("invalid Huffman padding")
    return bytes(out)


def huffman_encode(data: bytes) -> bytes:
    """Used by the tests' fake server; requests are sent as plain literals."""
    acc = nbits = 0
    for b in data:
        c, n = _HUFFMAN[b].split("/")
        acc = acc << int(n) | int(c, 16)
        nbits += int(n)
    pad = -nbits % 8
    acc = acc << pad | (1 << pad) - 1
    return (acc.to_bytes((nbits + pad) // 8, "big")) if nbits else b""


def _int_encode(value: int, prefix: int, first: int) -> bytes:
    limit = (1 << prefix) - 1
    if value < limit:
        return bytes([first | value])
    out = bytearray([first | limit])
    value -= limit
    while value >= 128:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _int_decode(buf: bytes, i: int, prefix: int) -> tuple[int, int]:
    limit = (1 << prefix) - 1
    v = buf[i] & limit
    i += 1
    if v < limit:
        return v, i
    shift = 0
    while True:
        b = buf[i]
        i += 1
        v += (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return v, i


def _str_encode(s: str, huffman: bool = False) -> bytes:
    raw = s.encode()
    if huffman:
        h = huffman_encode(raw)
        return _int_encode(len(h), 7, 0x80) + h
    return _int_encode(len(raw), 7, 0) + raw


def _str_decode(buf: bytes, i: int) -> tuple[str, int]:
    huff = buf[i] & 0x80
    n, i = _int_decode(buf, i, 7)
    raw = buf[i : i + n]
    if len(raw) != n:
        raise ValueError("truncated header string")
    return (huffman_decode(raw) if huff else bytes(raw)).decode(
        "utf-8", "replace"
    ), i + n


def hpack_encode(headers: list[tuple[str, str]], huffman: bool = False) -> bytes:
    """Literal header fields without indexing, new names (RFC 7541 6.2.2):
    no table state on either side."""
    return b"".join(
        b"\x00" + _str_encode(k, huffman) + _str_encode(v, huffman) for k, v in headers
    )


class HpackDecoder:
    """One connection's decoding context (its dynamic table)."""

    def __init__(self, max_size: int = 4096):
        self.dynamic: list[tuple[str, str]] = []  # newest first
        self.max_size = max_size

    def _size(self) -> int:
        return sum(len(k.encode()) + len(v.encode()) + 32 for k, v in self.dynamic)

    def _evict(self) -> None:
        while self.dynamic and self._size() > self.max_size:
            self.dynamic.pop()

    def _get(self, idx: int) -> tuple[str, str]:
        if 1 <= idx <= len(STATIC):
            return STATIC[idx - 1]
        j = idx - len(STATIC) - 1
        if 0 <= j < len(self.dynamic):
            return self.dynamic[j]
        raise ValueError(f"HPACK index {idx} out of range")

    def decode(self, block: bytes) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        i = 0
        while i < len(block):
            b = block[i]
            if b & 0x80:  # indexed field
                idx, i = _int_decode(block, i, 7)
                out.append(self._get(idx))
            elif b & 0x40:  # literal with incremental indexing
                idx, i = _int_decode(block, i, 6)
                name = self._get(idx)[0] if idx else None
                if name is None:
                    name, i = _str_decode(block, i)
                value, i = _str_decode(block, i)
                out.append((name, value))
                self.dynamic.insert(0, (name, value))
                self._evict()
            elif b & 0x20:  # dynamic table size update
                self.max_size, i = _int_decode(block, i, 5)
                self._evict()
            else:  # literal without indexing (0000) or never indexed (0001)
                idx, i = _int_decode(block, i, 4)
                name = self._get(idx)[0] if idx else None
                if name is None:
                    name, i = _str_decode(block, i)
                value, i = _str_decode(block, i)
                out.append((name, value))
        return out


# ---------------------------------------------------------------------------
# HTTP/2 frames

PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
# Frame types (RFC 9113 6).
DATA, HEADERS, RST_STREAM, SETTINGS = 0, 1, 3, 4
PING, GOAWAY, WINDOW_UPDATE, CONTINUATION = 6, 7, 8, 9
END_STREAM, ACK, END_HEADERS, PADDED, PRIORITY = 0x1, 0x1, 0x4, 0x8, 0x20
MAX_FRAME = 16384
WINDOW = 1 << 24  # what this client lets a server send before it acknowledges


def frame(ftype: int, flags: int, stream: int, payload: bytes = b"") -> bytes:
    return (
        struct.pack(">I", len(payload))[1:]
        + bytes([ftype, flags])
        + struct.pack(">I", stream & 0x7FFFFFFF)
        + payload
    )


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("the server closed the connection")
        buf += chunk
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[int, int, int, bytes]:
    head = _recv_exact(sock, 9)
    length = int.from_bytes(head[:3], "big")
    ftype, flags = head[3], head[4]
    stream = int.from_bytes(head[5:9], "big") & 0x7FFFFFFF
    return ftype, flags, stream, _recv_exact(sock, length)


def _strip_padding(flags: int, payload: bytes, headers: bool) -> bytes:
    if flags & PADDED:
        pad = payload[0]
        payload = payload[1 : len(payload) - pad]
    if headers and flags & PRIORITY:
        payload = payload[5:]
    return payload


@dataclass
class Reply:
    status: int  # grpc-status
    message: str  # grpc-message, percent-decoded
    payload: bytes | None  # the response message (None when there is none)


def _target(addr: str) -> tuple[str, int]:
    if "://" in addr:
        addr = urllib.parse.urlsplit(addr).netloc
    host, _, port = addr.rpartition(":")
    if not host or not port.isdigit():
        raise RpcError(3, f"target {addr!r} is not host:port")
    return host.strip("[]"), int(port)


def call(addr: str, method: str, request: bytes, timeout: float = 30.0) -> bytes:
    """One unary call: `method` is "/package.Service/Method". Returns the
    response message bytes; raises RpcError on a non-OK status."""
    r = call_raw(addr, method, request, timeout)
    if r.status != 0:
        raise RpcError(r.status, r.message)
    if r.payload is None:
        raise RpcError(13, "the server sent grpc-status 0 and no message")
    return r.payload


def call_raw(addr: str, method: str, request: bytes, timeout: float = 30.0) -> Reply:
    host, port = _target(addr)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as e:
        raise RpcError(14, f"connect {addr}: {e}") from None
    try:
        return _exchange(sock, f"{host}:{port}", method, request)
    except (OSError, ConnectionError, ValueError, IndexError) as e:
        raise RpcError(14, f"{method} on {addr}: {type(e).__name__}: {e}") from None
    finally:
        sock.close()


def _exchange(
    sock: socket.socket, authority: str, method: str, request: bytes
) -> Reply:
    sid = 1
    # SETTINGS: no server push, a large initial stream window.
    settings = struct.pack(">HI", 2, 0) + struct.pack(">HI", 4, WINDOW)
    out = PREFACE + frame(SETTINGS, 0, 0, settings)
    out += frame(WINDOW_UPDATE, 0, 0, struct.pack(">I", WINDOW - 65535))
    block = hpack_encode(
        [
            (":method", "POST"),
            (":scheme", "http"),
            (":path", method),
            (":authority", authority),
            ("content-type", "application/grpc"),
            ("te", "trailers"),
            ("user-agent", "sscourse-grpcwire/1"),
        ]
    )
    out += frame(HEADERS, END_HEADERS, sid, block)
    body = b"\x00" + struct.pack(">I", len(request)) + request
    if len(body) > 65535:  # the default initial window; requests here are small
        raise ValueError(f"request of {len(body)} bytes exceeds the initial window")
    chunks = [body[i : i + MAX_FRAME] for i in range(0, len(body), MAX_FRAME)] or [b""]
    for k, c in enumerate(chunks):
        out += frame(DATA, END_STREAM if k == len(chunks) - 1 else 0, sid, c)
    sock.sendall(out)

    hp = HpackDecoder()
    header_sets: list[list[tuple[str, str]]] = []
    pending = b""
    headers_end = False
    data = bytearray()
    while True:
        ftype, flags, stream, payload = read_frame(sock)
        if ftype == SETTINGS:
            if not flags & ACK:
                sock.sendall(frame(SETTINGS, ACK, 0))
        elif ftype == PING:
            if not flags & ACK:
                sock.sendall(frame(PING, ACK, 0, payload))
        elif ftype == GOAWAY:
            last = int.from_bytes(payload[:4], "big") & 0x7FFFFFFF
            err = int.from_bytes(payload[4:8], "big")
            if last < sid:
                raise ConnectionError(
                    f"GOAWAY (error {err}) before the call: {payload[8:]!r}"
                )
        elif ftype == RST_STREAM and stream == sid:
            raise ConnectionError(
                f"stream reset by the server (HTTP/2 error {int.from_bytes(payload[:4], 'big')})"
            )
        elif ftype in (HEADERS, CONTINUATION) and stream == sid:
            if ftype == HEADERS:
                headers_end = bool(flags & END_STREAM)
            pending += _strip_padding(flags, payload, ftype == HEADERS)
            if flags & END_HEADERS:
                header_sets.append(hp.decode(pending))
                pending = b""
                if headers_end:  # trailers, or a trailers-only response
                    break
        elif ftype == DATA and stream == sid:
            payload = _strip_padding(flags, payload, False)
            data += payload
            if payload:  # give the window back so a long reply keeps flowing
                inc = struct.pack(">I", len(payload))
                sock.sendall(frame(WINDOW_UPDATE, 0, 0, inc))
                if not flags & END_STREAM:
                    sock.sendall(frame(WINDOW_UPDATE, 0, sid, inc))
            if flags & END_STREAM:
                break
    return _reply(header_sets, bytes(data))


def _reply(header_sets: list[list[tuple[str, str]]], data: bytes) -> Reply:
    merged: dict[str, str] = {}
    for hs in header_sets:
        for k, v in hs:
            merged[k] = v
    status = merged.get(":status", "")
    if status != "200":
        return Reply(2 if status else 13, f"HTTP status {status or '(none)'}", None)
    if "grpc-status" not in merged:
        return Reply(13, "the response ended without grpc-status", None)
    try:
        code = int(merged["grpc-status"])
    except ValueError:
        return Reply(2, f"grpc-status {merged['grpc-status']!r}", None)
    msg = urllib.parse.unquote(merged.get("grpc-message", ""))
    payload = None
    if len(data) >= 5:
        if data[0] != 0:
            return Reply(12, "compressed responses are not supported", None)
        n = struct.unpack(">I", data[1:5])[0]
        payload = data[5 : 5 + n]
        if len(payload) != n:
            return Reply(13, "truncated response message", None)
    return Reply(code, msg, payload)
