"""The disagg suite and greedy-equal (disagg.py) and the stdlib gRPC client
under them (grpcwire.py), against in-process fakes: a cleartext HTTP/2 server
that answers like a gRPC server (Huffman-coded headers, trailers), fake
engines over HTTP, and a fake Prefill that pushes bytes to the decode
engine's kv port, so the harness's reset proxy has something to cut."""

import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sscourse import disagg, grpcwire as g
from sscourse.matchers import MATCHERS, NO_COMMAND

# -- HPACK and protobuf --------------------------------------------------------


def test_huffman_and_hpack_match_rfc7541_vectors():
    h = bytes.fromhex("f1e3c2e5f23a6ba0ab90f4ff")  # C.4.1
    assert g.huffman_decode(h) == b"www.example.com"
    assert g.huffman_encode(b"www.example.com") == h
    d = g.HpackDecoder()  # C.4: three requests share one dynamic table
    assert d.decode(bytes.fromhex("828684418cf1e3c2e5f23a6ba0ab90f4ff"))[-1] == (
        ":authority",
        "www.example.com",
    )
    assert d.decode(bytes.fromhex("828684be5886a8eb10649cbf"))[-1] == (
        "cache-control",
        "no-cache",
    )
    assert d.decode(bytes.fromhex("828785bf408825a849e95ba97d7f8925a849e95bb8e8b4bf"))[
        -1
    ] == ("custom-key", "custom-value")
    d = g.HpackDecoder(256)  # C.6.1, a response, with eviction later
    got = d.decode(
        bytes.fromhex(
            "488264025885aec3771a4b6196d07abe941054d444a8200595040b8166e082a62d1bff"
            "6e919d29ad171863c78f0b97c8e9ae82ae43d3"
        )
    )
    assert got[0] == (":status", "302") and got[-1] == (
        "location",
        "https://www.example.com",
    )
    assert d.decode(bytes.fromhex("4883640effc1c0bf"))[0] == (":status", "307")
    with pytest.raises(ValueError):
        g.huffman_decode(b"\x00")  # padding that is not EOS bits


def test_protobuf_round_trip_of_the_engine_messages():
    req = {
        "request_id": "r1",
        "model": "tracer",
        "prompt_ids": [1, 300, 70000],
        "sampling": {"temperature": 0.5, "seed": 2**63, "max_new_tokens": 8},
        "decode_target": "127.0.0.1:9",
        "priority": -3,
    }
    f = g.decode(disagg.encode_prefill(req))
    assert g.packed_varints(f, 3) == [1, 300, 70000]
    assert g.as_int32(g.one(f, 7)) == -3
    s = g.decode(g.one(f, 4))
    assert g.as_float(g.one(s, 1)) == 0.5 and g.one(s, 8) == 2**63
    # proto3: zero scalars are absent, so greedy (temperature 0) sends nothing
    assert 1 not in g.decode(disagg.encode_sampling({"temperature": 0}))
    handle = (
        g.field_bytes(1, "h1")
        + g.field_packed(4, [2**64 - 1, 5])
        + g.field_varint(6, 42)
        + g.field_varint(8, 1)
    )
    out = disagg.decode_prefill(
        g.field_message(1, handle) + g.field_varint(4, 7) + g.field_float(3, -1.5)
    )
    assert out["handle"]["handle_id"] == "h1"
    assert out["handle"]["block_hashes"] == [2**64 - 1, 5]
    assert out["handle"]["rng_draws_consumed"] == 1
    assert (out["prompt_tokens"], out["first_logprob"]) == (7, -1.5)


# -- a gRPC server over h2c, for the client ---------------------------------------


class H2Server:
    """Answers each connection's single unary call with `handler(path, msg)`
    -> (grpc status, message, response bytes or None), the way grpc-go and
    tonic do: SETTINGS first, Huffman-coded headers, trailers after DATA."""

    def __init__(self, handler):
        self.handler = handler
        self.ln = socket.socket()
        self.ln.bind(("127.0.0.1", 0))
        self.ln.listen(8)
        self.addr = f"127.0.0.1:{self.ln.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.ln.accept()
            except OSError:
                return
            threading.Thread(target=self._conn, args=(c,), daemon=True).start()

    def _conn(self, c):
        with c:
            assert g._recv_exact(c, len(g.PREFACE)) == g.PREFACE
            c.sendall(g.frame(g.SETTINGS, 0, 0, struct.pack(">HI", 3, 100)))
            c.sendall(g.frame(g.PING, 0, 0, b"12345678"))
            hp, path, body = g.HpackDecoder(), None, b""
            while True:
                t, flags, sid, p = g.read_frame(c)
                if t == g.HEADERS:
                    path = dict(hp.decode(p))[":path"]
                elif t == g.DATA:
                    body += p
                    if flags & g.END_STREAM:
                        break
            status, msg, resp = self.handler(path, body[5:])
            if resp is None:  # trailers-only
                block = b"\x88" + g.hpack_encode(
                    [("grpc-status", str(status)), ("grpc-message", msg)], True
                )
                c.sendall(g.frame(g.HEADERS, g.END_HEADERS | g.END_STREAM, sid, block))
            else:
                head = b"\x88" + g.hpack_encode(
                    [("content-type", "application/grpc")], True
                )
                c.sendall(g.frame(g.HEADERS, g.END_HEADERS, sid, head))
                data = b"\x00" + struct.pack(">I", len(resp)) + resp
                c.sendall(g.frame(g.DATA, 0, sid, data))
                trailers = g.hpack_encode([("grpc-status", str(status))], True)
                c.sendall(
                    g.frame(g.HEADERS, g.END_HEADERS | g.END_STREAM, sid, trailers)
                )
            try:
                while c.recv(4096):
                    pass
            except OSError:
                pass

    def close(self):
        # On Linux, close() alone leaves a thread blocked in accept() holding the
        # listener, so a later connect still lands here; shutdown wakes it first.
        try:
            self.ln.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.ln.close()


def test_unary_call_ok_error_and_unreachable():
    seen = {}

    def handler(path, msg):
        seen[path] = msg
        if path.endswith("/Info"):
            return 0, "", g.field_bytes(1, "tracer") + g.field_bytes(2, "prefill")
        return 9, "this engine serves ['decode'], not prefill", None

    srv = H2Server(handler)
    try:
        assert disagg.Rpc().info(srv.addr)["roles"] == ["prefill"]
        with pytest.raises(g.RpcError) as e:
            disagg.Rpc().prefill(srv.addr, {"prompt_ids": [1, 2], "model": "m"})
        assert e.value.name == "FAILED_PRECONDITION"
        assert "not prefill" in e.value.message
        f = g.decode(seen["/tl.engine.v1.EngineControl/Prefill"])
        assert g.packed_varints(f, 3) == [1, 2]
    finally:
        srv.close()
    with pytest.raises(g.RpcError) as e:
        g.call(srv.addr, "/x.Y/Z", b"", timeout=2)
    assert e.value.name == "UNAVAILABLE"


# -- the reset proxy ---------------------------------------------------------------


def _sink():
    """A TCP server that reads everything and answers 'ok' at EOF."""
    ln = socket.socket()
    ln.bind(("127.0.0.1", 0))
    ln.listen(8)

    def serve():
        while True:
            try:
                c, _ = ln.accept()
            except OSError:
                return

            def one(c=c):
                with c:
                    try:
                        while c.recv(65536):
                            pass
                        c.sendall(b"ok")
                    except OSError:
                        pass

            threading.Thread(target=one, daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    return ln, f"127.0.0.1:{ln.getsockname()[1]}"


def _push(addr: str, n: int) -> bytes:
    host, _, port = addr.rpartition(":")
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(b"x" * n)
        s.shutdown(socket.SHUT_WR)
        out = b""
        while chunk := s.recv(100):
            out += chunk
        return out


def test_reset_proxy_cuts_only_past_the_byte_budget():
    ln, up = _sink()
    p = disagg.ResetProxy(up, 1000)
    try:
        assert _push(p.addr, 500) == b"ok"
        with pytest.raises(OSError):
            _push(p.addr, 50_000)
        assert (p.connections, p.resets) == (2, 1)
    finally:
        p.close()
        ln.close()


# -- the suite against fake engines ---------------------------------------------------


def _gen(prompt: str, body: dict) -> str:
    """The fake model: deterministic in the prompt, the seed, and greedy-ness."""
    salt = "" if not body.get("temperature") else f"~{body.get('seed')}"
    return (prompt[::-1] + salt)[: int(body.get("max_tokens", 16))]


class FakeEngine:
    """/v1/completions (plain and SSE), /v1/tokenize, /metrics. A decode
    engine resumes handles from `handles` and 404s unknown ones; its used KV
    blocks are the handles it holds (plus `leak`)."""

    def __init__(self, name, broken=None):
        self.name, self.broken = name, broken or set()
        self.handles: dict[str, str] = {}
        self.leak = 0
        self.suffix = ""  # greedy-equal: an engine whose greedy text differs
        eng = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, status, obj):
                b = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                used = len(eng.handles) + eng.leak
                b = f'# TYPE x gauge\ntl_engine_kv_blocks{{state="free"}} 9\ntl_engine_kv_blocks{{state="used"}} {used}\n'.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v1/tokenize":
                    return self._json(200, {"ids": [1, *body["text"].encode()]})
                h = self.headers.get("X-TL-KV-Handle")
                if h is not None:
                    if h not in eng.handles:
                        return self._json(
                            404,
                            {
                                "error": {
                                    "message": "unknown",
                                    "type": "invalid_request_error",
                                    "param": None,
                                    "code": "kv_handle_not_found",
                                }
                            },
                        )
                    eng.handles.pop(h)
                text = _gen(body["prompt"], body) + eng.suffix
                if h is not None and "seeded" in eng.broken and body.get("temperature"):
                    text = text.upper()
                if not body.get("stream"):
                    return self._json(
                        200,
                        {
                            "choices": [{"text": text}],
                            "usage": {"completion_tokens": len(text)},
                        },
                    )
                chunks = [{"choices": [{"text": ch}]} for ch in text]
                out = (
                    "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
                    + "data: [DONE]\n\n"
                )
                b = out.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.kv_ln, self.kv = _sink()

    def engine(self):
        base = f"http://127.0.0.1:{self.http.server_address[1]}"
        return disagg.Engine(self.name, base, base, f"grpc-{self.name}", self.kv)

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.kv_ln.close()


class FakeRpc:
    """Prefill pushes ~6 KB to decode_target (through the proxy in the reset
    case) and registers the handle on the decode engine whose kv port it is."""

    def __init__(self, engines, broken=()):
        self.by_kv = {e.kv: e for e in engines}
        self.by_grpc = {f"grpc-{e.name}": e for e in engines}
        self.broken = set(broken)

    def info(self, addr):
        e = self.by_grpc[addr]
        role = (
            "prefill"
            if e.name == "p"
            else "decode"
            if e.name.startswith("d")
            else "unified"
        )
        return {
            "model": "tracer",
            "roles": [role],
            "kv_format": 1,
            "kv_formats_read": [1],
        }

    def prefill(self, addr, req):
        target = req["decode_target"]
        try:
            _push(target, 6000)
        except OSError as e:
            if "ignore-reset" not in self.broken:
                if "leak" in self.broken:
                    self.by_grpc[addr].leak += 1
                raise g.RpcError(14, f"kv transfer: {e}") from None
        hid = req["request_id"]
        d = self.by_kv.get(target)
        if d is not None:
            d.handles[hid] = req["model"]
        return {"handle": {"handle_id": hid}, "prompt_tokens": len(req["prompt_ids"])}

    def release(self, addr, handle_id):
        self.by_kv[addr].handles.pop(handle_id, None)


@pytest.fixture
def fakes():
    made = []

    def make(broken_rpc=(), broken_decode=None):
        es = [
            FakeEngine("u"),
            FakeEngine("p"),
            FakeEngine("da", broken_decode),
            FakeEngine("db"),
        ]
        made.extend(es)
        rpc = FakeRpc(es, broken_rpc)
        u, p, da, db = (e.engine() for e in es)
        return disagg.Suite(u, p, [da, db], "tracer", rpc=rpc, settle_s=0.5)

    yield make
    for e in made:
        e.close()


def _status(results):
    return {r.case: r.status for r in results}


def test_the_suite_passes_a_correct_disaggregated_pair(fakes):
    res = fakes().run()
    assert all(r.status == "pass" for r in res), [(r.case, r.detail) for r in res]
    assert [r.case for r in res] == [c for c, _ in disagg.Suite.CASES]


def test_a_broken_rng_hand_off_fails_only_the_seeded_case(fakes):
    st = _status(fakes(broken_decode={"seeded"}).run())
    assert st.pop("seeded.equals-unified") == "fail"
    assert set(st.values()) == {"pass"}


def test_a_prefill_that_ignores_the_reset_fails(fakes):
    res = {r.case: r for r in fakes(broken_rpc={"ignore-reset"}).run()}
    assert res["reset.prefill-fails"].status == "fail"
    assert "want an error" in res["reset.prefill-fails"].detail


def test_blocks_leaked_by_a_reset_fail_the_case(fakes):
    res = {r.case: r for r in fakes(broken_rpc={"leak"}).run()}
    assert res["reset.prefill-fails"].status == "fail"
    assert "leaked" in res["reset.prefill-fails"].detail


def test_greedy_equal_compares_text_and_token_counts():
    a, b = FakeEngine("plain"), FakeEngine("spec")
    try:
        ea, eb = a.engine(), b.engine()
        assert disagg.greedy_equal([ea, eb], "tracer", ["abc", "hello"], 8) == []
        errs = disagg.greedy_equal(
            [ea, eb], "tracer", ["abc"], 2
        ) + disagg.greedy_equal([ea], "tracer", ["abc"], 3)
        assert errs == []
        b.suffix = "!"
        errs = disagg.greedy_equal([ea, eb], "tracer", ["abc"], 8)
        assert (
            len(errs) == 1
            and "spec 'cba!' (4 tokens) != plain 'cba' (3 tokens)" in errs[0]
        )
        broken = disagg.Engine("dead", "http://127.0.0.1:9", "", "", "")
        errs = disagg.greedy_equal([ea, broken], "tracer", ["abc"], 8, timeout=1)
        assert errs and errs[0].startswith("dead:")
    finally:
        a.close()
        b.close()


def test_the_matchers_are_registered_as_command_free():
    for m in ("disagg", "greedy-equal"):
        assert m in MATCHERS and m in NO_COMMAND
