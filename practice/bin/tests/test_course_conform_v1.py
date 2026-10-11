"""ol conform v1 and v2 (DESIGN 5.8): every engine and gateway case against
the fixture system, the recording upstream for priority.internal, pending
cases (requires, --base), the v2 contract probes side by side, and failures
for a broken stop rule and a leaked priority header."""

KEY = {"TL_API_KEY": "tl_k1_secret", "TL_API_KEY_NOSCOPE": "tl_k2_noscope"}


def _setup(ss):
    ss.add_extras("conform-v1")
    ss.init()
    ss.install_system()


def _cases(out: str) -> dict:
    res = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("pass", "fail", "pending"):
            res[parts[1]] = parts[0]
    return res


def test_v1_engine_tier(ss):
    _setup(ss)
    out = ss(
        "conform", "openapi:v1", "--target", "engine", rc=0, env=KEY, timeout=300
    ).out
    got = _cases(out)
    for case in (
        "v1.healthz",
        "schema.models",
        "chat.schema",
        "chat.nonstream.greedy",
        "chat.stream.framing",
        "chat.stream.equals_nonstream",
        "chat.finish.length",
        "chat.stop",
        "chat.usage",
        "chat.seed",
        "err.400",
        "err.422",
        "cancel.disconnect",
        "concurrency.16",
        "client.openai_sdk",
    ):
        assert got.get(case) == "pass", (case, out)
    assert got["tools.call"] == "pending" and "requires L10.9" in out
    v = ss.verdicts("conform:openapi:v1:engine")[-1]
    assert v["result"] == "pass" and v["cases"]["chat.stop"] == "pass"


def test_v1_gateway_tier_with_the_recording_upstream(ss):
    _setup(ss)
    out = ss(
        "conform", "openapi:v1", "--target", "gateway", rc=0, env=KEY, timeout=300
    ).out
    got = _cases(out)
    for case in (
        "auth.401",
        "auth.403",
        "ratelimit.429",
        "route.model",
        "cache.hit",
        "schema.models",
        "priority.internal",
        "client.openai_sdk",
    ):
        assert got.get(case) == "pass", (case, out)
    assert "gateway against the recording upstream" in out
    assert got["policy.451"] == "pending"
    out = ss(
        "conform",
        "openapi:v1",
        "--target",
        "gateway",
        rc=None,
        env={"TL_API_KEY": "tl_k1_secret"},
        timeout=300,
    ).out
    assert "auth.403" in out and "set $TL_API_KEY_NOSCOPE" in out


def test_v2_side_by_side_and_failures(ss):
    _setup(ss)
    out = ss(
        "conform", "openapi:v2", "--target", "engine", rc=0, env=KEY, timeout=300
    ).out
    assert _cases(out).get("v2.side_by_side") == "pass"
    eng = ss.learner / "serve/engine.py"
    eng.write_text(
        eng.read_text().replace(
            "hit = min((text.find(s) for s in stops if s in text), default=-1)",
            "hit = -1",
        )
    )
    out = ss("conform", "openapi:v1:engine", rc=1, env=KEY, timeout=300).out
    assert _cases(out).get("chat.stop") == "fail" and "the stop text excluded" in out
    gw = ss.learner / "serve/gateway.py"
    gw.write_text(
        gw.read_text().replace(
            '"X-TL-Priority": tier}',
            '"X-TL-Priority": self.headers.get("X-TL-Priority") or tier}',
        )
    )
    out = ss("conform", "openapi:v1:gateway", rc=1, env=KEY, timeout=300).out
    assert _cases(out).get("priority.internal") == "fail" and "must be stripped" in out
