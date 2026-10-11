"""craft.20 artifact check: your consumer-driven contract for gateway to engine (rung R6).

Run by `ol check craft.20` in your repo. Your artifacts live in primers/craft.20/:

  engineclient.go             the kata: the gateway's client of the engine API
  pacts/gateway-engine.json   your pact: every interaction the gateway relies on
  consumer_test.go            your consumer tests, driven by the pact's mock provider

The check verifies the pact against the provider's contract
(openapi/openai-subset.v1.yaml, engine tier), runs the course's contract suite
against your kata, runs your tests against your kata and the course's, and
then grades your tests by mutation: each of the planted faults in
course/mutants/craft.20 (a forwarded key, a client-chosen priority, a lost
usage chunk, a truncated stream taken for a success, ...) must make your tests
fail. Rung R6: at least 0.80 of them, and every semantic fault (tagged y).

Every Go run happens in a scratch module named craft20 (your kata, the
course's pact kit, and the tests), in its own process group with a timeout,
one at a time. The first run writes primers/craft.20/go.mod and
primers/craft.20/pact/pact.go when they are missing, so `go test` works in
your directory too. It never overwrites a file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import Ctx, Fail, run  # noqa: E402

DIR = "primers/craft.20"
UNIT = "primers/craft.20/engineclient.go"
PACT = f"{DIR}/pacts/gateway-engine.json"
THRESHOLD = 0.80
HERE = Path(__file__).resolve().parent


def course_tree() -> Path:
    return Path(os.environ.get("OL_COURSE_TREE") or HERE.parents[1])


# -- processes ------------------------------------------------------------------


def sh(argv: list[str], cwd: Path, timeout: float = 120) -> tuple[int, str]:
    """Run argv in its own process group; kill the group on timeout (exit 124)."""
    env = dict(
        os.environ, GOWORK="off", GOTOOLCHAIN="local", GOFLAGS="-count=1", GOPROXY="off"
    )
    p = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return 124, (out or "") + f"\n(timed out after {timeout:.0f} s)"


def tail(text: str, n: int = 25) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def reference_kata() -> str:
    sys.path.insert(0, str(course_tree() / "harness" / "src"))
    from olcourse import markers

    return markers.drop_markers((course_tree() / "ref" / UNIT).read_text())


def faulty(mid: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / UNIT
        dst.parent.mkdir(parents=True)
        dst.write_text(reference_kata())
        rc, out = sh(
            [
                "patch",
                "-s",
                "-p1",
                "-d",
                d,
                "-i",
                str(course_tree() / "mutants" / "craft.20" / f"{mid}.patch"),
            ],
            Path(d),
            30,
        )
        if rc != 0:
            raise Fail(f"fault {mid} does not apply: {out}")
        return dst.read_text()


def faults() -> list[tuple[str, bool, str]]:
    rows = []
    for line in (
        (course_tree() / "mutants" / "craft.20" / "manifest.tsv")
        .read_text()
        .splitlines()
    ):
        if line.strip() and not line.startswith("#"):
            c = line.split("\t")
            rows.append((c[0], c[5] == "y", c[6]))
    return rows


def module(c: Ctx, kata: str, tests: str) -> Path:
    """A scratch module craft20: kata, the course's pact kit, and either your
    tests and pact ("learner") or the course's contract suite ("course")."""
    d = Path(tempfile.mkdtemp(prefix="ol-craft20-"))
    c.cleanups.append(lambda: shutil.rmtree(d, ignore_errors=True))
    (d / "go.mod").write_text("module craft20\n\ngo 1.22\n")
    (d / "engineclient.go").write_text(kata)
    (d / "pact").mkdir()
    shutil.copy(HERE / "pact" / "pact.go", d / "pact" / "pact.go")
    if tests == "learner":
        for f in sorted(c.path(DIR).glob("*_test.go")):
            shutil.copy(f, d / f.name)
        shutil.copytree(c.path(f"{DIR}/pacts"), d / "pacts")
    else:
        for f in sorted(
            (course_tree() / "tests" / "go" / "craft_20").glob("*_test.go")
        ):
            shutil.copy(f, d / f.name)
    return d


def go_test(d: Path, tags: str = "") -> tuple[int, str]:
    argv = ["go", "test"] + (["-tags", tags] if tags else []) + ["./..."]
    return sh(argv, d, 120)


def learner_kata(c: Ctx) -> str:
    return c.require_file(UNIT).read_text()


# -- a small JSON Schema validator (the subset the OpenAPI contract uses) ----------


def check_schema(
    v, s: dict, root: dict, at: str = "$", partial: bool = False
) -> list[str]:
    if "$ref" in s:
        node = root
        for part in s["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return check_schema(v, node, root, at, partial)
    errs: list[str] = []
    t = s.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = any(
            (x == "null" and v is None)
            or (x == "boolean" and isinstance(v, bool))
            or (x == "integer" and isinstance(v, int) and not isinstance(v, bool))
            or (
                x == "number"
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            )
            or (x == "string" and isinstance(v, str))
            or (x == "array" and isinstance(v, list))
            or (x == "object" and isinstance(v, dict))
            for x in types
        )
        if not ok:
            return [f"{at}: {json.dumps(v)[:40]} is not {t}"]
    if "const" in s and v != s["const"]:
        errs.append(f"{at}: {v!r} != {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{at}: {v!r} not in {s['enum']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            errs.append(f"{at}: {v} < {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            errs.append(f"{at}: {v} > {s['maximum']}")
        if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]:
            errs.append(f"{at}: {v} <= {s['exclusiveMinimum']}")
    if isinstance(v, str):
        if len(v) < s.get("minLength", 0):
            errs.append(f"{at}: shorter than {s['minLength']}")
        if "pattern" in s and not re.search(s["pattern"], v):
            errs.append(f"{at}: {v!r} does not match {s['pattern']}")
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0):
            errs.append(f"{at}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            errs.append(f"{at}: more than {s['maxItems']} items")
        if "items" in s:
            for i, x in enumerate(v):
                errs += check_schema(x, s["items"], root, f"{at}[{i}]")
    if isinstance(v, dict):
        if not partial:
            for k in s.get("required", []):
                if k not in v:
                    errs.append(f"{at}: missing {k}")
        props = s.get("properties", {})
        for k, x in v.items():
            if k in props:
                errs += check_schema(x, props[k], root, f"{at}.{k}")
            elif isinstance(s.get("additionalProperties"), dict):
                errs += check_schema(x, s["additionalProperties"], root, f"{at}.{k}")
    for sub in s.get("allOf", []):
        errs += check_schema(v, sub, root, at, partial)
    if "oneOf" in s or "anyOf" in s:
        alts = s.get("oneOf") or s.get("anyOf")
        results = [check_schema(v, sub, root, at, partial) for sub in alts]
        good = sum(1 for r in results if not r)
        if good == 0 or ("oneOf" in s and good > 1 and not partial):
            errs.append(f"{at}: matches {good} of the {len(alts)} alternatives")
    return errs


def openapi(c: Ctx) -> dict:
    if "openapi" not in c.cache:
        for p in (
            c.path("contracts/openapi/openai-subset.v1.yaml"),
            course_tree() / "contracts/openapi/openai-subset.v1.yaml",
        ):
            if p.is_file():
                docs = c.yaml_docs(p.read_text(), str(p))
                c.cache["openapi"] = docs[0]
                break
        else:
            raise Fail(
                "contracts/openapi/openai-subset.v1.yaml is missing: run `ol contracts sync`"
            )
    return c.cache["openapi"]


def verify_interaction(spec: dict, it: dict) -> list[str]:
    """Provider verification against the contract: the pact may only expect
    what the engine tier of openai-subset.v1.yaml promises."""
    errs: list[str] = []
    name = it.get("description", "?")
    req, resp = it.get("request") or {}, it.get("response") or {}
    path = (req.get("path") or "").split("?")[0]
    op = ((spec.get("paths") or {}).get(path) or {}).get(
        (req.get("method") or "").lower()
    )
    if op is None:
        return [f"{name!r}: the engine has no {req.get('method')} {path}"]
    if req.get("body") is not None:
        rs = (
            ((op.get("requestBody") or {}).get("content") or {})
            .get("application/json", {})
            .get("schema")
        )
        if rs:
            errs += [
                f"{name!r} request: {e}"
                for e in check_schema(req["body"], rs, spec, "body", partial=True)
            ]
    status = str(resp.get("status"))
    r = (op.get("responses") or {}).get(status)
    if r is None:
        return errs + [f"{name!r}: {path} never answers {status}"]
    if "$ref" in r:
        node = spec
        for part in r["$ref"].lstrip("#/").split("/"):
            node = node[part]
        r = node
    content = r.get("content") or {}
    headers = {k.lower(): v for k, v in (resp.get("headers") or {}).items()}
    if status == "429" and not str(headers.get("retry-after", "")).isdigit():
        errs.append(f"{name!r}: a 429 carries Retry-After in seconds")
    if resp.get("events") is not None:
        sse = content.get("text/event-stream")
        if not sse:
            return errs + [f"{name!r}: {path} {status} does not stream"]
        chunk = sse.get("x-tl-chunk") or {}
        err_schema = {"$ref": "#/components/schemas/Error"}
        for i, ev in enumerate(resp["events"]):
            if not ev.endswith("\n\n"):
                errs.append(
                    f"{name!r} event {i}: an event ends with a blank line (\\n\\n)"
                )
            for line in ev.rstrip("\n").split("\n"):
                if line.startswith(":") or not line:
                    continue
                if not line.startswith("data:"):
                    errs.append(
                        f"{name!r} event {i}: {line[:30]!r} is not a data or comment line"
                    )
                    continue
                data = line[5:].lstrip(" ")
                if data == "[DONE]":
                    continue
                try:
                    v = json.loads(data)
                except json.JSONDecodeError:
                    errs.append(f"{name!r} event {i}: not JSON")
                    continue
                schema = err_schema if isinstance(v, dict) and "error" in v else chunk
                errs += [
                    f"{name!r} event {i}: {e}"
                    for e in check_schema(v, schema, spec, "data")
                ]
    else:
        js = (content.get("application/json") or {}).get("schema")
        if js is None:
            errs.append(f"{name!r}: {path} {status} has no JSON body")
        else:
            errs += [
                f"{name!r} response: {e}"
                for e in check_schema(resp.get("body"), js, spec, "body")
            ]
    return errs


def load_pact(c: Ctx) -> dict:
    try:
        return json.loads(c.require_file(PACT).read_text())
    except json.JSONDecodeError as e:
        raise Fail(f"{PACT} is not JSON: {e}") from None


# -- tests ----------------------------------------------------------------------

HAND = {
    "description": "a streamed chat with usage",
    "request": {
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": {"X-TL-Priority": "5", "Accept": "text/event-stream"},
        "absentHeaders": ["Authorization"],
        "body": {
            "model": "smol",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    },
    "response": {
        "status": 200,
        "headers": {"Content-Type": "text/event-stream"},
        "events": [
            'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            ": ping\n\n",
            'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}\n\n',
            'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":"length"}]}\n\n',
            'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"smol-135m@v3","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":30,"total_tokens":42}}\n\n',
            "data: [DONE]\n\n",
        ],
    },
}


def test_hand_example_interaction(c: Ctx) -> None:
    # WHY: the chapter's worked example (section 3): the interaction "a streamed
    #      chat with usage" passes provider verification against the engine
    #      tier, and its usage chunk is the only one with choices: [] and
    #      12 + 30 = 42 tokens; with stream_options removed from the request
    #      side it still verifies (the pact may expect less, never more).
    # KIND: unit
    # CHAPTER: craft.20 section 3, Worked example by hand
    spec = openapi(c)
    errs = verify_interaction(spec, HAND)
    if errs:
        raise Fail(
            "the worked example fails provider verification:\n" + "\n".join(errs)
        )
    usage = [json.loads(e[6:]) for e in HAND["response"]["events"] if '"usage"' in e]
    if (
        len(usage) != 1
        or usage[0]["choices"] != []
        or usage[0]["usage"]["total_tokens"] != 12 + 30
    ):
        raise Fail(f"the usage chunk of the worked example is {usage}")
    bad = json.loads(json.dumps(HAND))
    bad["response"]["events"][4] = bad["response"]["events"][4].replace(
        '"total_tokens":42', '"total_tokens":"42"'
    )
    if not verify_interaction(spec, bad):
        raise Fail("provider verification accepted total_tokens as a string")


def test_files_present(c: Ctx) -> None:
    # WHY: the check and the chapter agree on the layout; the first run writes
    #      go.mod and the course's pact kit so `go test` works in primers/craft.20.
    # KIND: unit
    # CHAPTER: craft.20 section 4, The artifact and its check
    d = c.path(DIR)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "go.mod").exists():
        (d / "go.mod").write_text("module craft20\n\ngo 1.22\n")
        print("       wrote primers/craft.20/go.mod")
    if not (d / "pact" / "pact.go").exists():
        (d / "pact").mkdir(exist_ok=True)
        shutil.copy(HERE / "pact" / "pact.go", d / "pact" / "pact.go")
        print(
            "       wrote primers/craft.20/pact/pact.go (the course's kit; the check uses its own copy)"
        )
    missing = [
        p for p in (UNIT, PACT, f"{DIR}/consumer_test.go") if not c.path(p).is_file()
    ]
    if missing:
        raise Fail(
            f"missing: {', '.join(missing)} (run `ol start craft.20` for the kata; section 4 for the rest)"
        )


def test_pact_is_well_formed(c: Ctx) -> None:
    # WHY: a pact names its consumer and provider and lists interactions with
    #      unique descriptions (tests find them by description), each with a
    #      request (method, path) and a response (status, and a body or events).
    # KIND: unit
    # CHAPTER: craft.20 section 2.1
    p = load_pact(c)
    errs = []
    if p.get("consumer") != "gateway" or p.get("provider") != "engine":
        errs.append(
            f"consumer/provider are {p.get('consumer')!r}/{p.get('provider')!r}; want gateway/engine"
        )
    its = p.get("interactions") or []
    if len(its) < 6:
        errs.append(
            f"{len(its)} interactions; the gateway relies on at least the six of section 4"
        )
    names = [i.get("description") for i in its]
    if len(set(names)) != len(names) or not all(names):
        errs.append("every interaction needs a unique, non-empty description")
    for it in its:
        rq, rs = it.get("request") or {}, it.get("response") or {}
        if (
            not rq.get("method")
            or not rq.get("path")
            or not isinstance(rs.get("status"), int)
        ):
            errs.append(
                f"{it.get('description')!r}: needs request.method, request.path, response.status"
            )
        if rs.get("body") is None and rs.get("events") is None:
            errs.append(
                f"{it.get('description')!r}: the response needs a body or events"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_pact_covers_what_the_gateway_relies_on(c: Ctx) -> None:
    # WHY: consumer-driven means the pact lists what the CONSUMER needs, and
    #      the gateway needs six behaviours: a completion with usage, a stream
    #      whose usage chunk it asked for, a stream that fails mid-way, a 429
    #      with Retry-After, a token count, and a request side that pins the
    #      headers it must (X-TL-Priority) and must not (Authorization) send.
    # KIND: unit
    # CHAPTER: craft.20 section 4, The artifact and its check
    its = load_pact(c).get("interactions") or []

    def events(it):
        return (it.get("response") or {}).get("events") or []

    have = {
        "a non-streamed completion with usage": any(
            (
                it["response"].get("status") == 200
                and isinstance(it["response"].get("body"), dict)
                and "usage" in it["response"]["body"]
                and "choices" in it["response"]["body"]
            )
            for it in its
        ),
        "a stream with a usage chunk and [DONE]": any(
            any('"choices":[]' in e.replace(" ", "") for e in events(it))
            and any("[DONE]" in e for e in events(it))
            and (
                ((it.get("request") or {}).get("body") or {}).get("stream_options")
                or {}
            ).get("include_usage")
            is True
            for it in its
        ),
        "a stream with an error event": any(
            any('"error"' in e for e in events(it)) for it in its
        ),
        "a 429 with Retry-After": any(
            it["response"].get("status") == 429 for it in its
        ),
        "a /v1/tokenize call": any(
            (it.get("request") or {}).get("path") == "/v1/tokenize" for it in its
        ),
        "a request that pins X-TL-Priority and an absent Authorization": any(
            "x-tl-priority"
            in {k.lower() for k in ((it.get("request") or {}).get("headers") or {})}
            and "authorization"
            in {
                k.lower()
                for k in ((it.get("request") or {}).get("absentHeaders") or [])
            }
            for it in its
        ),
    }
    missing = [k for k, ok in have.items() if not ok]
    if missing:
        raise Fail("your pact has no interaction for: " + "; ".join(missing))


def test_pact_matches_the_provider_contract(c: Ctx) -> None:
    # WHY: provider verification. A pact that expects what the engine never
    #      promised (a status the path does not answer, a chunk without
    #      choices, total_tokens as a string) passes against your mock and
    #      fails against every real engine; each interaction must fit the
    #      engine tier of openai-subset.v1.yaml (request bodies may be partial).
    # KIND: conformance
    # CHAPTER: craft.20 section 2.4
    spec = openapi(c)
    errs = []
    for it in load_pact(c).get("interactions") or []:
        errs += verify_interaction(spec, it)
    if errs:
        raise Fail("\n".join(errs[:20]))


def test_your_kata_passes_the_course_contract_suite(c: Ctx) -> None:
    # WHY: your client must itself be right before your tests are graded on
    #      the course's: the course's own suite (course/tests/go/craft_20) runs
    #      against your engineclient.go.
    # KIND: conformance
    # CHAPTER: craft.20 section 4, What the tests check
    rc, out = go_test(module(c, learner_kata(c), "course"), "primer,coursepact")
    if rc != 0:
        raise Fail("the course's contract suite fails on your kata:\n" + tail(out))


def test_your_tests_pass_on_your_kata(c: Ctx) -> None:
    # WHY: your consumer tests and your pact agree with your own client.
    # KIND: unit
    # CHAPTER: craft.20 section 4, What the tests check
    rc, out = go_test(module(c, learner_kata(c), "learner"))
    if rc != 0:
        raise Fail("your consumer tests fail on your kata:\n" + tail(out))


def test_your_tests_pass_on_the_course_kata(c: Ctx) -> None:
    # WHY: baseline A of mutation grading (DESIGN 5.6): a test that rejects a
    #      correct client grades nothing. Your tests must pass on the course's
    #      kata before any planted fault counts.
    # KIND: unit
    # CHAPTER: craft.20 section 4, What the tests check
    rc, out = go_test(module(c, reference_kata(), "learner"))
    if rc != 0:
        raise Fail("your tests reject the course's correct client:\n" + tail(out))
    c.cache["baseline"] = True


def test_your_tests_catch_the_planted_faults(c: Ctx) -> None:
    # WHY: rung R6 grades your contract tests by what they catch: each planted
    #      fault runs alone, and your tests must fail on at least 0.80 of them
    #      and on every semantic one (the chapter's pitfalls).
    # KIND: fault
    # CHAPTER: craft.20 section 5, Pitfalls
    if not c.cache.get("baseline"):
        raise Fail("graded only after your tests pass on the course's kata")
    rows = faults()
    killed, survived, required_missed = 0, [], []
    for mid, required, public in rows:
        rc, _ = go_test(module(c, faulty(mid), "learner"))
        if rc != 0:
            killed += 1
        else:
            survived.append(f"{mid} ({'a planted pitfall' if required else public})")
            if required:
                required_missed.append(mid)
    score = killed / len(rows)
    print(f"       mutation score {killed}/{len(rows)} = {score:.2f}")
    if score < THRESHOLD or required_missed:
        raise Fail(
            f"score {score:.2f} (threshold {THRESHOLD}); survivors: {', '.join(survived)}"
        )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
