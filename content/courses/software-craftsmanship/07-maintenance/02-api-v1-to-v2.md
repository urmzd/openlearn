# Interface migration: API v1 to v2

## Overview

| | |
|---|---|
| **Module** | `craft.14` · build · rust, go · Pass 11 · 1 h reading |
| **Solution** | Engine: [`api_version.rs`](../../../../course/solutions/rust/crates/tl-serve/src/api_version.rs) (`ApiVersion`, `VersionError`, `select`, `response_headers`, `error_code`, `cached_tokens`) and the serve loop [`server.rs`](../../../../course/solutions/rust/crates/tl-serve/src/server.rs), taken over from `L10.9` (`api_config`, `versioned`, `usage`, `version_headers`, `version_error`): every API request selects its version before dispatch, every response echoes it, v2 answers carry `cached_tokens` and the v2 error codes, and a configured window (`[ext.api]` in runtime.toml) adds `Deprecation` and `Sunset`. Gateway: [`apiversion.go`](../../../../course/solutions/go/gateway/server/apiversion.go) (`VersionedAPI`, `APIVersionConfig`), wired in [`compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go) |
| **Contract** | [`course/contracts/openapi/openai-subset.v2.yaml`](../../../../course/contracts/openapi/openai-subset.v2.yaml) ("Versions side by side", the `X-TL-API-Version`, `Deprecation`, and `Sunset` headers, and the error code enum) |
| **Needs** | [`craft.12`](../06-releases-and-deprecation/02-deprecation-policy.md) · [`L10.9`](../../../courses/build-an-llm/p10-serving/09-tool-calls-and-constrained-decoding.md) the serve loop taken over · [`L10.6`](../../../courses/build-an-llm/p10-serving/06-disaggregated-prefill-decode.md) registry and engine control |
| **Used by** | [`ops.05`](../../systems/05-incident-response-and-chaos/05-api-v2-migration.md), the API v2 migration drill |

## Key Takeaways

- The request header chooses v1 or v2; absent a header, a configured default applies.
- During the overlap, v1 replies include Deprecation and Sunset and every reply echoes its selected version.
- After sunset v1 returns 410, and per-version ledger grouping shows remaining clients.

## How to read this chapter

Sections 1 to 3 explain why two API versions must run side by side and trace one request through version selection. Section 4 lists the cases a correct implementation must handle, in the engine and in the gateway. Section 5 walks through the solution: the pure selection function, the headers and error mapping, where the serve loop calls them, and the gateway middleware. Try to predict each step of the solution before reading it.

---

## 1. Why now

API v2 changes client-visible behavior while v1 clients remain in service. The gateway and engine must route versions deliberately and meter them separately so operators can see migration progress.

## 2. Principles

Keep v1 and v2 contracts explicit. Validate requests against the selected version, preserve error semantics, and never infer a client version from payload fields that can be ambiguous.

## 3. Worked example

The worked request sends `X-TL-API-Version: 2` while the configured default remains `1`; the reply echoes `X-TL-API-Version: 2`, carries `usage.prompt_tokens_details.cached_tokens`, and has no `Deprecation` header (it is not on a deprecated path). The same request without the header gets the default, v1: it echoes `1`, has no `cached_tokens`, and, once a window is configured, carries `Deprecation: @<deprecated_at>` and `Sunset: <HTTP date>`. A request with `X-TL-API-Version: 3` gets 400 `unsupported_api_version` and never reaches a handler. After the sunset instant, the header-less request gets 410 `api_version_sunset`.

In operation: introduce v2 routes while v1 remains default for existing clients, send a canary cohort to v2 with the header, compare request errors, streaming behavior, and usage totals, then expand. Retain a switch back to v1 (the default) until all consumers migrate.

## 4. The interface and cases

The engine's selection core is four pure functions:

```rust
pub fn select(header: Option<&str>, default: ApiVersion, now_unix: i64, sunset_unix: i64) -> Result<ApiVersion, VersionError>;
pub fn response_headers(v: ApiVersion, deprecated_unix: i64, sunset_http_date: &str) -> Vec<(&'static str, String)>;
pub fn error_code(v: ApiVersion, code: &str) -> &str;
pub fn cached_tokens(v: ApiVersion, reported: Option<u32>) -> Option<u32>;
```

The gateway's is one middleware, `VersionedAPI(cfg APIVersionConfig, next http.Handler) http.Handler`. Configuration in runtime.toml:

```toml
[ext.api]
default_version = "1"      # or "2"
deprecated_at = 1767225600 # Unix seconds; set together with sunset or not at all
sunset = 1775001600        # must come after deprecated_at
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example: explicit v2 is echoed (gateway) | unit | The worked explicit v2 request gets a response marked v2. | Confirms the header-selected contract end to end. |
| an explicit version overrides the default | unit | A request header overrides the configured default. | Allows controlled migration cohorts. |
| an unknown version is rejected before dispatch | boundary | Unknown versions fail before backend work starts. | Avoids ambiguous execution under an unsupported contract. |
| the version is echoed and v1 is marked deprecated | unit | Responses identify the selected version and mark v1 deprecated. | Clients and operators can observe migration state. |
| v1 carries sunset headers during the window (gateway) | unit | v1 replies carry the deprecation window headers. | Gives consumers a concrete removal date. |
| the meter sees the API version (gateway) | integration | The middleware passes the selected API version to the downstream meter grouping map. | Operators can separate v1 and v2 usage during the migration. |
| v1 sunset is a hard boundary | boundary | v1 is rejected after sunset. | Prevents indefinite compatibility promises. |
| sunset and unsupported errors keep the public error shape (gateway) | boundary | Sunset and unsupported-version errors retain the public error shape. | Existing clients can handle failure consistently. |
| v2 error names are specific, v1 codes are stable | unit | v2 errors are specific while v1 codes remain stable. | Preserves compatibility through the overlap period. |
| cached-token usage only in v2 | unit | Exposes cached-token usage only in v2 responses. | Preserves the versioned response contract. |
| the engine serves v1 and v2 side by side | conformance | The engine echoes the selected version, marks v1 and v2 `/v1/completions` deprecated during the window, adds `cached_tokens` to v2 usage, refuses an unknown version, and answers a sunset v1 with 410. | The OpenAPI v2 conformance run and the ops.05 drill run against this serve loop. |
| the engine still registers and obeys a registry drain | conformance | A real registry receives the running engine's identity, reachable ports and KV capacity, then drains it. | Migration must preserve worker discovery and administrative shutdown. |

The upgraded server preserves L10.6 registration: a nonempty `engine.gateway_registry` starts heartbeats every two seconds. It advertises the actual bound ports using `engine.advertise_host` (default `127.0.0.1`), including EngineControl for unified engines. It reads load for each heartbeat; a registry outage does not stop serving. A drain acknowledgement enters the same bounded shutdown path as SIGTERM.

## 5. The solution, step by step

### 5.1 From first principles

1. Two contracts share the same paths, so something outside the payload must say which one applies. A payload field cannot: the same JSON is valid under both. So the version is a request header, `X-TL-API-Version`, and its absence means a server-chosen default.
2. Operators move the default, not clients. Keeping the default in configuration lets one deployment flip from "v1 unless asked" to "v2 unless asked" and back, which is the rollback switch.
3. Any value other than `"1"` or `"2"` is a client bug or a future version this server does not know. Serving it as either version would be a guess, so it is a 400 before any handler runs.
4. A deprecation is a promise with a date. While the window is open, every v1 answer carries `Deprecation` (RFC 9745, `@<unix seconds>`) and `Sunset` (RFC 8594, an HTTP date), so clients and their logs see the deadline. After the sunset instant, v1 is refused with 410 Gone, which tells the client the resource existed and was withdrawn.
5. The client must be able to see which contract it got, and the meter must be able to count it. So every response echoes the selected version, and the gateway rewrites the request header to the selected value before metering.
6. Version-specific response differences (the `cached_tokens` field, new error codes) are applied at the edges, in small functions of `(version, value)`, so the handlers stay version-agnostic and v1 answers are byte-for-byte what they were.

### 5.2 The code, annotated

**Selection.** One pure function decides everything the request needs before dispatch.

```rust
pub fn select(header: Option<&str>, default: ApiVersion, now_unix: i64, sunset_unix: i64)
    -> Result<ApiVersion, VersionError>
{
    let v = match header {
        None => default,                                 // step 2: the operator's default
        Some("1") => ApiVersion::V1,
        Some("2") => ApiVersion::V2,
        Some(_) => return Err(VersionError::Unsupported), // step 3: never guess
    };
    // step 4: sunset applies to v1 however it was chosen, explicitly or by default;
    // sunset_unix == 0 means no window is configured
    if v == ApiVersion::V1 && sunset_unix > 0 && now_unix >= sunset_unix {
        return Err(VersionError::Sunset)
    }
    Ok(v)
}
```

Taking `now_unix` as a parameter rather than reading the clock keeps the function testable at exact boundary instants (`now == sunset` is already sunset).

**Headers, error codes, and usage.**

```rust
pub fn response_headers(v: ApiVersion, deprecated_unix: i64, sunset_http_date: &str)
    -> Vec<(&'static str, String)>
{
    let mut h = vec![("X-TL-API-Version", v.number().to_string())];   // always echoed
    if v == ApiVersion::V1 {
        h.push(("Deprecation", format!("@{deprecated_unix}")));
        h.push(("Sunset", sunset_http_date.to_string()));
    }
    h
}

pub fn error_code(v: ApiVersion, code: &str) -> &str {
    if v == ApiVersion::V1 { return code }               // v1 codes never change
    match code {
        "rate_limit_exceeded" => "requests_limit_exceeded",
        "no_capacity" => "model_draining",
        other => other,
    }
}

pub fn cached_tokens(v: ApiVersion, reported: Option<u32>) -> Option<u32> {
    // v2 requires the field; an engine that cannot measure it reports 0, not absent
    if v == ApiVersion::V2 { Some(reported.unwrap_or(0)) } else { None }
}
```

**Where the serve loop uses them.** The handler reads the header and selects first; an error never reaches a route.

```rust
    // the version is chosen before dispatch: an unknown or sunset one never
    // reaches a handler
    let header = parts.headers.get("x-tl-api-version").map(|h| h.to_str().unwrap_or(""));
    let v = match api_version::select(header, shared.api.default, unix_now() as i64, shared.api.sunset_unix) {
        Ok(v) => v,
        Err(e) => {
            let mut resp = version_error(&shared.api, e, &path);
            // ... request id, a SERVER span, and the HTTP metric are still recorded
            return Ok(resp);
        }
    };
    // ... dispatch to chat, completions, embeddings with `v`
    for (k, val) in version_headers(&shared.api, v, &path) {   // every answer, success or error
        if let Ok(h) = HeaderValue::from_str(&val) {
            resp.headers_mut().insert(k, h);
        }
    }
```

A non-UTF-8 header becomes `""`, which `select` treats as an unknown version, so a garbled header is a 400, not a silent default.

`version_headers` only adds the window headers when a window is configured, and also marks v2's `/v1/completions`, which the v2 contract deprecates as a path:

```rust
fn version_headers(api: &ApiConfig, v: ApiVersion, path: &str) -> Vec<(&'static str, String)> {
    if api.deprecated_unix <= 0 || api.sunset_unix <= 0 {
        return vec![("X-TL-API-Version", v.number().to_string())];   // no window: echo only
    }
    let sunset = http_date(api.sunset_unix as u64);     // IMF-fixdate, e.g. "Wed, 01 Apr 2026 00:00:00 GMT"
    let mut h = api_version::response_headers(v, api.deprecated_unix, &sunset);
    if v == ApiVersion::V2 && path == "/v1/completions" {
        h.push(("Deprecation", format!("@{}", api.deprecated_unix)));
        h.push(("Sunset", sunset));
    }
    h
}

fn version_error(api: &ApiConfig, e: VersionError, path: &str) -> Response<RespBody> {
    let (status, code, headers) = match e {
        VersionError::Unsupported => (400, "unsupported_api_version", Vec::new()),
        // a sunset reply is still a v1 reply: it carries the deprecation headers
        VersionError::Sunset => (410, "api_version_sunset", version_headers(api, ApiVersion::V1, path)),
    };
    let err = ApiError { status, kind: "invalid_request_error",
                         message: "API version is unavailable".to_string(), param: None, code: Some(code) };
    // ... error_response(&err) plus the headers
}
```

The engine maps its own errors with `versioned`. Its only 429 is its admission queue, so in v2 it is the specific `queue_full`, and a 503 becomes `model_draining` only when the engine really is draining:

```rust
fn versioned(shared: &Shared, v: ApiVersion, e: &ApiError) -> ApiError {
    let mut e = e.clone();
    if v == ApiVersion::V2 {
        e.code = match e.code {
            Some("rate_limit_exceeded") => Some("queue_full"),
            Some("no_capacity") if shared.draining.load(Ordering::SeqCst) =>
                Some(api_version::error_code(v, "no_capacity")),   // "model_draining"
            c => c,                                       // other 503s stay no_capacity
        };
    }
    e
}

fn usage(v: ApiVersion, prompt_tokens: usize, completion_tokens: usize) -> Value {
    let mut u = openai::usage(prompt_tokens, completion_tokens);
    // the serve loop does not see per-request prefix-cache hits, so v2 reports 0
    if let Some(n) = api_version::cached_tokens(v, None) {
        u["prompt_tokens_details"] = serde_json::json!({ "cached_tokens": n });
    }
    u
}
```

Configuration is validated at startup by `api_config`: unknown keys under `[ext.api]` are errors, `default_version` goes through `select` itself (so the config accepts exactly what a header accepts), `deprecated_at` and `sunset` must be set together, and `sunset` must come after `deprecated_at`. A half-configured window fails the start rather than producing headers with a zero date.

**The gateway middleware.**

```go
func VersionedAPI(cfg APIVersionConfig, next http.Handler) http.Handler {
    if cfg.Default == "" { cfg.Default = "1" }
    if cfg.Now == nil { cfg.Now = time.Now }             // injectable clock for tests
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        v := r.Header.Get(APIVersionHeader)
        if v == "" { v = cfg.Default }
        if v != "1" && v != "2" { writeVersionError(w, http.StatusBadRequest, "unsupported_api_version"); return }
        w.Header().Set(APIVersionHeader, v)
        if v == "1" {
            if !cfg.DeprecatedAt.IsZero() {
                w.Header().Set("Deprecation", "@"+strconv.FormatInt(cfg.DeprecatedAt.Unix(), 10))
            }
            if !cfg.Sunset.IsZero() {
                w.Header().Set("Sunset", cfg.Sunset.UTC().Format(http.TimeFormat))
                if !cfg.Now().Before(cfg.Sunset) {       // now >= sunset
                    writeVersionError(w, http.StatusGone, "api_version_sunset"); return
                }
            }
        }
        r.Header.Set(APIVersionHeader, v)                // the usage meter records this selected version
        next.ServeHTTP(w, r)
    })
}
```

`writeVersionError` writes the same `{"error": {message, type, param, code}}` shape as every other gateway error, with `type` `invalid_request_error`, so clients need no special case. In [`compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go) the middleware is placed after authentication but before rate limits, the response cache, and metering: an unauthenticated caller learns nothing about versions, a rejected version costs no quota, cached v1 and v2 answers are not mixed up, and the meter sees the resolved version (step 5) so the ledger's `api_version` column groups traffic.

Full solution: [`api_version.rs`](../../../../course/solutions/rust/crates/tl-serve/src/api_version.rs), [`server.rs`](../../../../course/solutions/rust/crates/tl-serve/src/server.rs), [`apiversion.go`](../../../../course/solutions/go/gateway/server/apiversion.go).

### 5.3 Why this solution

- **Cost** is a header lookup and a comparison per request.
- **Header, not path** (`/v2/chat/completions`): path versioning would duplicate every route and every client's base URL, and the contract keeps the OpenAI-compatible `/v1/...` paths that existing SDKs hard-code. A header lets one deployment serve both on the same routes.
- **Header, not content sniffing**: inferring v2 from the presence of a v2-only field makes the same v1-valid payload ambiguous, and a v1 client would silently switch contract the day it added an optional field.
- **Both layers select.** The gateway selects for metering, caching, and early refusal; the engine selects again because it is also reachable directly and must honour the same contract (the conformance run targets the engine alone).
- **Tradeoff accepted:** v2 reports `cached_tokens: 0` from the serve loop, which is honest about what this layer can see but under-reports prefix-cache hits until the engine reports them per request.

### 5.4 Explain it back

**Why does `select` apply the sunset check after resolving the default, rather than only to explicit `"1"` headers?**

<details><summary>Answer</summary>

Most v1 traffic sends no header at all. If the default were still v1 at sunset and only explicit headers were refused, the withdrawn contract would keep serving everyone who never opted in. Checking the resolved version makes the sunset real; flipping the default to `"2"` before the sunset moves header-less clients over cleanly.

</details>

**Why 410 after sunset, and 400 for an unknown version?**

<details><summary>Answer</summary>

410 Gone says "this existed and was deliberately withdrawn", which is the truth for v1 and tells the client to migrate rather than retry. 400 says the request is malformed for this server: version 3 was never served, so there is nothing to have withdrawn.

</details>

**Why does the gateway overwrite `r.Header` with the selected version before calling `next`?**

<details><summary>Answer</summary>

A header-less request was served as the default version, but its header is empty. The meter downstream reads the header; without the rewrite, all default traffic would be grouped as "no version" and operators could not tell how many v1 clients remain.

</details>

**Why is the engine's v2 code for its 429 `queue_full` and not `requests_limit_exceeded`?**

<details><summary>Answer</summary>

v2 names errors by cause. The engine has no per-key rate limits; its only 429 is its own admission queue being full. `requests_limit_exceeded` is the gateway's per-key budget, which `error_code` maps for the gateway's errors. Giving each its own name lets clients decide whether waiting or slowing down helps.

</details>

**What would break if v1 responses gained `cached_tokens` too?**

<details><summary>Answer</summary>

Probably nothing for lenient clients, but v1 is a frozen contract: strict clients and recorded golden responses would see a changed body, and the migration would no longer be "v1 answers are exactly what they were". New fields belong to the new version.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Inferring a version from request contents when the header is absent | The same payload can dispatch under different contracts. | `select` reads only the header and the configured default, and rejects any other header value with `Unsupported` before dispatch. |
| 2 | Returning v1 success after its sunset | Clients continue depending on a contract the service has withdrawn. | `select` (engine) and `VersionedAPI` (gateway) return 410 `api_version_sunset` once `now >= sunset`, for explicit and default v1 alike. |
| 3 | Combining usage across v1 and v2 | Operators cannot tell whether old clients remain. | `VersionedAPI` sets the request header to the resolved version before the meter runs. |
| 4 | A half-configured window | `Deprecation: @0` or a sunset before deprecation reaches clients. | `api_config` refuses to start unless both instants are set together and sunset is later. |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | [`craft.12`](../06-releases-and-deprecation/02-deprecation-policy.md) | The deprecation policy that sets the window and the sunset date. |
| Back | [`craft.02`](../../diagramming/02-documentation-writing/01-architecture-decision-records.md) | An ADR records why versions are selected by header. |
| Back | [`course/contracts/openapi/openai-subset.v2.yaml`](../../../../course/contracts/openapi/openai-subset.v2.yaml) | The v2 contract: headers, `cached_tokens`, and the error code enum. |
| Back | [`gw.07`](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md) | The usage ledger whose `api_version` column the meter fills. |
| Back | [`L10.6`](../../../courses/build-an-llm/p10-serving/06-disaggregated-prefill-decode.md) | Registry heartbeats and EngineControl stay live while the HTTP interface changes. |
| Back | [`L10.9`](../../../courses/build-an-llm/p10-serving/09-tool-calls-and-constrained-decoding.md) | The serve loop taken over: version selection runs before its routes, and its usage and error documents take the selected version. |
| Back | [`L10.5`](../../../courses/build-an-llm/p10-serving/05-openai-server-on-tokio.md) | The OpenAI-compatible server the routes come from. |
| Forward | [`ops.05`](../../systems/05-incident-response-and-chaos/05-api-v2-migration.md) | The migration drill run against this serve loop and gateway. |

## Going further

| Resource | What to inspect |
|---|---|
| RFC 9745 (Deprecation header) and RFC 8594 (Sunset header) | The exact syntax `version_headers` and `VersionedAPI` emit, and how clients are expected to react. |
| [`openai-subset.v2.yaml`](../../../../course/contracts/openapi/openai-subset.v2.yaml), "Versions side by side" | The rules the selection function encodes, stated as contract. |
