//! Why these cases exist: API version selection must be deterministic across
//! the engine and gateway while v1 and v2 clients are served side by side.
mod api_version { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../crates/tl-serve/src/api_version.rs")); }
use api_version::{cached_tokens, error_code, response_headers, select, ApiVersion, VersionError};

#[test]
// WHY: An explicit request must select its named contract over the deployment default.
// KIND: unit
fn hand_example_explicit_version_overrides_default() {
    // A request must select the contract named by its explicit version header.
    assert_eq!(select(Some("2"), ApiVersion::V1, 100, 200), Ok(ApiVersion::V2));
}
#[test]
// WHY: Unknown versions must fail closed before handler dispatch.
// KIND: boundary
fn rejects_unknown_version_before_dispatch() {
    // Unknown versions must fail closed rather than accidentally use a schema.
    assert_eq!(select(Some("3"), ApiVersion::V1, 100, 200), Err(VersionError::Unsupported));
}
#[test]
// WHY: A sunset v1 request must not reach a v2-only handler.
// KIND: boundary
fn v1_sunset_is_a_hard_boundary() {
    // Once sunset passes, v1 must not reach a handler that expects v2 fields.
    assert_eq!(select(Some("1"), ApiVersion::V2, 200, 200), Err(VersionError::Sunset));
}
#[test]
// WHY: Clients need the selected version and v1 migration deadline on responses.
// KIND: unit
fn response_version_is_echoed_and_v1_is_marked_deprecated() {
    // Clients need an observable version and migration deadline on every v1 response.
    let h=response_headers(ApiVersion::V1, 120, "Thu, 01 Jan 1970 00:03:20 GMT");
    assert!(h.contains(&("X-TL-API-Version", "1".to_string())));
    assert!(h.contains(&("Deprecation", "@120".to_string())));
    assert!(h.iter().any(|x| x.0=="Sunset"));
}
#[test]
// WHY: Versioned capacity errors must be actionable without changing v1 codes.
// KIND: unit
fn v2_error_names_are_specific_and_v1_codes_are_stable() {
    // Versioned errors preserve v1 compatibility while making v2 capacity states actionable.
    assert_eq!(error_code(ApiVersion::V1,"rate_limit_exceeded"),"rate_limit_exceeded");
    assert_eq!(error_code(ApiVersion::V2,"rate_limit_exceeded"),"requests_limit_exceeded");
    assert_eq!(error_code(ApiVersion::V2,"no_capacity"),"model_draining");
}

#[test]
// WHY: V2 requires cache usage while v1 remains schema-compatible.
// KIND: unit
fn cached_token_usage_is_present_only_in_v2() {
    // V2 makes the field mandatory; v1 remains schema-compatible and omits it.
    assert_eq!(cached_tokens(ApiVersion::V1,Some(3)),None);
    assert_eq!(cached_tokens(ApiVersion::V2,None),Some(0));
}

#[test]
// WHY: The unit only matters once the engine serves both versions on the same
//      paths: the header picks the contract and the response echoes it, v1
//      carries its deadline during the window, v2 usage carries
//      cached_tokens, v2's deprecated /v1/completions is marked, an unknown
//      version is refused, and a sunset v1 never reaches the model.
// KIND: conformance
// CHAPTER: craft.14 section 4
fn engine_serves_v1_and_v2_side_by_side() {
    use tl_serve::api_version::ApiVersion as Served;
    let chat = r#"{"model":"succ","messages":[{"role":"user","content":"ab"}],"max_tokens":2,"temperature":0}"#;
    let completion = r#"{"model":"succ","prompt":"ab","max_tokens":2,"temperature":0}"#;
    let cached = r#""prompt_tokens_details":{"cached_tokens":0}"#;
    let far = 4_102_444_800; // 2100-01-01
    let s = serve::start(Served::V1, 90, far);

    let (st, head, body) = serve::post(s.http(), "/v1/chat/completions", None, chat);
    assert_eq!(st, 200, "{body}");
    assert!(head.contains("x-tl-api-version: 1"), "no header: the default (v1) is echoed\n{head}");
    assert!(head.contains("deprecation: @90"), "v1 during the window is deprecated\n{head}");
    assert!(head.contains("sunset: fri, 01 jan 2100 00:00:00 gmt"), "v1 carries its sunset as an HTTP-date\n{head}");
    assert!(!body.contains("prompt_tokens_details"), "v1 usage keeps its v1 shape: {body}");

    let (st, head, body) = serve::post(s.http(), "/v1/chat/completions", Some("2"), chat);
    assert_eq!(st, 200, "{body}");
    assert!(head.contains("x-tl-api-version: 2"), "{head}");
    assert!(!head.contains("deprecation:"), "v2 chat is not deprecated\n{head}");
    assert!(body.contains(cached), "v2 usage carries cached_tokens: {body}");

    let (st, head, body) = serve::post(s.http(), "/v1/completions", Some("2"), completion);
    assert_eq!(st, 200, "{body}");
    assert!(head.contains("deprecation: @90") && head.contains("sunset: "), "v2 /v1/completions is deprecated\n{head}");
    assert!(body.contains(cached), "{body}");

    let (st, _, body) = serve::post(s.http(), "/v1/chat/completions", Some("3"), chat);
    assert_eq!(st, 400, "{body}");
    assert!(body.contains(r#""code":"unsupported_api_version""#), "{body}");
    drop(s);

    // past the sunset: v2 is the default and v1 is gone
    let s = serve::start(Served::V2, 90, 100);
    let (st, head, body) = serve::post(s.http(), "/v1/chat/completions", Some("1"), chat);
    assert_eq!(st, 410, "{body}");
    assert!(body.contains(r#""code":"api_version_sunset""#), "{body}");
    assert!(head.contains("x-tl-api-version: 1"), "{head}");
    let (st, head, body) = serve::post(s.http(), "/v1/chat/completions", None, chat);
    assert_eq!(st, 200, "{body}");
    assert!(head.contains("x-tl-api-version: 2") && body.contains(cached), "{head}\n{body}");
}

/// A one-file engine for the side-by-side test: a byte-level bigram that
/// always predicts the next byte, served on ephemeral ports.
mod serve {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpStream};
    use std::path::PathBuf;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use tl_engine::runner::{KvConfig, PrefixCache};
    use tl_serve::api_version::ApiVersion;
    use tl_serve::server::{self, ApiConfig, ServeConfig, ServerHandle};

    pub struct Srv(Option<ServerHandle>);

    impl Srv {
        pub fn http(&self) -> SocketAddr {
            self.0.as_ref().unwrap().http_addr
        }
    }

    impl Drop for Srv {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                h.drain();
                let _ = h.join();
            }
        }
    }

    fn succ_model() -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("ol-craft_14-{}-{nanos}", std::process::id())).join("succ");
        std::fs::create_dir_all(&d).unwrap();
        let mut w = vec![0.0f32; 256 * 256];
        for i in 0..256 {
            w[i * 256 + (i + 1) % 256] = 2.0;
        }
        let data: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let header = format!(r#"{{"__metadata__":{{"format":"tinyllm"}},"bigram.weight":{{"dtype":"F32","shape":[256,256],"data_offsets":[0,{}]}}}}"#, data.len());
        let mut h = header.into_bytes();
        while h.len() % 8 != 0 {
            h.push(b' ');
        }
        let mut st = (h.len() as u64).to_le_bytes().to_vec();
        st.extend_from_slice(&h);
        st.extend_from_slice(&data);
        std::fs::write(d.join("model.safetensors"), st).unwrap();
        std::fs::write(d.join("config.json"), r#"{"tl_arch":"bigram","tl_tokenizer":"bytes","vocab_size":256,"tl_format":1}"#).unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"chat_template":"{% for m in messages %}{{ m.content }}{% endfor %}"}"#).unwrap();
        d
    }

    pub fn start(default: ApiVersion, deprecated_unix: i64, sunset_unix: i64) -> Srv {
        let mut c = ServeConfig::for_model(&succ_model());
        c.http_listen = "127.0.0.1:0".to_string();
        c.health_listen = "127.0.0.1:0".to_string();
        c.engine.kv = KvConfig { blocks: 64, block_size: 16 };
        c.engine.prefix_cache = PrefixCache::None;
        c.api = ApiConfig { default, deprecated_unix, sunset_unix };
        let h = server::spawn(c).expect("server::spawn");
        let s = Srv(Some(h));
        let t0 = Instant::now();
        while TcpStream::connect(s.0.as_ref().unwrap().health_addr).is_err() {
            assert!(t0.elapsed() < Duration::from_secs(5), "the health port never accepted");
            std::thread::sleep(Duration::from_millis(5));
        }
        s
    }

    /// (status, lower-cased head, body) of a non-streaming POST.
    pub fn post(addr: SocketAddr, path: &str, version: Option<&str>, body: &str) -> (u16, String, String) {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let v = version.map(|v| format!("X-TL-API-Version: {v}\r\n")).unwrap_or_default();
        let head = format!("POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n{v}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
        s.write_all(head.as_bytes()).unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).expect("read until the server closes");
        let split = out.windows(4).position(|w| w == b"\r\n\r\n").expect("header end");
        let head = String::from_utf8_lossy(&out[..split]).to_ascii_lowercase();
        let status = head.split(' ').nth(1).unwrap().parse().unwrap();
        (status, head, String::from_utf8_lossy(&out[split + 4..]).into_owned())
    }
}
