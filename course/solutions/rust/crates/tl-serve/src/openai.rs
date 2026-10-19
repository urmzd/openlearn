//! The OpenAI wire format of API v1, engine tier (L10.5;
//! contracts/openapi/openai-subset.v1.yaml): request validation into typed
//! values, and the response and chunk documents.
//!
//! Validation follows the contract's rules: a value out of range is 400
//! `invalid_request_error` naming the field in `param`; a supported field
//! with an unsupported value (`n > 1`, `echo: true`, content as an array of
//! parts, and until L10.9 `tools`, `tool_choice`, `response_format`) is 422
//! `unsupported_parameter`; unknown fields are ignored.

use serde_json::{json, Map, Value};
use tl_engine::sample::SamplingParams;

/// An error response: status, `type`, `message`, `param`, `code`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<&'static str>,
}

impl ApiError {
    /// 400 `invalid_request_error`.
    pub fn invalid(param: Option<&str>, message: impl Into<String>) -> ApiError {
        ApiError { status: 400, kind: "invalid_request_error", message: message.into(), param: param.map(str::to_string), code: None }
    }

    /// 422 `unsupported_parameter`.
    pub fn unsupported(param: &str, message: impl Into<String>) -> ApiError {
        ApiError { status: 422, kind: "invalid_request_error", message: message.into(), param: Some(param.to_string()), code: Some("unsupported_parameter") }
    }

    /// 404 `model_not_found`.
    pub fn model_not_found(model: &str) -> ApiError {
        ApiError { status: 404, kind: "invalid_request_error", message: format!("model {model:?} is not served here"), param: Some("model".to_string()), code: Some("model_not_found") }
    }

    /// 400 `context_length_exceeded`.
    pub fn context_length(message: impl Into<String>) -> ApiError {
        ApiError { status: 400, kind: "invalid_request_error", message: message.into(), param: Some("messages".to_string()), code: Some("context_length_exceeded") }
    }

    /// 429 `rate_limit_exceeded`: the admission queue is full.
    pub fn queue_full() -> ApiError {
        ApiError { status: 429, kind: "rate_limit_error", message: "the engine's admission queue is full; retry later".to_string(), param: None, code: Some("rate_limit_exceeded") }
    }

    /// 503 `no_capacity`: draining, or no KV blocks before the deadline.
    pub fn no_capacity(message: impl Into<String>) -> ApiError {
        ApiError { status: 503, kind: "server_error", message: message.into(), param: None, code: Some("no_capacity") }
    }

    /// 500 `internal_error`.
    pub fn internal(message: impl Into<String>) -> ApiError {
        ApiError { status: 500, kind: "server_error", message: message.into(), param: None, code: Some("internal_error") }
    }

    /// `{"error": {"message", "type", "param", "code"}}`.
    pub fn to_json(&self) -> String {
        json!({"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}}).to_string()
    }
}

/// One chat message: role and text content (`None`: an assistant turn with
/// no text).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
}

/// The prompt of a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prompt {
    Text(String),
    Chat(Vec<ChatMessage>),
}

/// A validated completion or chat request.
#[derive(Clone, Debug, PartialEq)]
pub struct GenSpec {
    pub model: String,
    pub prompt: Prompt,
    pub params: SamplingParams,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    /// `None`: the default (16 for completions, the rest of the context for chat).
    pub max_tokens: Option<usize>,
    pub stream: bool,
    pub include_usage: bool,
    /// Report each token's logprob (chat `logprobs: true`).
    pub logprobs: bool,
}

/// The request body as a JSON object, or 400.
fn object(body: &[u8]) -> Result<Map<String, Value>, ApiError> {
    let v: Value = serde_json::from_slice(body).map_err(|e| ApiError::invalid(None, format!("the body is not valid JSON: {e}")))?;
    match v {
        Value::Object(m) => Ok(m),
        _ => Err(ApiError::invalid(None, "the body must be a JSON object")),
    }
}

/// A number field within [lo, hi] (`lo_open`: lo excluded), or its default.
fn number(m: &Map<String, Value>, key: &str, default: f64, lo: f64, lo_open: bool, hi: f64) -> Result<f64, ApiError> {
    match m.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => {
            let x = n.as_f64().unwrap_or(f64::NAN);
            let low_ok = if lo_open { x > lo } else { x >= lo };
            if low_ok && x <= hi {
                Ok(x)
            } else {
                Err(ApiError::invalid(Some(key), format!("{key} must be in {}{lo}, {hi}], got {x}", if lo_open { "(" } else { "[" })))
            }
        }
        Some(_) => Err(ApiError::invalid(Some(key), format!("{key} must be a number"))),
    }
}

/// An optional integer field >= lo.
fn integer(m: &Map<String, Value>, key: &str, lo: i64) -> Result<Option<i64>, ApiError> {
    match m.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_i64() {
            Some(x) if x >= lo => Ok(Some(x)),
            _ => Err(ApiError::invalid(Some(key), format!("{key} must be an integer >= {lo}"))),
        },
    }
}

/// An optional boolean field.
fn boolean(m: &Map<String, Value>, key: &str, default: bool) -> Result<bool, ApiError> {
    match m.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(ApiError::invalid(Some(key), format!("{key} must be a boolean"))),
    }
}

/// `model`: a non-empty string.
fn model(m: &Map<String, Value>) -> Result<String, ApiError> {
    match m.get("model") {
        Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        _ => Err(ApiError::invalid(Some("model"), "model must be a non-empty string")),
    }
}

/// The fields both completion endpoints share (SamplingFields).
fn sampling(m: &Map<String, Value>) -> Result<(SamplingParams, Option<u64>, Vec<String>, bool, bool), ApiError> {
    let params = SamplingParams {
        temperature: number(m, "temperature", 1.0, 0.0, false, 2.0)?,
        top_p: number(m, "top_p", 1.0, 0.0, true, 1.0)?,
        top_k: integer(m, "top_k", 0)?.unwrap_or(0) as usize,
        min_p: number(m, "min_p", 0.0, 0.0, false, 1.0)?,
        repetition_penalty: number(m, "repetition_penalty", 1.0, 0.0, true, 10.0)?,
        presence_penalty: number(m, "presence_penalty", 0.0, -2.0, false, 2.0)?,
        frequency_penalty: number(m, "frequency_penalty", 0.0, -2.0, false, 2.0)?,
    };
    let seed = integer(m, "seed", 0)?.map(|s| s as u64);
    let stop = match m.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(a)) if (1..=4).contains(&a.len()) => a
            .iter()
            .map(|x| match x {
                Value::String(s) if !s.is_empty() => Ok(s.clone()),
                _ => Err(ApiError::invalid(Some("stop"), "each stop must be a non-empty string")),
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(ApiError::invalid(Some("stop"), "stop must be a non-empty string or an array of 1 to 4")),
    };
    let stream = boolean(m, "stream", false)?;
    let include_usage = match m.get("stream_options") {
        None | Some(Value::Null) => false,
        Some(Value::Object(o)) => boolean(o, "include_usage", false).map_err(|_| ApiError::invalid(Some("stream_options"), "include_usage must be a boolean"))?,
        Some(_) => return Err(ApiError::invalid(Some("stream_options"), "stream_options must be an object")),
    };
    Ok((params, seed, stop, stream, include_usage))
}

/// `POST /v1/chat/completions`.
pub fn parse_chat(body: &[u8]) -> Result<GenSpec, ApiError> {
    let m = object(body)?;
    let model = model(&m)?;
    for key in ["tools", "tool_choice", "response_format"] {
        if m.get(key).is_some_and(|v| !v.is_null()) {
            return Err(ApiError::unsupported(key, format!("{key} is not supported by this engine yet (L10.9)")));
        }
    }
    if integer(&m, "n", 1)?.unwrap_or(1) != 1 {
        return Err(ApiError::unsupported("n", "only n = 1 is supported"));
    }
    let msgs = m.get("messages").and_then(Value::as_array).filter(|a| !a.is_empty()).ok_or_else(|| ApiError::invalid(Some("messages"), "messages must be a non-empty array"))?;
    let mut messages = Vec::with_capacity(msgs.len());
    for (i, msg) in msgs.iter().enumerate() {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
        if !["system", "user", "assistant", "tool"].contains(&role) {
            return Err(ApiError::invalid(Some("messages"), format!("messages[{i}].role must be system, user, assistant, or tool")));
        }
        let content = match msg.get("content") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(_)) => return Err(ApiError::unsupported("messages", "content as an array of parts is not supported")),
            None | Some(Value::Null) if role == "assistant" => None,
            _ => return Err(ApiError::invalid(Some("messages"), format!("messages[{i}].content must be a string"))),
        };
        messages.push(ChatMessage { role: role.to_string(), content });
    }
    let (params, seed, stop, stream, include_usage) = sampling(&m)?;
    let max_tokens = match integer(&m, "max_completion_tokens", 1)? {
        Some(n) => Some(n as usize),
        None => integer(&m, "max_tokens", 1)?.map(|n| n as usize),
    };
    let logprobs = boolean(&m, "logprobs", false)?;
    if integer(&m, "top_logprobs", 0)?.unwrap_or(0) > 0 {
        return Err(ApiError::unsupported("top_logprobs", "top_logprobs > 0 is not supported by this engine"));
    }
    Ok(GenSpec { model, prompt: Prompt::Chat(messages), params, seed, stop, max_tokens, stream, include_usage, logprobs })
}

/// `POST /v1/completions`.
pub fn parse_completion(body: &[u8]) -> Result<GenSpec, ApiError> {
    let m = object(body)?;
    let model = model(&m)?;
    let prompt = match m.get("prompt") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => return Err(ApiError::invalid(Some("prompt"), "prompt must be a non-empty string")),
    };
    if boolean(&m, "echo", false)? {
        return Err(ApiError::unsupported("echo", "echo is not supported"));
    }
    if integer(&m, "n", 1)?.unwrap_or(1) != 1 {
        return Err(ApiError::unsupported("n", "only n = 1 is supported"));
    }
    if integer(&m, "logprobs", 0)?.unwrap_or(0) > 0 {
        return Err(ApiError::unsupported("logprobs", "logprobs > 0 is not supported for completions by this engine"));
    }
    let (params, seed, stop, stream, include_usage) = sampling(&m)?;
    let max_tokens = Some(integer(&m, "max_tokens", 1)?.unwrap_or(16) as usize);
    Ok(GenSpec { model, prompt: Prompt::Text(prompt), params, seed, stop, max_tokens, stream, include_usage, logprobs: false })
}

/// `POST /v1/embeddings`: (model, inputs).
pub fn parse_embeddings(body: &[u8]) -> Result<(String, Vec<String>), ApiError> {
    let m = object(body)?;
    let model = model(&m)?;
    let inputs = match m.get("input") {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(a)) if (1..=256).contains(&a.len()) => a
            .iter()
            .map(|x| match x {
                Value::String(s) if !s.is_empty() => Ok(s.clone()),
                _ => Err(ApiError::invalid(Some("input"), "each input must be a non-empty string")),
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(ApiError::invalid(Some("input"), "input must be a non-empty string or an array of 1 to 256")),
    };
    match m.get("encoding_format") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == "float" => {}
        Some(_) => return Err(ApiError::unsupported("encoding_format", "only float is supported")),
    }
    Ok((model, inputs))
}

/// `POST /v1/tokenize`: (model, text, add_special).
pub fn parse_tokenize(body: &[u8]) -> Result<(String, String, bool), ApiError> {
    let m = object(body)?;
    let model = model(&m)?;
    let text = m.get("text").and_then(Value::as_str).ok_or_else(|| ApiError::invalid(Some("text"), "text must be a string"))?.to_string();
    Ok((model, text, boolean(&m, "add_special", false)?))
}

/// `usage`.
pub fn usage(prompt: usize, completion: usize) -> Value {
    json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion})
}

/// One generated token as a chat logprobs entry.
pub fn token_logprob(text: &str, bytes: &[u8], logprob: f64) -> Value {
    json!({"token": text, "logprob": logprob, "bytes": bytes, "top_logprobs": []})
}

/// A non-streamed `chat.completion`.
pub fn chat_completion(id: &str, created: u64, model: &str, content: &str, finish: &str, usage: Value, logprobs: Option<Vec<Value>>) -> String {
    let mut choice = json!({"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish});
    if let Some(lp) = logprobs {
        choice["logprobs"] = json!({"content": lp});
    }
    json!({"id": id, "object": "chat.completion", "created": created, "model": model, "choices": [choice], "usage": usage}).to_string()
}

/// A `chat.completion.chunk`: `role` only in the first, `finish` only in the
/// last content chunk.
pub fn chat_chunk(id: &str, created: u64, model: &str, role: bool, content: &str, finish: Option<&str>, logprobs: Option<Vec<Value>>) -> String {
    let mut delta = json!({"content": content});
    if role {
        delta["role"] = json!("assistant");
    }
    let mut choice = json!({"index": 0, "delta": delta, "finish_reason": finish});
    if let Some(lp) = logprobs {
        choice["logprobs"] = json!({"content": lp});
    }
    json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [choice]}).to_string()
}

/// The usage chunk of `stream_options.include_usage`: no choices.
pub fn usage_chunk(id: &str, created: u64, model: &str, object: &str, usage: Value) -> String {
    json!({"id": id, "object": object, "created": created, "model": model, "choices": [], "usage": usage}).to_string()
}

/// A non-streamed `text_completion`.
pub fn completion(id: &str, created: u64, model: &str, text: &str, finish: &str, usage: Value) -> String {
    json!({"id": id, "object": "text_completion", "created": created, "model": model,
           "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})
    .to_string()
}

/// One streamed `text_completion` chunk.
pub fn completion_chunk(id: &str, created: u64, model: &str, text: &str, finish: Option<&str>) -> String {
    json!({"id": id, "object": "text_completion", "created": created, "model": model,
           "choices": [{"index": 0, "text": text, "finish_reason": finish}]})
    .to_string()
}

/// The `list` of embeddings, in input order.
pub fn embeddings(model: &str, vectors: &[Vec<f32>], prompt_tokens: usize) -> String {
    let data: Vec<Value> = vectors.iter().enumerate().map(|(i, v)| json!({"object": "embedding", "index": i, "embedding": v})).collect();
    json!({"object": "list", "model": model, "data": data, "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}}).to_string()
}

/// One `model` object.
pub fn model_object(id: &str, created: u64) -> Value {
    json!({"id": id, "object": "model", "created": created, "owned_by": "tinyllm"})
}
