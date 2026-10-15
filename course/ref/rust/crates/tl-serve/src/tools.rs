//! Tool calls and response formats (L10.9): the HTTP side of constrained
//! decoding.
//!
//! The pieces, in request order:
//!
//! 1. [`parse_tool_request`] reads `tools`, `tool_choice`, `response_format`,
//!    and `parallel_tool_calls` from the raw request body (keys in the order
//!    written: a schema's properties keep their order) and validates them
//!    into a [`ToolRequest`] (400 or 422 with the OpenAI error shape).
//! 2. [`render_messages`] turns the request's messages into plain (role,
//!    content) pairs for the chat template: the tools are described in a
//!    system message, an assistant message's `tool_calls` become
//!    `<tool_call>` blocks, and a `tool` message becomes a `<tool_response>`
//!    block in a user turn.
//! 3. [`grammar`] gives the regex the engine constrains decoding to
//!    (tl_engine::constrain), and the text that triggers it in `auto` mode.
//! 4. [`ToolParser`] turns the generated text, piece by piece, into
//!    [`ToolEvent`]s: content outside calls, then each call's name and its
//!    argument fragments as they stream; [`delta_json`] and [`message_json`]
//!    write the OpenAI `tool_calls` deltas and message, and
//!    `finish_reason` is `tool_calls` when the answer called a tool.
//!
//! The call format is the Hermes/Qwen one that small instruct models are
//! trained on: `<tool_call>{"name":"<name>","arguments":<object>}</tool_call>`,
//! several calls separated by "\n".
//!
//! Chapter: content/courses/build-an-llm/p10-serving/09-tool-calls-and-constrained-decoding.md.
//! Conformance: `tools.call`, `tools.stream`, `tools.choice`
//! (course/conformance/openapi/cases, `requires = ["L10.9"]`).

use tl_engine::constrain::{escape, json_object_regex, json_quote, json_schema_to_regex, Json};
use tl_tok::Tokenizer;

use crate::openai::ApiError;
use crate::template::Template;

/// The text that opens a call; in `auto` mode it also switches the
/// constraint on.
pub const CALL_OPEN: &str = "<tool_call>";
/// The text that closes a call.
pub const CALL_CLOSE: &str = "</tool_call>";
/// Most calls one answer may make with parallel_tool_calls (the default).
pub const MAX_PARALLEL: usize = 4;
/// Nesting bound of the response_format json_object grammar.
pub const JSON_OBJECT_DEPTH: usize = 3;

/// A rejected request: status 400 or 422 with the OpenAI error fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolError {
    pub status: u16,
    pub param: String,
    pub code: Option<String>,
    pub message: String,
}

impl ToolError {
    fn invalid(param: &str, message: impl Into<String>) -> ToolError {
        // SOLUTION-BEGIN L10.9
        ToolError { status: 400, param: param.to_string(), code: None, message: message.into() }
        // SOLUTION-END
    }

    fn unsupported(param: &str, message: impl Into<String>) -> ToolError {
        // SOLUTION-BEGIN L10.9
        ToolError { status: 422, param: param.to_string(), code: Some("unsupported_parameter".to_string()), message: message.into() }
        // SOLUTION-END
    }

    /// The server's error (L10.5's ApiError): 422 `unsupported_parameter`
    /// or 400 `invalid_request_error`, naming the parameter.
    pub fn to_api_error(&self) -> ApiError {
        // SOLUTION-BEGIN L10.9
        if self.status == 422 {
            ApiError::unsupported(&self.param, self.message.clone())
        } else {
            ApiError::invalid(Some(self.param.as_str()).filter(|p| !p.is_empty()), self.message.clone())
        }
        // SOLUTION-END
    }

    /// `{"error":{"message","type","param","code"}}`.
    pub fn to_json(&self) -> String {
        // SOLUTION-BEGIN L10.9
        self.to_api_error().to_json()
        // SOLUTION-END
    }
}

/// One function the model may call.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    /// The JSON-schema subset of tl_engine::constrain; `{"type":"object",
    /// "properties":{}}` when the request gives none.
    pub parameters: Json,
}

/// `tool_choice`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model may answer in text or call tools (the default with tools).
    Auto,
    /// No call; the tools are still described to the model.
    None,
    /// At least one call.
    Required,
    /// Exactly one call, of this function.
    Named(String),
}

/// `response_format.type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

/// The validated tool-related fields of one chat request.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolRequest {
    pub tools: Vec<Tool>,
    pub choice: ToolChoice,
    pub response_format: ResponseFormat,
    pub parallel: bool,
}

fn valid_name(n: &str) -> bool {
    // SOLUTION-BEGIN L10.9
    !n.is_empty() && n.len() <= 64 && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    // SOLUTION-END
}

/// Reads and validates the tool fields of a chat request body:
///
/// * `tools`: an array of `{"type":"function","function":{"name",
///   "description"?, "parameters"?}}`; names match `[A-Za-z0-9_-]{1,64}`
///   and are distinct (400); parameters outside the schema subset are 422
///   `unsupported_parameter` naming the keyword.
/// * `tool_choice`: "auto" (the default with tools), "none", "required", or
///   `{"type":"function","function":{"name"}}` naming a listed tool (400);
///   any value other than "none" without tools is 400.
/// * `response_format`: `{"type":"text"}` or `{"type":"json_object"}`; any
///   other type is 422.
/// * `parallel_tool_calls`: a boolean, default true.
pub fn parse_tool_request(body: &str) -> Result<ToolRequest, ToolError> {
    // SOLUTION-BEGIN L10.9
    let doc = Json::parse(body).map_err(|e| ToolError::invalid("", format!("request body: {e}")))?;
    let mut tools = Vec::new();
    match doc.get("tools") {
        None | Some(Json::Null) => {}
        Some(Json::Arr(items)) => {
            for (i, t) in items.iter().enumerate() {
                let param = format!("tools[{i}]");
                if t.get("type") != Some(&Json::Str("function".into())) {
                    return Err(ToolError::invalid(&param, "every tool must have \"type\": \"function\""));
                }
                let Some(f) = t.get("function") else { return Err(ToolError::invalid(&param, "a tool needs a \"function\" object")) };
                let name = match f.get("name") {
                    Some(Json::Str(n)) if valid_name(n) => n.clone(),
                    _ => return Err(ToolError::invalid(&format!("{param}.function.name"), "function names match [A-Za-z0-9_-]{1,64}")),
                };
                if tools.iter().any(|x: &Tool| x.name == name) {
                    return Err(ToolError::invalid(&format!("{param}.function.name"), format!("duplicate function name {name:?}")));
                }
                let description = match f.get("description") {
                    Some(Json::Str(d)) => Some(d.clone()),
                    _ => None,
                };
                let parameters = match f.get("parameters") {
                    None | Some(Json::Null) => Json::Obj(vec![("type".into(), Json::Str("object".into())), ("properties".into(), Json::Obj(Vec::new()))]),
                    Some(p) => p.clone(),
                };
                if let Err(e) = json_schema_to_regex(&parameters) {
                    return Err(ToolError::unsupported(&format!("{param}.function.parameters"), e.0));
                }
                tools.push(Tool { name, description, parameters });
            }
        }
        Some(_) => return Err(ToolError::invalid("tools", "tools must be an array")),
    }
    let choice = match doc.get("tool_choice") {
        None | Some(Json::Null) => ToolChoice::Auto,
        Some(Json::Str(s)) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => return Err(ToolError::invalid("tool_choice", format!("tool_choice {other:?} is not auto, none, or required"))),
        },
        Some(obj @ Json::Obj(_)) => {
            let name = obj.get("function").and_then(|f| f.get("name"));
            match (obj.get("type"), name) {
                (Some(Json::Str(t)), Some(Json::Str(n))) if t == "function" => {
                    if !tools.iter().any(|x| &x.name == n) {
                        return Err(ToolError::invalid("tool_choice", format!("tool_choice names {n:?}, which is not in tools")));
                    }
                    ToolChoice::Named(n.clone())
                }
                _ => return Err(ToolError::invalid("tool_choice", "tool_choice object must be {\"type\":\"function\",\"function\":{\"name\":...}}")),
            }
        }
        Some(_) => return Err(ToolError::invalid("tool_choice", "tool_choice must be a string or an object")),
    };
    if tools.is_empty() && doc.get("tool_choice").is_some_and(|c| *c != Json::Str("none".into()) && *c != Json::Null) {
        return Err(ToolError::invalid("tool_choice", "tool_choice needs tools"));
    }
    let response_format = match doc.get("response_format") {
        None | Some(Json::Null) => ResponseFormat::Text,
        Some(rf) => match rf.get("type") {
            Some(Json::Str(t)) if t == "text" => ResponseFormat::Text,
            Some(Json::Str(t)) if t == "json_object" => ResponseFormat::JsonObject,
            Some(Json::Str(t)) => return Err(ToolError::unsupported("response_format", format!("response_format type {t:?} is not served (text, json_object)"))),
            _ => return Err(ToolError::invalid("response_format", "response_format needs a \"type\"")),
        },
    };
    let parallel = match doc.get("parallel_tool_calls") {
        None | Some(Json::Null) => true,
        Some(Json::Bool(b)) => *b,
        Some(_) => return Err(ToolError::invalid("parallel_tool_calls", "parallel_tool_calls must be a boolean")),
    };
    Ok(ToolRequest { tools, choice, response_format, parallel })
    // SOLUTION-END
}

/// What decoding is constrained to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grammar {
    /// The regex (tl_engine::constrain subset) of the constrained text.
    pub pattern: String,
    /// None: the whole answer is constrained from its first token. Some(t):
    /// the answer is free until it ends with t, and the text after t is
    /// constrained (auto mode: the model decides whether to call).
    pub trigger: Option<String>,
}

/// The regex of one call's JSON `{"name":"<name>","arguments":<args>}`.
pub fn call_body_regex(t: &Tool) -> String {
    // SOLUTION-BEGIN L10.9
    let args = json_schema_to_regex(&t.parameters).expect("validated by parse_tool_request");
    format!("{}{}{}{args}{}", escape("{\"name\":"), escape(&json_quote(&t.name)), escape(",\"arguments\":"), escape("}"))
    // SOLUTION-END
}

/// The grammar of a request, or None for free text:
///
/// | tool_choice | grammar |
/// |---|---|
/// | named f | `<tool_call>CALL_f</tool_call>` from the first token |
/// | required | one call (or up to MAX_PARALLEL, "\n"-separated) of any tool |
/// | auto | free text; after `<tool_call>`, the rest of a call |
/// | none | free text, or a JSON object with response_format json_object |
pub fn grammar(r: &ToolRequest) -> Option<Grammar> {
    // SOLUTION-BEGIN L10.9
    let open = escape(CALL_OPEN);
    let close = escape(CALL_CLOSE);
    let any_body = || format!("(?:{})", r.tools.iter().map(call_body_regex).collect::<Vec<_>>().join("|"));
    let calls = |body: &str| {
        let one = format!("{open}{body}{close}");
        if r.parallel && MAX_PARALLEL > 1 {
            format!("{one}(?:\\n{one}){{0,{}}}", MAX_PARALLEL - 1)
        } else {
            one
        }
    };
    match (&r.choice, r.tools.is_empty()) {
        (ToolChoice::Named(n), false) => {
            let t = r.tools.iter().find(|t| &t.name == n)?;
            Some(Grammar { pattern: format!("{open}{}{close}", call_body_regex(t)), trigger: None })
        }
        (ToolChoice::Required, false) => Some(Grammar { pattern: calls(&any_body()), trigger: None }),
        (ToolChoice::Auto, false) => Some(Grammar { pattern: format!("{}{close}", any_body()), trigger: Some(CALL_OPEN.to_string()) }),
        _ => match r.response_format {
            ResponseFormat::JsonObject => Some(Grammar { pattern: json_object_regex(JSON_OBJECT_DEPTH), trigger: None }),
            ResponseFormat::Text => None,
        },
    }
    // SOLUTION-END
}

/// The system text that describes the tools to the model.
pub fn tools_system_prompt(tools: &[Tool]) -> String {
    // SOLUTION-BEGIN L10.9
    let mut s = String::from("You can call functions. The available functions, one JSON object per line:\n");
    for t in tools {
        let mut f = vec![("name".to_string(), Json::Str(t.name.clone()))];
        if let Some(d) = &t.description {
            f.push(("description".to_string(), Json::Str(d.clone())));
        }
        f.push(("parameters".to_string(), t.parameters.clone()));
        s.push_str(&Json::Obj(vec![("type".into(), Json::Str("function".into())), ("function".into(), Json::Obj(f))]).dumps());
        s.push('\n');
    }
    s.push_str("To call a function, answer with ");
    s.push_str(CALL_OPEN);
    s.push_str("{\"name\":<function name>,\"arguments\":<arguments object>}");
    s.push_str(CALL_CLOSE);
    s.push_str(", one per call.");
    s
    // SOLUTION-END
}

/// The request's `messages` as (role, content) pairs for the chat
/// template. With tools, their description is the first system message
/// (appended to an existing one); an assistant message's `tool_calls` are
/// written back as `<tool_call>` blocks after its content; a `tool` message
/// becomes a user turn `<tool_response>{content}</tool_response>`. Error
/// (400) for a message without a string role.
pub fn render_messages(messages: &Json, tools: &[Tool]) -> Result<Vec<(String, String)>, ToolError> {
    // SOLUTION-BEGIN L10.9
    let Json::Arr(ms) = messages else { return Err(ToolError::invalid("messages", "messages must be an array")) };
    let mut out: Vec<(String, String)> = Vec::new();
    for (i, m) in ms.iter().enumerate() {
        let Some(Json::Str(role)) = m.get("role") else { return Err(ToolError::invalid(&format!("messages[{i}].role"), "every message needs a role")) };
        let content = match m.get("content") {
            Some(Json::Str(c)) => c.clone(),
            _ => String::new(),
        };
        match role.as_str() {
            "tool" => out.push(("user".into(), format!("<tool_response>{content}</tool_response>"))),
            "assistant" => {
                let mut text = content;
                if let Some(Json::Arr(calls)) = m.get("tool_calls") {
                    for c in calls {
                        let f = c.get("function");
                        let name = f.and_then(|f| f.get("name")).cloned().unwrap_or(Json::Null);
                        let args = match f.and_then(|f| f.get("arguments")) {
                            Some(Json::Str(a)) => Json::parse(a).unwrap_or(Json::Str(a.clone())),
                            Some(other) => other.clone(),
                            None => Json::Obj(Vec::new()),
                        };
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(CALL_OPEN);
                        text.push_str(&Json::Obj(vec![("name".into(), name), ("arguments".into(), args)]).dumps());
                        text.push_str(CALL_CLOSE);
                    }
                }
                out.push(("assistant".into(), text));
            }
            r => out.push((r.to_string(), content)),
        }
    }
    if !tools.is_empty() {
        let desc = tools_system_prompt(tools);
        match out.first_mut() {
            Some((r, c)) if r == "system" => {
                c.push_str("\n\n");
                c.push_str(&desc);
            }
            _ => out.insert(0, ("system".into(), desc)),
        }
    }
    Ok(out)
    // SOLUTION-END
}

/// The prompt of a tool request: [`render_messages`] through the model's
/// chat template (L10.5), ending with the generation prompt.
pub fn render_prompt(t: &Template, messages: &Json, tools: &[Tool], bos: &str, eos: &str) -> Result<String, ToolError> {
    // SOLUTION-BEGIN L10.9
    let pairs = render_messages(messages, tools)?;
    let refs: Vec<(&str, &str)> = pairs.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
    t.render_messages(&refs, true, bos, eos).map_err(|e| ToolError::invalid("messages", format!("chat template: {e}")))
    // SOLUTION-END
}

/// The bytes of every token for the constraint's TokenIndex: None for the
/// ids in `specials` (EOS, chat markers, added tokens), which a grammar never
/// allows.
pub fn vocab_bytes(tok: &dyn Tokenizer, specials: &[u32]) -> Vec<Option<Vec<u8>>> {
    // SOLUTION-BEGIN L10.9
    (0..tok.vocab_size()).map(|id| if specials.contains(&id) { None } else { tok.token_bytes(id).map(|b| b.to_vec()) }).collect()
    // SOLUTION-END
}

/// What the generated text holds, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolEvent {
    /// Text outside any call.
    Content(String),
    /// A call starts: its index in the answer, id, and function name.
    CallStart { index: usize, id: String, name: String },
    /// The next piece of a call's arguments (JSON text).
    Arguments { index: usize, fragment: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    /// Outside a call; `pending` may end with a prefix of CALL_OPEN.
    Text,
    /// After CALL_OPEN, before the arguments value starts.
    Header,
    /// Inside the arguments value: nesting depth, in a string, after a backslash.
    Args { depth: usize, in_str: bool, esc: bool, started: bool },
    /// After the arguments value: expecting `}` then CALL_CLOSE.
    Tail,
}

/// Turns generated text into [`ToolEvent`]s as it streams. Text that may be
/// the start of `<tool_call>` is held back until the next piece decides it;
/// a call's header `{"name":"<n>","arguments":` (whitespace allowed between
/// tokens) gives CallStart, and the arguments value is passed on fragment by
/// fragment until its closing brace. Markup is never content.
pub struct ToolParser {
    request_id: String,
    mode: Mode,
    pending: String,
    calls: usize,
}

/// The longest suffix of `s` that is a proper prefix of `pat`.
fn partial_suffix(s: &str, pat: &str) -> usize {
    // SOLUTION-BEGIN L10.9
    (1..pat.len()).rev().find(|&k| s.len() >= k && s.is_char_boundary(s.len() - k) && pat.starts_with(&s[s.len() - k..])).unwrap_or(0)
    // SOLUTION-END
}

/// Parses `{"name":"<n>","arguments":` at the start of `s` (JSON
/// whitespace allowed between tokens): Ok(Some((name, consumed))) when
/// complete, Ok(None) when `s` is a prefix of one, Err when it cannot be.
fn parse_header(s: &str) -> Result<Option<(String, usize)>, ()> {
    // SOLUTION-BEGIN L10.9
    let b = s.as_bytes();
    let mut i = 0;
    let ws = |i: &mut usize| {
        while *i < b.len() && matches!(b[*i], b' ' | b'\n' | b'\t' | b'\r') {
            *i += 1;
        }
    };
    // literal: matches fully (true), is cut short by the end (None), or differs (Err)
    let lit = |i: &mut usize, l: &[u8]| -> Result<bool, ()> {
        let n = l.len().min(b.len() - *i);
        if b[*i..*i + n] != l[..n] {
            return Err(());
        }
        *i += n;
        Ok(n == l.len())
    };
    ws(&mut i);
    if !lit(&mut i, b"{")? {
        return Ok(None);
    }
    ws(&mut i);
    if !lit(&mut i, b"\"name\"")? {
        return Ok(None);
    }
    ws(&mut i);
    if !lit(&mut i, b":")? {
        return Ok(None);
    }
    ws(&mut i);
    if !lit(&mut i, b"\"")? {
        return Ok(None);
    }
    let start = i;
    let mut esc = false;
    loop {
        if i >= b.len() {
            return Ok(None);
        }
        match b[i] {
            b'\\' if !esc => esc = true,
            b'"' if !esc => break,
            _ => esc = false,
        }
        i += 1;
    }
    let name = match Json::parse(&s[start - 1..=i]) {
        Ok(Json::Str(n)) => n,
        _ => return Err(()),
    };
    i += 1;
    ws(&mut i);
    if !lit(&mut i, b",")? {
        return Ok(None);
    }
    ws(&mut i);
    if !lit(&mut i, b"\"arguments\"")? {
        return Ok(None);
    }
    ws(&mut i);
    if !lit(&mut i, b":")? {
        return Ok(None);
    }
    ws(&mut i);
    Ok(Some((name, i)))
    // SOLUTION-END
}

impl ToolParser {
    pub fn new(request_id: &str) -> ToolParser {
        // SOLUTION-BEGIN L10.9
        ToolParser { request_id: request_id.to_string(), mode: Mode::Text, pending: String::new(), calls: 0 }
        // SOLUTION-END
    }

    /// The id of call `index`: `call_<request_id>_<index>`.
    pub fn call_id(&self, index: usize) -> String {
        // SOLUTION-BEGIN L10.9
        format!("call_{}_{index}", self.request_id)
        // SOLUTION-END
    }

    /// Calls started so far.
    pub fn calls(&self) -> usize {
        // SOLUTION-BEGIN L10.9
        self.calls
        // SOLUTION-END
    }

    /// Feeds the next piece of generated text.
    pub fn push(&mut self, piece: &str) -> Vec<ToolEvent> {
        // SOLUTION-BEGIN L10.9
        self.pending.push_str(piece);
        let mut out = Vec::new();
        loop {
            match self.mode.clone() {
                Mode::Text => {
                    if let Some(at) = self.pending.find(CALL_OPEN) {
                        let before = self.pending[..at].to_string();
                        // "\n" between two calls is markup, not content.
                        if !(self.calls > 0 && before.trim().is_empty()) && !before.is_empty() {
                            out.push(ToolEvent::Content(before));
                        }
                        self.pending.drain(..at + CALL_OPEN.len());
                        self.mode = Mode::Header;
                        continue;
                    }
                    let keep = partial_suffix(&self.pending, CALL_OPEN);
                    let emit = self.pending.len() - keep;
                    if emit > 0 {
                        let text: String = self.pending.drain(..emit).collect();
                        if !(self.calls > 0 && text.trim().is_empty()) {
                            out.push(ToolEvent::Content(text));
                        }
                    }
                    return out;
                }
                Mode::Header => match parse_header(&self.pending) {
                    Ok(None) => return out,
                    Ok(Some((name, used))) => {
                        let index = self.calls;
                        self.calls += 1;
                        out.push(ToolEvent::CallStart { index, id: self.call_id(index), name });
                        self.pending.drain(..used);
                        self.mode = Mode::Args { depth: 0, in_str: false, esc: false, started: false };
                    }
                    Err(()) => {
                        // Not a call after all: the markup was content.
                        let text = format!("{CALL_OPEN}{}", self.pending);
                        self.pending.clear();
                        self.mode = Mode::Text;
                        out.push(ToolEvent::Content(text));
                        return out;
                    }
                },
                Mode::Args { mut depth, mut in_str, mut esc, mut started } => {
                    let index = self.calls - 1;
                    let mut end = None;
                    for (k, c) in self.pending.char_indices() {
                        if in_str {
                            match c {
                                _ if esc => esc = false,
                                '\\' => esc = true,
                                '"' => in_str = false,
                                _ => {}
                            }
                        } else {
                            match c {
                                '"' => in_str = true,
                                '{' | '[' => {
                                    depth += 1;
                                    started = true;
                                }
                                '}' | ']' if depth == 0 => {
                                    // a scalar value ended at the call's own `}`
                                    end = Some(k);
                                    break;
                                }
                                '}' | ']' => {
                                    depth -= 1;
                                    if depth == 0 {
                                        end = Some(k + 1);
                                        break;
                                    }
                                }
                                c if !started && !c.is_whitespace() => started = true,
                                _ => {}
                            }
                        }
                    }
                    match end {
                        Some(e) => {
                            let frag: String = self.pending.drain(..e).collect();
                            out.push(ToolEvent::Arguments { index, fragment: frag });
                            self.mode = Mode::Tail;
                        }
                        None => {
                            let frag: String = std::mem::take(&mut self.pending);
                            if !frag.is_empty() {
                                out.push(ToolEvent::Arguments { index, fragment: frag });
                            }
                            self.mode = Mode::Args { depth, in_str, esc, started };
                            return out;
                        }
                    }
                }
                Mode::Tail => {
                    // `}` then `</tool_call>`, whitespace allowed before each.
                    let t = self.pending.trim_start();
                    if t.is_empty() {
                        return out;
                    }
                    let after = t.strip_prefix('}').map(str::trim_start);
                    match after {
                        Some(a) if a.starts_with(CALL_CLOSE) => {
                            let consumed = self.pending.len() - (a.len() - CALL_CLOSE.len());
                            self.pending.drain(..consumed);
                            self.mode = Mode::Text;
                            continue;
                        }
                        Some(a) if CALL_CLOSE.starts_with(a) => return out,
                        _ => {
                            // A malformed end: the rest is content.
                            let text = std::mem::take(&mut self.pending);
                            self.mode = Mode::Text;
                            out.push(ToolEvent::Content(text));
                            return out;
                        }
                    }
                }
            }
        }
        // SOLUTION-END
    }

    /// The end of the answer: text held back as a possible `<tool_call>`
    /// prefix is content after all; an unfinished call stays as far as it
    /// got (finish_reason "length" tells the client).
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        // SOLUTION-BEGIN L10.9
        let rest = std::mem::take(&mut self.pending);
        match self.mode {
            Mode::Text if !rest.is_empty() && !(self.calls > 0 && rest.trim().is_empty()) => vec![ToolEvent::Content(rest)],
            Mode::Header if !rest.is_empty() || self.calls == 0 => {
                self.mode = Mode::Text;
                vec![ToolEvent::Content(format!("{CALL_OPEN}{rest}"))]
            }
            _ => Vec::new(),
        }
        // SOLUTION-END
    }
}

/// One complete call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON text of the arguments object.
    pub arguments: String,
}

/// Folds events into the content text (None when there is none) and the
/// calls, in order.
pub fn collect(events: &[ToolEvent]) -> (Option<String>, Vec<ToolCall>) {
    // SOLUTION-BEGIN L10.9
    let mut content = String::new();
    let mut calls: Vec<ToolCall> = Vec::new();
    for e in events {
        match e {
            ToolEvent::Content(t) => content.push_str(t),
            ToolEvent::CallStart { index, id, name } => {
                debug_assert_eq!(*index, calls.len());
                calls.push(ToolCall { id: id.clone(), name: name.clone(), arguments: String::new() });
            }
            ToolEvent::Arguments { index, fragment } => calls[*index].arguments.push_str(fragment),
        }
    }
    ((!content.is_empty()).then_some(content), calls)
    // SOLUTION-END
}

/// The finish_reason of an answer: "tool_calls" when it called a tool, else
/// the engine's own ("stop" or "length").
pub fn finish_reason(calls: usize, engine: &str) -> String {
    // SOLUTION-BEGIN L10.9
    if calls > 0 {
        "tool_calls".to_string()
    } else {
        engine.to_string()
    }
    // SOLUTION-END
}

/// The non-stream `choices[0].message`: content (null when empty) and
/// tool_calls (omitted when none).
pub fn message_json(content: Option<&str>, calls: &[ToolCall]) -> String {
    // SOLUTION-BEGIN L10.9
    let mut s = format!(r#"{{"role":"assistant","content":{}"#, content.map(json_quote).unwrap_or_else(|| "null".to_string()));
    if !calls.is_empty() {
        let items: Vec<String> = calls
            .iter()
            .map(|c| format!(r#"{{"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}"#, json_quote(&c.id), json_quote(&c.name), json_quote(&c.arguments)))
            .collect();
        s.push_str(&format!(r#","tool_calls":[{}]"#, items.join(",")));
    }
    s.push('}');
    s
    // SOLUTION-END
}

/// The `delta` object of one stream chunk for an event: content, a call's
/// first delta (id, type, name, empty arguments), or an arguments fragment.
pub fn delta_json(e: &ToolEvent) -> String {
    // SOLUTION-BEGIN L10.9
    match e {
        ToolEvent::Content(t) => format!(r#"{{"content":{}}}"#, json_quote(t)),
        ToolEvent::CallStart { index, id, name } => format!(
            r#"{{"tool_calls":[{{"index":{index},"id":{},"type":"function","function":{{"name":{},"arguments":""}}}}]}}"#,
            json_quote(id),
            json_quote(name)
        ),
        ToolEvent::Arguments { index, fragment } => format!(r#"{{"tool_calls":[{{"index":{index},"function":{{"arguments":{}}}}}]}}"#, json_quote(fragment)),
    }
    // SOLUTION-END
}
