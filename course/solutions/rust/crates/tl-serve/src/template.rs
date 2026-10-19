//! Chat templates (L10.5): the Jinja subset of formats/generation-config
//! .schema.json, enough for the chat templates Hugging Face models ship.
//!
//! A chat model was trained on conversations written in one exact text
//! format (roles, separators, special tokens). The model directory carries
//! that format as a template; the server renders `messages` through it and
//! appends the generation prompt (the opening of an assistant turn), so the
//! model continues as the assistant.
//!
//! Supported: text, `{{ expr }}`, `{% for x in xs %}...{% endfor %}` with
//! `loop.first`, `loop.last`, `loop.index`, `loop.index0`;
//! `{% if %}...{% elif %}...{% else %}...{% endif %}`; whitespace control
//! `{%-`, `-%}`, `{{-`, `-}}`; expressions with string, integer, `true`,
//! `false`, `none` literals, variables, `a.b`, `a['b']`, `a[0]`, `==`, `!=`,
//! `and`, `or`, `not`, `+` and `~` (string concatenation), `is defined`,
//! `is not defined`, and the filters `trim`, `tojson`, `length`. Anything
//! else is a parse error, reported once at load, never per request.

use serde_json::Value;

/// A parsed template.
#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    nodes: Vec<Node>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Text(String),
    Out(Expr),
    For { var: String, iter: Expr, body: Vec<Node> },
    If { arms: Vec<(Expr, Vec<Node>)>, otherwise: Vec<Node> },
}

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Lit(Value),
    Var(String),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>, bool),
    Concat(Box<Expr>, Box<Expr>),
    Defined(Box<Expr>, bool),
    Filter(Box<Expr>, String),
}

/// A segment of the source: text, `{{ }}`, or `{% %}`, with its trim flags.
#[derive(Clone, Debug, PartialEq)]
enum Seg {
    Text(String),
    Expr(String),
    Stmt(String),
}

/// Splits the source into segments, applying `-` whitespace control.
fn segments(src: &str) -> Result<Vec<Seg>, String> {
    let mut out = Vec::new();
    let mut rest = src;
    let mut trim_next = false;
    while !rest.is_empty() {
        let next = [rest.find("{{"), rest.find("{%")].into_iter().flatten().min();
        let Some(i) = next else {
            let t = if trim_next { rest.trim_start() } else { rest };
            out.push(Seg::Text(t.to_string()));
            break;
        };
        let mut text = &rest[..i];
        if trim_next {
            text = text.trim_start();
        }
        let is_expr = rest[i..].starts_with("{{");
        let close = if is_expr { "}}" } else { "%}" };
        let mut inner_start = i + 2;
        if rest[inner_start..].starts_with('-') {
            text = text.trim_end();
            inner_start += 1;
        }
        if !text.is_empty() {
            out.push(Seg::Text(text.to_string()));
        }
        let end = rest[inner_start..].find(close).ok_or_else(|| format!("template: unclosed {}", &rest[i..i + 2]))? + inner_start;
        let mut inner = &rest[inner_start..end];
        trim_next = inner.ends_with('-');
        if trim_next {
            inner = &inner[..inner.len() - 1];
        }
        let inner = inner.trim().to_string();
        out.push(if is_expr { Seg::Expr(inner) } else { Seg::Stmt(inner) });
        rest = &rest[end + 2..];
    }
    Ok(out)
}

/// Expression tokens.
#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Str(String),
    Int(i64),
    Ident(String),
    Sym(&'static str),
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '\'' || c == '"' {
            let q = c;
            let mut t = String::new();
            i += 1;
            loop {
                let ch = *b.get(i).ok_or("template: unterminated string literal")?;
                i += 1;
                if ch == q {
                    break;
                }
                if ch == '\\' {
                    let e = *b.get(i).ok_or("template: unterminated escape")?;
                    i += 1;
                    t.push(match e {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        other => other,
                    });
                } else {
                    t.push(ch);
                }
            }
            out.push(Tok::Str(t));
        } else if c.is_ascii_digit() {
            let st = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let t: String = b[st..i].iter().collect();
            out.push(Tok::Int(t.parse().map_err(|_| format!("template: bad number {t}"))?));
        } else if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(b[st..i].iter().collect()));
        } else {
            let two: String = b[i..(i + 2).min(b.len())].iter().collect();
            let sym = match two.as_str() {
                "==" => Some("=="),
                "!=" => Some("!="),
                _ => None,
            };
            if let Some(s2) = sym {
                out.push(Tok::Sym(s2));
                i += 2;
                continue;
            }
            let one = match c {
                '.' => ".",
                '[' => "[",
                ']' => "]",
                '(' => "(",
                ')' => ")",
                '+' => "+",
                '~' => "~",
                '|' => "|",
                _ => return Err(format!("template: unexpected character {c:?} in {s:?}")),
            };
            out.push(Tok::Sym(one));
            i += 1;
        }
    }
    Ok(out)
}

/// A recursive-descent parser over one expression's tokens.
struct P {
    t: Vec<Tok>,
    i: usize,
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }

    fn eat_sym(&mut self, s: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(x)) if *x == s) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn eat_word(&mut self, w: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Ident(x)) if x == w) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn or(&mut self) -> Result<Expr, String> {
        let mut l = self.and()?;
        while self.eat_word("or") {
            l = Expr::Or(Box::new(l), Box::new(self.and()?));
        }
        Ok(l)
    }

    fn and(&mut self) -> Result<Expr, String> {
        let mut l = self.not()?;
        while self.eat_word("and") {
            l = Expr::And(Box::new(l), Box::new(self.not()?));
        }
        Ok(l)
    }

    fn not(&mut self) -> Result<Expr, String> {
        if self.eat_word("not") {
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.cmp()
    }

    fn cmp(&mut self) -> Result<Expr, String> {
        let l = self.concat()?;
        if self.eat_sym("==") {
            return Ok(Expr::Eq(Box::new(l), Box::new(self.concat()?), true));
        }
        if self.eat_sym("!=") {
            return Ok(Expr::Eq(Box::new(l), Box::new(self.concat()?), false));
        }
        if self.eat_word("is") {
            let negate = self.eat_word("not");
            if !self.eat_word("defined") {
                return Err("template: only `is defined` and `is not defined` tests are supported".to_string());
            }
            return Ok(Expr::Defined(Box::new(l), !negate));
        }
        Ok(l)
    }

    fn concat(&mut self) -> Result<Expr, String> {
        let mut l = self.postfix()?;
        while self.eat_sym("+") || self.eat_sym("~") {
            l = Expr::Concat(Box::new(l), Box::new(self.postfix()?));
        }
        Ok(l)
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        loop {
            if self.eat_sym(".") {
                match self.t.get(self.i).cloned() {
                    Some(Tok::Ident(name)) => {
                        self.i += 1;
                        e = Expr::Attr(Box::new(e), name);
                    }
                    other => return Err(format!("template: expected a name after `.`, got {other:?}")),
                }
            } else if self.eat_sym("[") {
                let k = self.or()?;
                if !self.eat_sym("]") {
                    return Err("template: expected `]`".to_string());
                }
                e = Expr::Index(Box::new(e), Box::new(k));
            } else if self.eat_sym("|") {
                match self.t.get(self.i).cloned() {
                    Some(Tok::Ident(f)) if ["trim", "tojson", "length"].contains(&f.as_str()) => {
                        self.i += 1;
                        e = Expr::Filter(Box::new(e), f);
                    }
                    other => return Err(format!("template: unsupported filter {other:?}")),
                }
            } else {
                return Ok(e);
            }
        }
    }

    fn primary(&mut self) -> Result<Expr, String> {
        let t = self.t.get(self.i).cloned().ok_or("template: expression ends early")?;
        self.i += 1;
        match t {
            Tok::Str(s) => Ok(Expr::Lit(Value::String(s))),
            Tok::Int(n) => Ok(Expr::Lit(Value::from(n))),
            Tok::Ident(w) => Ok(match w.as_str() {
                "true" | "True" => Expr::Lit(Value::Bool(true)),
                "false" | "False" => Expr::Lit(Value::Bool(false)),
                "none" | "None" => Expr::Lit(Value::Null),
                _ => Expr::Var(w),
            }),
            Tok::Sym("(") => {
                let e = self.or()?;
                if !self.eat_sym(")") {
                    return Err("template: expected `)`".to_string());
                }
                Ok(e)
            }
            Tok::Sym(s) => Err(format!("template: unexpected `{s}`")),
        }
    }
}

/// Parses one whole expression.
fn expr(src: &str) -> Result<Expr, String> {
    let mut p = P { t: lex(src)?, i: 0 };
    let e = p.or()?;
    if p.i != p.t.len() {
        return Err(format!("template: trailing tokens in {src:?}"));
    }
    Ok(e)
}

/// Parses segments until one of `stops` (a statement keyword); returns the
/// nodes and the stop statement met (None at the end of input).
fn block(segs: &[Seg], pos: &mut usize, stops: &[&str]) -> Result<(Vec<Node>, Option<String>), String> {
    let mut nodes = Vec::new();
    while *pos < segs.len() {
        let seg = segs[*pos].clone();
        *pos += 1;
        match seg {
            Seg::Text(t) => nodes.push(Node::Text(t)),
            Seg::Expr(e) => nodes.push(Node::Out(expr(&e)?)),
            Seg::Stmt(s) => {
                let word = s.split_whitespace().next().unwrap_or("").to_string();
                if stops.contains(&word.as_str()) {
                    return Ok((nodes, Some(s)));
                }
                match word.as_str() {
                    "for" => {
                        let rest = s["for".len()..].trim();
                        let (var, iter) = rest.split_once(" in ").ok_or_else(|| format!("template: bad for: {s:?}"))?;
                        let (body, end) = block(segs, pos, &["endfor"])?;
                        if end.is_none() {
                            return Err("template: `for` without `endfor`".to_string());
                        }
                        nodes.push(Node::For { var: var.trim().to_string(), iter: expr(iter.trim())?, body });
                    }
                    "if" => {
                        let mut arms = Vec::new();
                        let mut cond = expr(s["if".len()..].trim())?;
                        let otherwise;
                        loop {
                            let (body, end) = block(segs, pos, &["elif", "else", "endif"])?;
                            arms.push((cond, body));
                            let end = end.ok_or("template: `if` without `endif`")?;
                            if let Some(c) = end.strip_prefix("elif") {
                                cond = expr(c.trim())?;
                                continue;
                            }
                            if end.trim() == "else" {
                                let (body, end2) = block(segs, pos, &["endif"])?;
                                if end2.is_none() {
                                    return Err("template: `else` without `endif`".to_string());
                                }
                                otherwise = body;
                            } else {
                                otherwise = Vec::new();
                            }
                            break;
                        }
                        nodes.push(Node::If { arms, otherwise });
                    }
                    other => return Err(format!("template: unsupported statement `{other}`")),
                }
            }
        }
    }
    Ok((nodes, None))
}

/// Jinja truthiness: false, none, 0, "", [], {} are false.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// How a value prints in `{{ }}`: strings raw, none as "", others as JSON.
fn show(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        other => other.to_string(),
    }
}

/// Variable scopes, innermost last.
struct Env<'a> {
    scopes: Vec<Vec<(String, Value)>>,
    root: &'a Value,
}

impl Env<'_> {
    fn get(&self, name: &str) -> Option<Value> {
        for s in self.scopes.iter().rev() {
            if let Some((_, v)) = s.iter().rev().find(|(k, _)| k == name) {
                return Some(v.clone());
            }
        }
        self.root.get(name).cloned()
    }
}

/// Evaluates; `None` is "undefined" (a missing variable, key, or index).
fn eval(e: &Expr, env: &Env) -> Result<Option<Value>, String> {
    Ok(match e {
        Expr::Lit(v) => Some(v.clone()),
        Expr::Var(n) => env.get(n),
        Expr::Attr(x, name) => eval(x, env)?.and_then(|v| v.get(name).cloned()),
        Expr::Index(x, k) => {
            let (v, k) = (eval(x, env)?, eval(k, env)?);
            match (v, k) {
                (Some(Value::Array(a)), Some(Value::Number(n))) => {
                    let i = n.as_i64().unwrap_or(0);
                    let i = if i < 0 { a.len() as i64 + i } else { i };
                    a.get(i.max(0) as usize).cloned()
                }
                (Some(v @ Value::Object(_)), Some(Value::String(s))) => v.get(&s).cloned(),
                _ => None,
            }
        }
        Expr::Not(x) => Some(Value::Bool(!eval(x, env)?.as_ref().is_some_and(truthy))),
        Expr::And(a, b) => Some(Value::Bool(eval(a, env)?.as_ref().is_some_and(truthy) && eval(b, env)?.as_ref().is_some_and(truthy))),
        Expr::Or(a, b) => Some(Value::Bool(eval(a, env)?.as_ref().is_some_and(truthy) || eval(b, env)?.as_ref().is_some_and(truthy))),
        Expr::Eq(a, b, want) => Some(Value::Bool((eval(a, env)? == eval(b, env)?) == *want)),
        Expr::Concat(a, b) => {
            let s = |v: Option<Value>| v.as_ref().map(show).unwrap_or_default();
            Some(Value::String(s(eval(a, env)?) + &s(eval(b, env)?)))
        }
        Expr::Defined(x, want) => Some(Value::Bool(eval(x, env)?.is_some() == *want)),
        Expr::Filter(x, f) => {
            let v = eval(x, env)?.unwrap_or(Value::Null);
            Some(match f.as_str() {
                "trim" => Value::String(show(&v).trim().to_string()),
                "tojson" => Value::String(v.to_string()),
                "length" => Value::from(match &v {
                    Value::String(s) => s.chars().count(),
                    Value::Array(a) => a.len(),
                    Value::Object(o) => o.len(),
                    _ => 0,
                }),
                other => return Err(format!("template: unsupported filter {other}")),
            })
        }
    })
}

fn render_nodes(nodes: &[Node], env: &mut Env, out: &mut String) -> Result<(), String> {
    for n in nodes {
        match n {
            Node::Text(t) => out.push_str(t),
            Node::Out(e) => out.push_str(&eval(e, env)?.as_ref().map(show).unwrap_or_default()),
            Node::If { arms, otherwise } => {
                let mut done = false;
                for (c, body) in arms {
                    if eval(c, env)?.as_ref().is_some_and(truthy) {
                        render_nodes(body, env, out)?;
                        done = true;
                        break;
                    }
                }
                if !done {
                    render_nodes(otherwise, env, out)?;
                }
            }
            Node::For { var, iter, body } => {
                let items = match eval(iter, env)? {
                    Some(Value::Array(a)) => a,
                    Some(Value::Null) | None => Vec::new(),
                    Some(other) => return Err(format!("template: cannot loop over {other}")),
                };
                let n_items = items.len();
                for (i, item) in items.into_iter().enumerate() {
                    let lp = serde_json::json!({
                        "first": i == 0, "last": i + 1 == n_items, "index": i + 1, "index0": i, "length": n_items
                    });
                    env.scopes.push(vec![(var.clone(), item), ("loop".to_string(), lp)]);
                    let r = render_nodes(body, env, out);
                    env.scopes.pop();
                    r?;
                }
            }
        }
    }
    Ok(())
}

impl Template {
    /// Parses a template; every syntax error is reported here.
    pub fn parse(src: &str) -> Result<Template, String> {
        let segs = segments(src)?;
        let mut pos = 0;
        let (nodes, stop) = block(&segs, &mut pos, &[])?;
        if let Some(s) = stop {
            return Err(format!("template: unexpected `{s}`"));
        }
        Ok(Template { nodes })
    }

    /// Renders with `ctx` (a JSON object of variables).
    pub fn render(&self, ctx: &Value) -> Result<String, String> {
        let mut env = Env { scopes: Vec::new(), root: ctx };
        let mut out = String::new();
        render_nodes(&self.nodes, &mut env, &mut out)?;
        Ok(out)
    }

    /// The prompt of a chat: `messages` (an array of {role, content}),
    /// `add_generation_prompt`, the special-token strings, and `tools`.
    pub fn render_chat(&self, messages: &Value, add_generation_prompt: bool, bos_token: &str, eos_token: &str, tools: Option<&Value>) -> Result<String, String> {
        let ctx = serde_json::json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "tools": tools.cloned().unwrap_or(Value::Null),
        });
        self.render(&ctx)
    }

    /// `render_chat` for (role, content) pairs and no tools.
    pub fn render_messages(&self, messages: &[(&str, &str)], add_generation_prompt: bool, bos_token: &str, eos_token: &str) -> Result<String, String> {
        let m: Vec<Value> = messages.iter().map(|(r, c)| serde_json::json!({"role": r, "content": c})).collect();
        self.render_chat(&Value::Array(m), add_generation_prompt, bos_token, eos_token, None)
    }
}

/// The template used when a model directory ships none (the tracer bigram,
/// the tiny test models): one line per message, `<|role|>` then the content,
/// and `<|assistant|>` as the generation prompt.
pub const DEFAULT_TEMPLATE: &str = "{% for message in messages %}<|{{ message.role }}|>\n{{ message.content }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";
