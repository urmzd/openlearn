# Tools, registry, JSON Schema argument validation

## Overview

| | |
|---|---|
| **Module** | `ag.02` · build · Go · Pass 10 · 3 h |
| **Solution** | [`tool.go`](../../../../course/solutions/go/agent/tool/tool.go) (`Tool`, `Func`, `New`, `Result`), [`registry.go`](../../../../course/solutions/go/agent/tool/registry.go) (`Registry`: `Register`, `Definitions`, `Call`), [`schema.go`](../../../../course/solutions/go/agent/tool/schema.go) (`Compile`, `Schema.ValidateJSON`) |
| **Contract** | the Tool and ToolCall schemas of [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml); the Go API is section 4 |
| **Needs** | [`ag.01` types and provider](01-types-and-provider.md) (`ToolDef`, `ToolCall`, `Message`) · reading: the JSON-schema subset the engine constrains to (`L8.7`) |
| **Used by** | `ag.03` runs every call through `Registry.Call` · `ag.04` the `query_usage` tool · `ag.05` a reconciled write's `Result` · `ag.08` the `search_docs` tool · `ag.09` agent subjects carry a registry · `ag.10` JSON schema scorer |
| **Optional depth** | [JSON Schema Validation, draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation) (free), sections 6.1 to 6.5 |

## Key Takeaways

- The model's arguments are **untrusted input**: they are validated against the tool's schema before the tool runs, and a call that fails validation never runs.
- Every failure (unknown tool, malformed JSON, schema violations, an error or a panic inside the tool) becomes a **tool result the model can read**, prefixed `error: `, so it can correct itself.
- The validator implements a **small, exact subset** of JSON Schema and refuses any keyword it does not implement, instead of silently ignoring it.
- JSON's numbers and strings have sharp edges: `3.0` is an integer, `minimum` is inclusive, length counts characters, and `pattern` matches anywhere.
- Definitions are sent **sorted by name**, so the prompt is the same bytes on every run.

## How to read this chapter

Sections 1 to 3 build the registry and the validator from first principles and validate one bad call by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each choice is the right one; try to predict each step before reading it.

---

## 1. Why now

`ag.01` gives the model's tool call as a name and a byte string of arguments. The model wrote those bytes. It writes `{"days": "3"}` when the tool wants an integer, invents a `country` field, forgets the required `city`, calls `delete_everything` because a document mentioned it, or emits `{"x": "unterminated`. If the tool runs on that, the best case is a panic that kills the agent; the worst is a tool that does something with garbage. The registry is the layer that makes every call either valid or a readable refusal, before any tool code runs.

## 2. Principles

### 2.1 A tool is a contract with the model

A tool has a `Definition()`: a name (the contract allows `^[A-Za-z0-9_-]{1,64}$`), a description the model reads to decide when to use it, and `Parameters`, a JSON Schema whose top-level type is `object` (the model always sends an object). `Execute(ctx, args)` receives arguments that already passed that schema and returns a string, the tool's answer in the model's context.

### 2.2 Validation, rule by rule

A JSON Schema is a JSON object of **keywords**, each a constraint on a value. Two characters carry meaning below:

| Symbol | Meaning |
|---|---|
| `$` (in an error path) | the whole argument value; `$.days` is its property `days`, `$.tags[1]` the second element of `tags` |
| `^`, `$` (in a pattern) | the start and the end of the string; without them a pattern may match anywhere |

The supported subset:

| Keyword | Applies to | Passes when |
|---|---|---|
| `type` | any | the value's JSON type is the named one, or one of a list; `integer` is a number with no fractional part (`3.0` is an integer), and `number` accepts integers |
| `enum`, `const` | any | the value equals one of the listed values; numbers compare by value (`1` equals `1.0`) |
| `properties` | object | each named property present validates against its schema |
| `required` | object | each listed property is present |
| `additionalProperties` | object | `false`: no property outside `properties`; a schema: every other property validates against it |
| `items`, `minItems`, `maxItems` | array | every element validates; the length bounds hold |
| `minLength`, `maxLength` | string | the length in **characters** (Unicode code points) is within bounds: `"héllo"` has 5, though 6 bytes |
| `pattern` | string | the regular expression matches **somewhere** in the string; anchor it with `^...$` to match the whole string |
| `minimum`, `maximum` | number | inclusive bounds |
| `anyOf` | any | at least one subschema passes |

`title`, `description`, `default`, `examples`, and `$schema` are annotations. Any other keyword (`format`, `oneOf`, `$ref`, ...) is a compile error: the validator would otherwise accept a schema that promises more than it checks, and `"format": "email"` would let anything through.

A violation has a **path**: `$` for the whole value, `.name` for a property, `[i]` for an array element. When the type is wrong the other keywords of that schema are not checked (they would only repeat the mismatch). Errors are sorted by path, then message, so the same call always gets the same text.

### 2.3 The registry's promise

`Call` never panics and always returns a `Result`. An unknown name lists the names that exist (the model can retry with a real one). Empty arguments mean `{}` (some providers send `""` for a call without parameters). A panic inside `Execute` is recovered; the reply says the tool panicked but never carries the stack trace, which is for logs, not for the model's context. `Result.Message()` is the tool message that answers the call: role `tool`, the call's id, and the content, prefixed `error: ` for a failure.

`Definitions()` lists the tools sorted by name. The tool list is part of the prompt; a list in map order changes the prompt's bytes from run to run, which makes runs irreproducible and defeats prompt caches.

## 3. Worked example by hand

The tool `get_weather` has this schema:

```json
{"type": "object",
 "properties": {"city": {"type": "string", "minLength": 1},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
                "unit": {"enum": ["c", "f"]}},
 "required": ["city"], "additionalProperties": false}
```

The model sends `{"days": "3", "unit": "kelvin", "country": "FR"}`. Walk the object:

| Rule | Value | Violation |
|---|---|---|
| `required: ["city"]` | no `city` | `$: missing required property "city"` |
| `additionalProperties: false` | `country` is not in `properties` | `$: unexpected property "country"` |
| `days`: `type: integer` | `"3"` is a string | `$.days: expected integer, got string` (minimum and maximum are skipped) |
| `unit`: `enum` | `"kelvin"` | `$.unit: "kelvin" is not one of ["c","f"]` |

Sorted by path then message, the two `$` errors come first ("m" before "u"). The tool does not run; the model reads:

```
error: invalid arguments for get_weather:
$: missing required property "city"
$: unexpected property "country"
$.days: expected integer, got string
$.unit: "kelvin" is not one of ["c","f"]
```

The corrected call `{"city":"Paris","days":3}` passes every rule and runs once, with its arguments unchanged.

## 4. The interface

```go
package tool // import "tinyllm/agent/tool"

type Tool interface {
	Definition() types.ToolDef
	Execute(ctx context.Context, args json.RawMessage) (string, error)
}
type Func struct { Def types.ToolDef; Fn func(context.Context, json.RawMessage) (string, error) }
func New(name, description, schema string, fn func(context.Context, json.RawMessage) (string, error)) Tool
type Result struct { CallID, Name, Content string; IsError bool }
func (r Result) Message() types.Message

var NamePattern *regexp.Regexp
func NewRegistry() *Registry
func (r *Registry) Register(t Tool) error
func (r *Registry) MustRegister(ts ...Tool) *Registry
func (r *Registry) Get(name string) (Tool, bool)
func (r *Registry) Names() []string
func (r *Registry) Definitions() []types.ToolDef
func (r *Registry) Call(ctx context.Context, c types.ToolCall) Result

func Compile(raw json.RawMessage) (*Schema, error)
func (s *Schema) ValidateJSON(raw []byte) []ValidationError // ValidationError{Path, Msg}; Error() is "Path: Msg"
```

Use only the standard library (`regexp` is RE2; JSON Schema's patterns are ECMA-262, and the subset both accept is what tools need). The registry must be safe for concurrent use: the loop runs calls in parallel.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example (section 3) | unit | the four errors in order, the exact tool message, no run; the valid call runs once with its bytes | the messages the model reads are pinned down |
| one row per rule | unit | integer and number, type unions, inclusive bounds, length in characters, unanchored patterns, enum by value, error paths with indexes, `additionalProperties` as a schema, `anyOf`, malformed JSON | the edges where hand-written validators differ from the spec |
| unsupported schemas | boundary | `format`, an unknown type, a bad pattern, an empty enum, a non-object schema are refused; annotations accepted | no half-enforced schema |
| registration | unit | name pattern, 64-character limit, duplicates, non-object parameters; a missing schema means `{"type":"object"}` | providers reject the whole request for one bad tool |
| definitions | unit | in name order with their descriptions | reproducible prompts |
| every failure mode of `Call` | fault | unknown tool, malformed JSON, schema violation, empty arguments, an error, a panic | the loop never dies of a model's mistake |

## 5. The solution, step by step

### 5.1 From first principles

1. **Validate before execute, always.** The arguments are written by a model that may be confused or manipulated. The only safe order is: find the tool, parse, validate, and only then call `Execute`, so tool code can assume its schema holds.
2. **A failure is data for the model, not an exception for the program.** The model can fix a call it can read about. So every failure path returns a `Result` with `IsError`, and `Message()` prefixes `error: ` so the model can tell a refusal from output.
3. **Numbers must be compared as numbers, not as Go types or text.** JSON has one number type. Decoding with `UseNumber` keeps `3.0` and `3` exact, and the schema decides: an integer is a number with no fractional part, `number` accepts integers, and `enum` compares by value.
4. **A string's length is in characters.** JSON Schema counts code points, so `utf8.RuneCountInString`, not `len`.
5. **Keywords the validator does not implement must be refused at compile time.** Ignoring `format` or `oneOf` would make a schema look enforced while accepting anything.
6. **Same input, same bytes out.** Errors are sorted by path then message, object keys are walked in sorted order, and definitions are sorted by name, so the model's context (and a prompt cache key) is reproducible.
7. **One bad tool must not take down the process.** A `recover` around `Execute` turns a panic into an error result, keeping the stack out of the model's context.

### 5.2 The code, annotated

**Registering** (`registry.go`). Everything that can be checked once is checked at registration, so `Call` only deals with the model's input:

```go
func (r *Registry) Register(t Tool) error {
	d := t.Definition()
	if !NamePattern.MatchString(d.Name) { // ^[A-Za-z0-9_-]{1,64}$: providers reject the request otherwise
		return fmt.Errorf("tool: name %q does not match %s", d.Name, NamePattern)
	}
	params := d.Parameters
	if len(params) == 0 {
		params = json.RawMessage(`{"type":"object"}`) // a tool with no parameters
	}
	s, err := Compile(params) // unsupported keywords fail here, at startup
	if err != nil {
		return fmt.Errorf("tool %s: %w", d.Name, err)
	}
	if len(s.types) != 1 || s.types[0] != "object" {
		return fmt.Errorf("tool %s: parameters must have type \"object\"", d.Name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, dup := r.tools[d.Name]; dup {
		return fmt.Errorf("tool: %q is already registered", d.Name) // never silently replace
	}
	r.tools[d.Name] = t
	r.schemas[d.Name] = s // compiled once, reused for every call
	return nil
}
```

**Calling.** The order of checks is step 1; the deferred `recover` is step 7, and it is installed only after validation, so it covers exactly the tool's own code:

```go
func (r *Registry) Call(ctx context.Context, c types.ToolCall) (res Result) {
	res = Result{CallID: c.ID, Name: c.Name}
	fail := func(msg string) Result {
		res.Content, res.IsError = msg, true
		return res
	}
	t, ok := r.Get(c.Name)
	if !ok {
		return fail(fmt.Sprintf("unknown tool %q; available: %s", c.Name, strings.Join(r.Names(), ", ")))
	}
	// ... s := the compiled schema, under the read lock
	args := c.Args
	if len(strings.TrimSpace(string(args))) == 0 {
		args = json.RawMessage("{}") // some providers send "" for a call without parameters
	}
	if errs := s.ValidateJSON(args); len(errs) > 0 {
		// ... one "Path: Msg" line per error
		return fail(fmt.Sprintf("invalid arguments for %s:\n%s", c.Name, strings.Join(lines, "\n")))
	}
	defer func() {
		if p := recover(); p != nil {
			_ = debug.Stack() // for a caller that logs; never returned to the model
			res = Result{CallID: c.ID, Name: c.Name, Content: fmt.Sprintf("tool %s panicked: %v", c.Name, p), IsError: true}
		}
	}()
	out, err := t.Execute(ctx, args) // args unchanged: the bytes the model sent
	if err != nil {
		return fail(err.Error())
	}
	res.Content = out
	return res
}
```

The named return `res` is what lets the deferred function replace the result after a panic.

**Compiling** (`schema.go`). `compile` walks the keywords in sorted order and builds a `Schema` struct with one field per supported keyword; the `default` branch is step 5:

```go
default:
	if !annotations[k] { // title, description, default, examples, $schema
		return nil, fmt.Errorf("schema %s: unsupported keyword %q", at, k)
	}
```

`enum` and `const` share one field (`const` is a one-element enum), and values are decoded with `UseNumber` so they compare exactly later.

**Types**, step 3. A decoded number is a `json.Number`; whether it is an integer is a property of its value:

```go
case json.Number:
	f, err := strconv.ParseFloat(string(x), 64)
	if err == nil && f == math.Trunc(f) && !math.IsInf(f, 0) {
		return "integer" // 3 and 3.0 alike
	}
	return "number"
```

**Validating.** The type check comes first and stops the schema's other checks when it fails, which is why section 3 shows one error for `days`, not three:

```go
got := typeOf(v)
if len(s.types) > 0 {
	ok := false
	for _, t := range s.types {
		if t == got || (t == "number" && got == "integer") { // number accepts integers
			ok = true
		}
	}
	if !ok {
		add(path, "expected %s, got %s", strings.Join(s.types, " or "), got)
		return errs // the other keywords would only repeat the mismatch
	}
}
```

Then, by the value's kind:

```go
case map[string]any:
	for _, r := range s.required {
		if _, ok := x[r]; !ok {
			add(path, "missing required property %q", r) // reported at the object's own path
		}
	}
	// ... names = the object's keys, sorted (step 6)
	for _, k := range names {
		p := path + "." + k
		if ps, ok := s.props[k]; ok {
			errs = append(errs, ps.validate(x[k], p)...)
		} else if s.addlFalse {
			add(path, "unexpected property %q", k)
		} else if s.addl != nil {
			errs = append(errs, s.addl.validate(x[k], p)...) // additionalProperties as a schema
		}
	}
case []any:
	// ... minItems, maxItems; each element at path[i], so the index is in the message
case string:
	n := utf8.RuneCountInString(x) // characters, not bytes (step 4)
	// ... minLength, maxLength
	if s.pattern != nil && !s.pattern.MatchString(x) { // MatchString: anywhere in the string
		add(path, "%q does not match pattern %q", x, s.patternSrc)
	}
case json.Number:
	f, _ := strconv.ParseFloat(string(x), 64)
	if s.minimum != nil && f < *s.minimum { // inclusive: only strictly below fails
		add(path, "%s is less than minimum %v", x, *s.minimum)
	}
	// ... maximum likewise
```

`anyOf` passes when any subschema returns no errors. `equalJSON` compares numbers by parsed value (`1` equals `1.0`) and recurses into arrays and objects. `ValidateJSON` decodes (malformed JSON is one error at `$`), validates, and sorts by path then message: `"$"` sorts before `"$.days"`, and within `$`, `missing...` before `unexpected...`, the order in section 3.

**The message** (`tool.go`):

```go
func (r Result) Message() types.Message {
	content := r.Content
	if r.IsError {
		content = "error: " + content // the model can tell a failure from data
	}
	return types.Message{Role: types.RoleTool, ToolCallID: r.CallID, Name: r.Name, Content: content}
}
```

Full solution: [`tool.go`](../../../../course/solutions/go/agent/tool/tool.go), [`registry.go`](../../../../course/solutions/go/agent/tool/registry.go), [`schema.go`](../../../../course/solutions/go/agent/tool/schema.go).

### 5.3 Why this solution

- **Cost.** Schemas are compiled once at registration; a call costs one JSON decode and one walk of the value, $O(\text{size of the arguments})$, plus a sort of the (usually empty) error list. Regular expressions are RE2, so a pattern cannot backtrack catastrophically on hostile input.
- **A small exact subset** rather than a full JSON Schema library: the subset is what tool schemas use, it fits in one file a reader can audit, and refusing everything else keeps the guarantee honest. A full validator brings `$ref` and `format` and a dependency.
- **All errors at once** rather than the first: the model can fix every problem in one retry instead of one per round trip.
- **Error results rather than Go errors** from `Call`: the loop does not need a separate path for model mistakes; it appends the tool message and lets the model react.
- **Tradeoff accepted:** patterns are RE2, not ECMA-262. Lookaheads and backreferences fail at compile time, which is the safe direction.

### 5.4 Explain it back

1. Why does `days: "3"` produce one error and not also "less than minimum"?
   <details><summary>Answer</summary>When the type check fails, `validate` returns immediately. `minimum` is defined only for numbers, so checking it on a string would only restate the type mismatch; one precise error is more useful to the model.</details>
2. Why is `3.0` accepted for `type: integer`, and how does the code know?
   <details><summary>Answer</summary>JSON Schema defines an integer as a number with zero fractional part, not as a lexical form. The decoder keeps the number as `json.Number`; `typeOf` parses it and checks `f == math.Trunc(f)`, so `3.0` is an integer and `3.5` is not.</details>
3. A schema has `"format": "email"`. What happens, and why is that better than ignoring it?
   <details><summary>Answer</summary>`Register` fails: `compile` reports `unsupported keyword "format"`. Ignoring it would accept any string while the schema's author believes emails are enforced; failing at startup forces the author to drop the keyword or enforce it in the tool.</details>
4. Why is the `recover` deferred after validation instead of at the top of `Call`?
   <details><summary>Answer</summary>Everything before it is registry code that is expected not to panic; the `recover` is there for tool code, which the registry does not control. Placing it right before `Execute` makes its scope exactly the untrusted part, and its replacement result is tagged as a tool panic.</details>
5. Why do the definitions have to be sorted, if the model does not care about order?
   <details><summary>Answer</summary>The definitions are part of the prompt's bytes. Go map order is random, so unsorted definitions would make each run's prompt different, defeating prompt caches and making recorded runs irreproducible, even though the model's behavior might not change.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. numbers judged by their text, `number` rejecting integers, enum comparing text | `3.0` refused as an integer; `1.0` not in `[1]` | `typeOf` checks the value's fraction; `number` accepts `integer`; `equalJSON` compares parsed values |
| 2. string length in bytes | `"héllo"` too long for `maxLength: 5` | `utf8.RuneCountInString` |
| 3. `pattern` anchored by the validator | `"[0-9]{3}"` refuses `"ab123cd"`, which JSON Schema accepts | `pattern.MatchString`, which matches anywhere; schema authors anchor with `^...$` |
| 4. `minimum` treated as exclusive | `days: 1` refused | only `f < minimum` is an error |
| 5. `required` or `additionalProperties: false` not enforced | the tool runs without `city`, or with a field it ignores | both checked on every object; `Call` returns before `Execute` on any error |
| 6. unknown keywords ignored | a schema with `format` or `oneOf` accepts anything | `compile` refuses any keyword that is not supported or an annotation |
| 7. the tool runs on invalid arguments, an error reads like data, empty arguments rejected | garbage reaches tool code; the model cannot tell failure from output | validation before `Execute`; `error: ` prefix in `Message`; empty arguments become `{}` |
| 8. no `recover` around `Execute` | one bad tool call kills the agent process | the deferred `recover` replaces the result with "tool ... panicked" |
| 9. definitions in map order, duplicates replacing tools, names outside the pattern | a different prompt every run; the provider rejects the request | `Definitions` sorts by name; `Register` refuses duplicates and bad names |
| 10. array element errors without the index | "expected string" with no way to tell which element | elements are validated at `path[i]` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | `ToolDef`, `ToolCall`, and `Message` are the registry's inputs and outputs |
| Forward | `ag.03` | the loop sends `Definitions()` with every model call and runs each call through `Call` |
| Forward | `ag.04` | `query_usage` is a `tool.Tool` whose schema the registry enforces |
| Forward | `ag.05` | a reconciled write is recorded as a `Result` the loop replays |
| Forward | `ag.08` | `search_docs` is a `tool.Tool` in the same registry |
| Forward | `ag.09` | the evaluation suites build the agents they score with a registry |
| Forward | `ag.10` | schema scorers validate JSON outputs against the same subset |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Tool` | saige `RichTool` | multimodal results (images, files) instead of a string | saige `agent/tool` |
| `Registry` | saige MCP bridge | tools served by other processes over the Model Context Protocol | saige `agent/mcp`; [MCP specification](https://modelcontextprotocol.io/) (free) |
| result handling | saige `toolcache` | caching tool results under revision-scoped keys | saige `agent/toolcache` |
| `Schema` | full JSON Schema validators | `$ref`, `oneOf`, `format`, dynamic references | [santhosh-tekuri/jsonschema](https://github.com/santhosh-tekuri/jsonschema) (free) |
