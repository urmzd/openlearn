<!-- ol:module ag.02 -->
# Tools, registry, JSON Schema argument validation

## Overview

| | |
|---|---|
| **Module** | `ag.02` · build · Go · Pass 10 · 3 h |
| **You build** | `go/agent/tool/tool.go` (`Tool`, `Func`, `New`, `Result`), `go/agent/tool/registry.go` (`Registry`: `Register`, `Definitions`, `Call`), `go/agent/tool/schema.go` (`Compile`, `Schema.ValidateJSON`) |
| **Contract** | the Tool and ToolCall schemas of [`course/contracts/openapi/openai-subset.v1.yaml`](../../course/contracts/openapi/openai-subset.v1.yaml); the Go API is section 4 |
| **Tests** | `course/tests/go/ag_02/` (what they check: section 4) |
| **Needs** | [`ag.01` types and provider](01-types-and-provider.md) (`ToolDef`, `ToolCall`, `Message`) · reading: the JSON-schema subset the engine constrains to (`L8.7`) |
| **Used by** | `ag.03` runs every call through `Registry.Call` · `ag.04` the `query_usage` tool · `ag.05` a reconciled write's `Result` · `ag.08` the `search_docs` tool · `ag.09` agent subjects carry a registry · `ag.10` JSON schema scorer |
| **Milestone** | MS-agent |
| **Optional depth** | [JSON Schema Validation, draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation) (free), sections 6.1 to 6.5 |

## Key Takeaways

- The model's arguments are **untrusted input**: they are validated against the tool's schema before the tool runs, and a call that fails validation never runs (`TestHandWorkedExample`, `TestCallNeverPanics`).
- Every failure (unknown tool, malformed JSON, schema violations, an error or a panic inside the tool) becomes a **tool result the model can read**, prefixed `error: `, so it can correct itself (`TestCallNeverPanics`).
- The validator implements a **small, exact subset** of JSON Schema and refuses any keyword it does not implement, instead of silently ignoring it (`TestCompileRejectsUnsupported`).
- JSON's numbers and strings have sharp edges: `3.0` is an integer, `minimum` is inclusive, length counts characters, and `pattern` matches anywhere (`TestValidateTable`).
- Definitions are sent **sorted by name**, so the prompt is the same bytes on every run (`TestDefinitionsSorted`).

## How to work this chapter

```bash
ol start ag.02
ol tests ag.02
ol check ag.02          # runs ag.01's smoke tests first
ol diff  ag.02
```

---

## 1. Why now

`ag.01` gives you the model's tool call as a name and a byte string of arguments. The model wrote those bytes. It writes `{"days": "3"}` when the tool wants an integer, invents a `country` field, forgets the required `city`, calls `delete_everything` because a document mentioned it, or emits `{"x": "unterminated`. If the tool runs on that, the best case is a panic that kills the agent; the worst is a tool that does something with garbage. The registry is the layer that makes every call either valid or a readable refusal, before any tool code runs.

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

The corrected call `{"city":"Paris","days":3}` passes every rule and runs once, with its arguments unchanged. This is `TestHandWorkedExample`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandWorkedExample` | unit | section 3: the four errors in order, the exact tool message, no run; the valid call runs once with its bytes | you and the tests agree on the messages the model reads |
| `TestValidateTable` | unit | one row per rule: integer and number, unions, inclusive bounds, length in characters, unanchored patterns, enum by value, error paths with indexes, `additionalProperties` as a schema, `anyOf`, malformed JSON | the edges where hand-written validators differ from the spec |
| `TestCompileRejectsUnsupported` | boundary | `format`, an unknown type, a bad pattern, an empty enum, a non-object schema are refused; annotations accepted | no half-enforced schema |
| `TestRegisterRules` | unit | name pattern, 64-character limit, duplicates, non-object parameters, a missing schema means `{"type":"object"}` | providers reject the whole request for one bad tool |
| `TestDefinitionsSorted` | unit | definitions in name order with their descriptions | reproducible prompts |
| `TestCallNeverPanics` | fault | unknown tool, malformed JSON, schema violation, empty arguments, an error, a panic | the loop never dies of a model's mistake |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. numbers judged by their text, `number` rejecting integers, enum comparing text | `3.0` refused as an integer; `1.0` not in `[1]` | `TestValidateTable` (mutants `s01`, `s02`, `s09`) |
| 2. string length in bytes | `"héllo"` too long for `maxLength: 5` | `TestValidateTable` (mutant `s03`) |
| 3. `pattern` anchored by the validator | `"[0-9]{3}"` refuses `"ab123cd"`, which JSON Schema accepts | `TestValidateTable` (mutant `s04`) |
| 4. `minimum` treated as exclusive | `days: 1` refused | `TestValidateTable` (mutant `s05`) |
| 5. `required` or `additionalProperties: false` not enforced | the tool runs without `city`, or with a field it ignores | `TestHandWorkedExample` (mutants `s06`, `s07`) |
| 6. unknown keywords ignored | a schema with `format` or `oneOf` accepts anything | `TestCompileRejectsUnsupported` (mutant `s08`) |
| 7. the tool runs on invalid arguments, an error reads like data, empty arguments rejected | garbage reaches tool code; the model cannot tell failure from output | `TestHandWorkedExample`, `TestCallNeverPanics` (mutants `s10`, `s15`, `s16`) |
| 8. no `recover` around `Execute` | one bad tool call kills the agent process | `TestCallNeverPanics` (mutant `s11`) |
| 9. definitions in map order, duplicates replacing tools, names outside the pattern | a different prompt every run; the provider rejects the request | `TestDefinitionsSorted`, `TestRegisterRules` (mutants `s12`, `s13`, `s14`) |
| 10. array element errors without the index | "expected string" with no way to tell which element | `TestValidateTable` (mutant `s17`) |

## 6. Where it's used next

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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Tool` | saige `RichTool` | multimodal results (images, files) instead of a string | saige `agent/tool` |
| `Registry` | saige MCP bridge | tools served by other processes over the Model Context Protocol | saige `agent/mcp`; [MCP specification](https://modelcontextprotocol.io/) (free) |
| result handling | saige `toolcache` | caching tool results under revision-scoped keys | saige `agent/toolcache` |
| `Schema` | full JSON Schema validators | `$ref`, `oneOf`, `format`, dynamic references | [santhosh-tekuri/jsonschema](https://github.com/santhosh-tekuri/jsonschema) (free) |
