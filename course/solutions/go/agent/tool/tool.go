// Package tool is the agent SDK's tool layer (ag.02): the Tool interface, a
// registry that hands the model its tool definitions, and a JSON Schema
// validator for the arguments the model writes. The rule the package exists
// for: whatever the model sends (an unknown tool, malformed JSON, arguments
// that break the schema, a tool that fails or panics), Call returns a tool
// result the model can read and correct against. It never panics and never
// runs a tool on invalid arguments.
//
// Chapter: content/courses/ai-engineering/28-agents/02-tools-and-schemas.md.
package tool

import (
	"context"
	"encoding/json"

	"tinyllm/agent/types"
)

// Tool is something the model may call. Execute receives arguments that
// already validated against Definition().Parameters.
type Tool interface {
	Definition() types.ToolDef
	Execute(ctx context.Context, args json.RawMessage) (string, error)
}

// Func adapts a function to a Tool.
type Func struct {
	Def types.ToolDef
	Fn  func(ctx context.Context, args json.RawMessage) (string, error)
}

func (f Func) Definition() types.ToolDef { return f.Def }

func (f Func) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	return f.Fn(ctx, args)
}

// New is a Func from its parts; schema is a JSON Schema object.
func New(name, description, schema string, fn func(ctx context.Context, args json.RawMessage) (string, error)) Tool {
	return Func{Def: types.ToolDef{Name: name, Description: description, Parameters: json.RawMessage(schema)}, Fn: fn}
}

// Result is the outcome of one call, ready to go back to the model.
type Result struct {
	CallID  string
	Name    string
	Content string
	IsError bool
}

// Message is the tool message answering the call. An error result's content
// starts with "error: " so the model can tell a failure from data.
func (r Result) Message() types.Message {
	content := r.Content
	if r.IsError {
		content = "error: " + content
	}
	return types.Message{Role: types.RoleTool, ToolCallID: r.CallID, Name: r.Name, Content: content}
}
