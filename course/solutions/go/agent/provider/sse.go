package provider

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"

	"tinyllm/agent/types"
)

// ErrNoDone is a stream that ended (EOF) before `data: [DONE]`: the answer
// may be cut short, so it is an error, never a quiet success.
var ErrNoDone = errors.New("provider: stream ended before [DONE]")

// StreamError is an error event sent after streaming began
// (`data: {"error": {...}}`, openai-subset.v1 Streaming).
type StreamError struct{ Message, Type, Code string }

func (e *StreamError) Error() string {
	return fmt.Sprintf("provider: stream error (%s): %s", e.Type, e.Message)
}

// EventReader reads server-sent events. Next returns the data of the next
// event: its `data:` lines (one leading space removed) joined with "\n".
// Comment lines (starting with ':') and other fields are skipped; an event
// with no data line is skipped. Lines may end in "\n" or "\r\n".
type EventReader struct {
	br *bufio.Reader
}

func NewEventReader(r io.Reader) *EventReader {
	return &EventReader{br: bufio.NewReaderSize(r, 32<<10)}
}

// Next returns io.EOF at a clean end (no partial event pending) and
// io.ErrUnexpectedEOF when the input ends inside an event.
func (e *EventReader) Next() ([]byte, error) {
	var data [][]byte
	pending := false
	for {
		line, err := e.br.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				if pending || len(line) > 0 {
					return nil, io.ErrUnexpectedEOF
				}
				return nil, io.EOF
			}
			return nil, err
		}
		line = bytes.TrimSuffix(bytes.TrimSuffix(line, []byte("\n")), []byte("\r"))
		if len(line) == 0 {
			if len(data) > 0 {
				return bytes.Join(data, []byte("\n")), nil
			}
			pending = false
			continue
		}
		pending = true
		if line[0] == ':' {
			continue
		}
		if v, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			v, _ = bytes.CutPrefix(v, []byte(" "))
			data = append(data, append([]byte(nil), v...))
		}
	}
}

// chunk is the part of a chat.completion.chunk the decoder reads.
type chunk struct {
	Choices []struct {
		Delta struct {
			Content   *string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Code    *string `json:"code"`
	} `json:"error"`
}

// Decoder turns event data into deltas. It keeps the tool calls seen so far,
// so fragments of interleaved calls are assembled by index.
type Decoder struct {
	calls  map[int]*types.ToolCall
	ended  map[int]bool
	finish string
}

func NewDecoder() *Decoder {
	return &Decoder{calls: map[int]*types.ToolCall{}, ended: map[int]bool{}}
}

// Event decodes one event's data. done is true after `[DONE]` or an error
// event: the returned deltas then end with a DoneDelta or an ErrorDelta.
func (d *Decoder) Event(data []byte) (out []types.Delta, done bool) {
	if string(bytes.TrimSpace(data)) == "[DONE]" {
		out = append(out, d.endCalls()...)
		return append(out, types.DoneDelta{FinishReason: d.finish}), true
	}
	var c chunk
	if err := json.Unmarshal(data, &c); err != nil {
		return []types.Delta{types.ErrorDelta{Err: fmt.Errorf("provider: bad chunk: %w", err)}}, true
	}
	if c.Error != nil {
		se := &StreamError{Message: c.Error.Message, Type: c.Error.Type}
		if c.Error.Code != nil {
			se.Code = *c.Error.Code
		}
		return []types.Delta{types.ErrorDelta{Err: se}}, true
	}
	for _, ch := range c.Choices {
		if ch.Delta.Content != nil && *ch.Delta.Content != "" {
			out = append(out, types.TextDelta{Text: *ch.Delta.Content})
		}
		for _, tc := range ch.Delta.ToolCalls {
			call, ok := d.calls[tc.Index]
			if !ok {
				call = &types.ToolCall{ID: tc.ID, Name: tc.Function.Name}
				d.calls[tc.Index] = call
				out = append(out, types.ToolCallStartDelta{Index: tc.Index, ID: tc.ID, Name: tc.Function.Name})
			}
			if tc.Function.Arguments != "" {
				call.Args = append(call.Args, tc.Function.Arguments...)
				out = append(out, types.ToolCallArgsDelta{Index: tc.Index, Fragment: tc.Function.Arguments})
			}
		}
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			d.finish = *ch.FinishReason
			out = append(out, d.endCalls()...)
		}
	}
	if c.Usage != nil {
		out = append(out, types.UsageDelta{In: c.Usage.PromptTokens, Out: c.Usage.CompletionTokens})
	}
	return out, false
}

// endCalls closes every open call, in index order.
func (d *Decoder) endCalls() []types.Delta {
	idx := make([]int, 0, len(d.calls))
	for i := range d.calls {
		if !d.ended[i] {
			idx = append(idx, i)
		}
	}
	sort.Ints(idx)
	out := make([]types.Delta, 0, len(idx))
	for _, i := range idx {
		d.ended[i] = true
		c := *d.calls[i]
		if len(c.Args) == 0 {
			c.Args = []byte("{}")
		}
		c.Args = append([]byte(nil), c.Args...)
		out = append(out, types.ToolCallEndDelta{Index: i, Call: c})
	}
	return out
}
