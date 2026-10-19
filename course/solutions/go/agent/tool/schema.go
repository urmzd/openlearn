package tool

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Schema is a compiled JSON Schema in the subset tools use:
//
//	type                 a name or a list: object array string integer number boolean null
//	properties, required, additionalProperties (true, false, or a schema)
//	items, minItems, maxItems
//	enum, const
//	minLength, maxLength (counted in characters), pattern (unanchored, RE2)
//	minimum, maximum
//	anyOf
//
// title, description, default, examples, and $schema are annotations and
// are ignored. Any other keyword is a compile error: a schema that says more
// than the validator can check must not be half enforced.
type Schema struct {
	types      []string
	props      map[string]*Schema
	required   []string
	addl       *Schema // nil: any extra property allowed
	addlFalse  bool
	items      *Schema
	minItems   *int
	maxItems   *int
	enum       []any
	minLength  *int
	maxLength  *int
	pattern    *regexp.Regexp
	patternSrc string
	minimum    *float64
	maximum    *float64
	anyOf      []*Schema
}

// ValidationError is one violation: Path is "$" for the whole value, then
// ".name" for a property and "[i]" for an array element.
type ValidationError struct {
	Path string
	Msg  string
}

func (e ValidationError) Error() string { return e.Path + ": " + e.Msg }

var annotations = map[string]bool{"title": true, "description": true, "default": true, "examples": true, "$schema": true}

var typeNames = map[string]bool{"object": true, "array": true, "string": true, "integer": true, "number": true, "boolean": true, "null": true}

// Compile parses raw into a Schema.
func Compile(raw json.RawMessage) (*Schema, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("schema: not a JSON object: %w", err)
	}
	return compile(m, "$")
}

func compile(m map[string]json.RawMessage, at string) (*Schema, error) {
	s := &Schema{}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	intField := func(k string) (*int, error) {
		var n int
		if err := json.Unmarshal(m[k], &n); err != nil || n < 0 {
			return nil, fmt.Errorf("schema %s: %s must be a non-negative integer", at, k)
		}
		return &n, nil
	}
	numField := func(k string) (*float64, error) {
		var f float64
		if err := json.Unmarshal(m[k], &f); err != nil {
			return nil, fmt.Errorf("schema %s: %s must be a number", at, k)
		}
		return &f, nil
	}
	sub := func(raw json.RawMessage, where string) (*Schema, error) {
		var mm map[string]json.RawMessage
		if err := json.Unmarshal(raw, &mm); err != nil {
			return nil, fmt.Errorf("schema %s: not an object", where)
		}
		return compile(mm, where)
	}
	var err error
	for _, k := range keys {
		v := m[k]
		switch k {
		case "type":
			var one string
			if json.Unmarshal(v, &one) == nil {
				s.types = []string{one}
			} else if err := json.Unmarshal(v, &s.types); err != nil {
				return nil, fmt.Errorf("schema %s: type must be a name or a list of names", at)
			}
			for _, t := range s.types {
				if !typeNames[t] {
					return nil, fmt.Errorf("schema %s: unknown type %q", at, t)
				}
			}
		case "properties":
			var ps map[string]json.RawMessage
			if err := json.Unmarshal(v, &ps); err != nil {
				return nil, fmt.Errorf("schema %s: properties must be an object", at)
			}
			s.props = map[string]*Schema{}
			for name, raw := range ps {
				if s.props[name], err = sub(raw, at+"."+name); err != nil {
					return nil, err
				}
			}
		case "required":
			if err := json.Unmarshal(v, &s.required); err != nil {
				return nil, fmt.Errorf("schema %s: required must be a list of names", at)
			}
		case "additionalProperties":
			var b bool
			if json.Unmarshal(v, &b) == nil {
				s.addlFalse = !b
			} else if s.addl, err = sub(v, at+".*"); err != nil {
				return nil, err
			}
		case "items":
			if s.items, err = sub(v, at+"[]"); err != nil {
				return nil, err
			}
		case "minItems":
			if s.minItems, err = intField(k); err != nil {
				return nil, err
			}
		case "maxItems":
			if s.maxItems, err = intField(k); err != nil {
				return nil, err
			}
		case "minLength":
			if s.minLength, err = intField(k); err != nil {
				return nil, err
			}
		case "maxLength":
			if s.maxLength, err = intField(k); err != nil {
				return nil, err
			}
		case "minimum":
			if s.minimum, err = numField(k); err != nil {
				return nil, err
			}
		case "maximum":
			if s.maximum, err = numField(k); err != nil {
				return nil, err
			}
		case "enum":
			if s.enum, err = decodeList(v); err != nil || len(s.enum) == 0 {
				return nil, fmt.Errorf("schema %s: enum must be a non-empty list", at)
			}
		case "const":
			c, err := decode(v)
			if err != nil {
				return nil, fmt.Errorf("schema %s: const: %w", at, err)
			}
			s.enum = []any{c}
		case "pattern":
			if err := json.Unmarshal(v, &s.patternSrc); err != nil {
				return nil, fmt.Errorf("schema %s: pattern must be a string", at)
			}
			if s.pattern, err = regexp.Compile(s.patternSrc); err != nil {
				return nil, fmt.Errorf("schema %s: pattern: %w", at, err)
			}
		case "anyOf":
			var list []json.RawMessage
			if err := json.Unmarshal(v, &list); err != nil || len(list) == 0 {
				return nil, fmt.Errorf("schema %s: anyOf must be a non-empty list", at)
			}
			for i, raw := range list {
				one, err := sub(raw, fmt.Sprintf("%s.anyOf[%d]", at, i))
				if err != nil {
					return nil, err
				}
				s.anyOf = append(s.anyOf, one)
			}
		default:
			if !annotations[k] {
				return nil, fmt.Errorf("schema %s: unsupported keyword %q", at, k)
			}
		}
	}
	return s, nil
}

// decode parses JSON keeping numbers exact (json.Number).
func decode(raw []byte) (any, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var v any
	if err := d.Decode(&v); err != nil {
		return nil, err
	}
	if d.More() {
		return nil, fmt.Errorf("trailing data after the JSON value")
	}
	return v, nil
}

func decodeList(raw []byte) ([]any, error) {
	v, err := decode(raw)
	if err != nil {
		return nil, err
	}
	l, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("not a list")
	}
	return l, nil
}

// ValidateJSON parses raw and validates it. Malformed JSON is one error at
// "$". The errors are sorted by path, then message.
func (s *Schema) ValidateJSON(raw []byte) []ValidationError {
	v, err := decode(raw)
	if err != nil {
		return []ValidationError{{Path: "$", Msg: "invalid JSON: " + err.Error()}}
	}
	errs := s.validate(v, "$")
	sort.SliceStable(errs, func(i, j int) bool {
		if errs[i].Path != errs[j].Path {
			return errs[i].Path < errs[j].Path
		}
		return errs[i].Msg < errs[j].Msg
	})
	return errs
}

// typeOf names a decoded value's JSON type ("integer" for whole numbers).
func typeOf(v any) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	case json.Number:
		f, err := strconv.ParseFloat(string(x), 64)
		if err == nil && f == math.Trunc(f) && !math.IsInf(f, 0) {
			return "integer"
		}
		return "number"
	}
	return fmt.Sprintf("%T", v)
}

func (s *Schema) validate(v any, path string) []ValidationError {
	var errs []ValidationError
	add := func(p, f string, a ...any) { errs = append(errs, ValidationError{Path: p, Msg: fmt.Sprintf(f, a...)}) }
	got := typeOf(v)
	if len(s.types) > 0 {
		ok := false
		for _, t := range s.types {
			if t == got || (t == "number" && got == "integer") {
				ok = true
			}
		}
		if !ok {
			add(path, "expected %s, got %s", strings.Join(s.types, " or "), got)
			return errs // the other keywords would only repeat the mismatch
		}
	}
	if len(s.enum) > 0 {
		found := false
		for _, e := range s.enum {
			if equalJSON(e, v) {
				found = true
				break
			}
		}
		if !found {
			want, _ := json.Marshal(s.enum)
			have, _ := json.Marshal(v)
			add(path, "%s is not one of %s", have, want)
		}
	}
	switch x := v.(type) {
	case map[string]any:
		for _, r := range s.required {
			if _, ok := x[r]; !ok {
				add(path, "missing required property %q", r)
			}
		}
		names := make([]string, 0, len(x))
		for k := range x {
			names = append(names, k)
		}
		sort.Strings(names)
		for _, k := range names {
			p := path + "." + k
			if ps, ok := s.props[k]; ok {
				errs = append(errs, ps.validate(x[k], p)...)
			} else if s.addlFalse {
				add(path, "unexpected property %q", k)
			} else if s.addl != nil {
				errs = append(errs, s.addl.validate(x[k], p)...)
			}
		}
	case []any:
		if s.minItems != nil && len(x) < *s.minItems {
			add(path, "has %d items, fewer than minItems %d", len(x), *s.minItems)
		}
		if s.maxItems != nil && len(x) > *s.maxItems {
			add(path, "has %d items, more than maxItems %d", len(x), *s.maxItems)
		}
		if s.items != nil {
			for i, e := range x {
				errs = append(errs, s.items.validate(e, fmt.Sprintf("%s[%d]", path, i))...)
			}
		}
	case string:
		n := utf8.RuneCountInString(x)
		if s.minLength != nil && n < *s.minLength {
			add(path, "has %d characters, fewer than minLength %d", n, *s.minLength)
		}
		if s.maxLength != nil && n > *s.maxLength {
			add(path, "has %d characters, more than maxLength %d", n, *s.maxLength)
		}
		if s.pattern != nil && !s.pattern.MatchString(x) {
			add(path, "%q does not match pattern %q", x, s.patternSrc)
		}
	case json.Number:
		f, _ := strconv.ParseFloat(string(x), 64)
		if s.minimum != nil && f < *s.minimum {
			add(path, "%s is less than minimum %v", x, *s.minimum)
		}
		if s.maximum != nil && f > *s.maximum {
			add(path, "%s is greater than maximum %v", x, *s.maximum)
		}
	}
	if len(s.anyOf) > 0 {
		ok := false
		for _, a := range s.anyOf {
			if len(a.validate(v, path)) == 0 {
				ok = true
				break
			}
		}
		if !ok {
			add(path, "matches none of the %d anyOf schemas", len(s.anyOf))
		}
	}
	return errs
}

// equalJSON compares two decoded JSON values; numbers compare by value
// (1 equals 1.0).
func equalJSON(a, b any) bool {
	switch x := a.(type) {
	case json.Number:
		y, ok := b.(json.Number)
		if !ok {
			return false
		}
		fx, e1 := strconv.ParseFloat(string(x), 64)
		fy, e2 := strconv.ParseFloat(string(y), 64)
		return e1 == nil && e2 == nil && fx == fy
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !equalJSON(x[i], y[i]) {
				return false
			}
		}
		return true
	case map[string]any:
		y, ok := b.(map[string]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for k, v := range x {
			w, ok := y[k]
			if !ok || !equalJSON(v, w) {
				return false
			}
		}
		return true
	}
	return a == b
}
