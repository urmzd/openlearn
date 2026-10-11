// Package mcpserver exposes the open course pack over the Model Context
// Protocol, so whatever agent a learner already uses can read the course,
// see their progress, and run their exercises.
//
// The app has no assistant of its own. Tutoring is the agent's job; this
// package's job is to give that agent the same view the learner has, and the
// same verdicts: a stage it marks or an exercise it checks goes through the
// workspace the desktop app uses, so neither can disagree with the other.
package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/urmzd/openlearn/app/internal/course"
	"github.com/urmzd/openlearn/app/internal/domain"
	"github.com/urmzd/openlearn/app/internal/packs"
	"github.com/urmzd/openlearn/app/internal/store"
	"github.com/urmzd/openlearn/app/internal/workspace"
)

// instructions is sent to the connecting agent once, at initialization. It is
// where the product's stance on teaching lives now that the agent is not ours.
const instructions = `openlearn is a self-paced learning platform. This server gives you one learner's course pack: its documents, their progress through its paths, and its exercises.

Start with course_overview. Call current_focus when the learner says "this page" or "what I'm reading": it names the document open in their app. Read documents before explaining them, and prefer the course's own explanation and notation to your own.

How to tutor here:
- Answer the question asked, first, in plain words. Ground it in the course document when one covers it, and say so when you are going beyond it.
- When the learner is working toward an answer, ask the one question that moves them forward instead of handing it over. When they ask you to just explain, explain.
- Exercises are graded by the pack, not by you. check_exercise runs the learner's attempt and returns the verdict; never tell them an attempt passes without it. Do not write their solution into an attempt unless they ask you to.
- A path stage is finished when its "done when" is true. Only call mark_stage when the learner says they have met it, or asks you to. Stages with a check can only be marked once that check has passed; the tool will tell you.
- If you are unsure, say what you are unsure about.`

// Options configures a server.
type Options struct {
	Store *store.Store
	// Pack pins the server to one pack ref. Empty follows whichever pack the
	// desktop app has open, which is what a learner switching packs expects.
	Pack    string
	Version string
	Log     *slog.Logger
}

// Server is the MCP server and the workspace it serves.
type Server struct {
	opts Options
	reg  packs.Registry

	mu sync.Mutex
	ws *workspace.Workspace
	// wanted is the ref the current workspace was opened for, so a change of
	// pack in the app is noticed on the next call.
	wanted string
}

// New builds the MCP server with every tool registered.
func New(opts Options) *mcp.Server {
	if opts.Log == nil {
		opts.Log = slog.New(slog.DiscardHandler)
	}
	s := &Server{opts: opts, reg: workspace.Registry(opts.Store.Root())}
	srv := mcp.NewServer(
		&mcp.Implementation{Name: "openlearn", Title: "openlearn", Version: opts.Version},
		&mcp.ServerOptions{Instructions: instructions, Logger: opts.Log},
	)
	s.register(srv)
	return srv
}

// Serve runs the server on stdin and stdout until ctx ends or the client
// disconnects. pack pins it to one pack directory; empty follows the app.
func Serve(ctx context.Context, pack, version string) error {
	// stdout carries the protocol, so everything a human should read goes to
	// stderr.
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
	st, err := store.New("")
	if err != nil {
		return err
	}
	if pack != "" {
		if pack, err = filepath.Abs(pack); err != nil {
			return err
		}
	}
	err = New(Options{Store: st, Pack: pack, Version: version, Log: log}).Run(ctx, &mcp.StdioTransport{})
	if ctx.Err() != nil {
		return nil
	}
	return err
}

// workspace returns the pack to serve, reopening it when the app has switched.
func (s *Server) workspace() *workspace.Workspace {
	settings := s.opts.Store.LoadSettings()
	want := s.opts.Pack
	if want == "" {
		want = settings.Pack
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ws == nil || s.wanted != want {
		s.ws = workspace.Open(s.reg, want, settings.LinkedPacks, s.opts.Store.Root(), s.opts.Log)
		s.wanted = want
	}
	return s.ws
}

// readOnly marks a tool that changes nothing, so clients can run it without
// asking the learner first.
var readOnly = &mcp.ToolAnnotations{ReadOnlyHint: true}

func (s *Server) register(srv *mcp.Server) {
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "course_overview",
		Description: "The open course pack: its title, its role paths with the learner's progress on each, and its tracks. Call this first.",
		Annotations: readOnly,
	}, s.overview)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "current_focus",
		Description: "The document the learner has open in the openlearn app right now. Use it to resolve \"this page\" or \"what I'm reading\".",
		Annotations: readOnly,
	}, s.focus)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "list_track",
		Description: "The topics of one track and the documents in each, in reading order. Track directories come from course_overview.",
		Annotations: readOnly,
	}, s.listTrack)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "search",
		Description: "Full-text search over every document in the pack. Every term must appear. Returns paths to pass to read_document.",
		Annotations: readOnly,
	}, s.search)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "read_document",
		Description: "Read one markdown document by its pack-relative path. Long documents come back in pages; pass the returned next_offset to continue.",
		Annotations: readOnly,
	}, s.readDocument)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "get_path",
		Description: "One role path: every stage with its documents, what \"done\" means for it, the check that verifies it if any, and whether the learner has finished it.",
		Annotations: readOnly,
	}, s.getPath)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "mark_stage",
		Description: "Mark a path stage finished or not finished for the learner. Only when they say they have met its \"done when\", or ask. A stage with a check is refused until that check passes.",
	}, s.markStage)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "session_profile",
		Description: "How the learner's timed sessions have scored, by skill area, for one mode. Use it to find what keeps slipping.",
		Annotations: readOnly,
	}, s.profile)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "list_exercises",
		Description: "The pack's exercises grouped by kind and language, with which ones the learner has started. Empty when the pack has no practice CLI that can run here.",
		Annotations: readOnly,
	}, s.listExercises)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "open_exercise",
		Description: "Open the learner's attempt at an exercise, cloning it into their scratchpad if they have not started. Returns the attempt's files. For predict exercises the snippet is read-only and the learner writes its expected output.",
	}, s.openExercise)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "write_exercise_file",
		Description: "Overwrite one file of the learner's attempt. This is their work: only on request, and never the whole solution unless they ask for it.",
	}, s.writeExerciseFile)
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "check_exercise",
		Description: "Run the learner's attempt through the pack's own checker. Returns the verdict (pass, fail, or none if not started) and the checker's output. This is the only source of truth for whether an attempt is right.",
	}, s.checkExercise)
}

// ---------- results ----------

// jsonResult returns v as the tool's text content. Structured data goes out as
// JSON because every client can show it and every model can read it.
func jsonResult(v any) (*mcp.CallToolResult, any, error) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, nil, err
	}
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: string(data)}}}, nil, nil
}

// failed reports a problem the agent can act on (a wrong path, a pack with no
// exercises) as a tool error rather than a protocol error, so the model sees
// the message and can correct itself.
func failed(format string, args ...any) (*mcp.CallToolResult, any, error) {
	return &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{&mcp.TextContent{Text: fmt.Sprintf(format, args...)}},
	}, nil, nil
}

// ---------- reading ----------

type noArgs struct{}

type trackSummary struct {
	Dir    string `json:"dir"`
	Title  string `json:"title"`
	Topics int    `json:"topics"`
}

func (s *Server) overview(_ context.Context, _ *mcp.CallToolRequest, _ noArgs) (*mcp.CallToolResult, any, error) {
	w := s.workspace()
	info := w.Info()
	o := w.Course.Outline()
	summarize := func(in []course.Track) []trackSummary {
		out := make([]trackSummary, 0, len(in))
		for _, t := range in {
			out = append(out, trackSummary{Dir: t.Dir, Title: t.Title, Topics: len(t.Topics)})
		}
		return out
	}
	return jsonResult(map[string]any{
		"id":        info.ID,
		"title":     info.Title,
		"tagline":   info.Tagline,
		"exercises": info.Practice,
		"modes":     info.Modes,
		"start":     o.Start,
		"paths":     w.Paths(),
		"tracks":    summarize(o.Tracks),
		"reference": summarize(o.Reference),
	})
}

func (s *Server) focus(_ context.Context, _ *mcp.CallToolRequest, _ noArgs) (*mcp.CallToolResult, any, error) {
	f, ok := s.opts.Store.LoadFocus()
	if !ok || f.Doc == "" {
		return jsonResult(map[string]any{"open": false, "note": "The learner has no document open in the app, or the app is not running."})
	}
	out := map[string]any{"open": true, "pack": f.PackTitle, "document": f.Doc, "since": f.At}
	w := s.workspace()
	if f.Pack != w.Ref {
		out["note"] = "This document is in a different pack from the one this server is serving."
	} else if doc, err := w.Course.Doc(f.Doc); err == nil {
		out["title"] = doc.Title
	}
	return jsonResult(out)
}

type trackArgs struct {
	Dir string `json:"dir" jsonschema:"the track's directory, as given by course_overview"`
}

func (s *Server) listTrack(_ context.Context, _ *mcp.CallToolRequest, in trackArgs) (*mcp.CallToolResult, any, error) {
	o := s.workspace().Course.Outline()
	for _, group := range [][]course.Track{o.Tracks, o.Reference} {
		for _, t := range group {
			if t.Dir == strings.Trim(in.Dir, "/") {
				return jsonResult(t)
			}
		}
	}
	return failed("no track %q; course_overview lists the track directories", in.Dir)
}

type searchArgs struct {
	Query string `json:"query" jsonschema:"terms to find; every one must appear in a document"`
	Limit int    `json:"limit,omitempty" jsonschema:"maximum documents to return, default 15"`
}

func (s *Server) search(_ context.Context, _ *mcp.CallToolRequest, in searchArgs) (*mcp.CallToolResult, any, error) {
	limit := in.Limit
	if limit <= 0 || limit > 50 {
		limit = 15
	}
	return jsonResult(s.workspace().Course.Search(in.Query, limit))
}

// pageChars is one page of a document. Chapters run to tens of kilobytes, and a
// tool result that large would crowd out the conversation it is meant to serve.
const pageChars = 12000

type readArgs struct {
	Path   string `json:"path" jsonschema:"pack-relative path of a markdown document"`
	Offset int    `json:"offset,omitempty" jsonschema:"character offset to start from, as returned in next_offset"`
}

func (s *Server) readDocument(_ context.Context, _ *mcp.CallToolRequest, in readArgs) (*mcp.CallToolResult, any, error) {
	doc, err := s.workspace().Course.Doc(in.Path)
	if err != nil {
		return failed("cannot read %q: it is not a document in this pack. Use search or list_track to find its path.", in.Path)
	}
	text := []rune(doc.Markdown)
	if in.Offset < 0 || in.Offset > len(text) {
		return failed("offset %d is outside the document, which has %d characters", in.Offset, len(text))
	}
	end := in.Offset + pageChars
	if end > len(text) {
		end = len(text)
	}
	header := fmt.Sprintf("path: %s\ntitle: %s\ncharacters: %d-%d of %d\n", doc.Path, doc.Title, in.Offset, end, len(text))
	if end < len(text) {
		header += fmt.Sprintf("next_offset: %d\n", end)
	}
	return &mcp.CallToolResult{Content: []mcp.Content{
		&mcp.TextContent{Text: header + "\n" + string(text[in.Offset:end])},
	}}, nil, nil
}

// ---------- progress ----------

type pathArgs struct {
	Name string `json:"name" jsonschema:"the path's name, as given by course_overview"`
}

func (s *Server) getPath(_ context.Context, _ *mcp.CallToolRequest, in pathArgs) (*mcp.CallToolResult, any, error) {
	p, err := s.workspace().Path(in.Name)
	if err != nil {
		return failed("%v; course_overview lists the path names", err)
	}
	return jsonResult(p)
}

type markArgs struct {
	Path  string `json:"path" jsonschema:"the path's name"`
	Stage string `json:"stage" jsonschema:"the stage's key from get_path, such as 3 or part:3"`
	Done  bool   `json:"done" jsonschema:"true to mark the stage finished, false to unmark it"`
}

func (s *Server) markStage(ctx context.Context, _ *mcp.CallToolRequest, in markArgs) (*mcp.CallToolResult, any, error) {
	res, err := s.workspace().MarkStage(ctx, in.Path, in.Stage, in.Done)
	if err != nil {
		return failed("%v", err)
	}
	if res.Verdict == "fail" {
		return failed("not marked: %s", strings.TrimSpace(res.Output))
	}
	return jsonResult(map[string]any{"path": in.Path, "stage": in.Stage, "done": in.Done, "output": res.Output})
}

type profileArgs struct {
	Mode string `json:"mode,omitempty" jsonschema:"system_design, coding, or written; defaults to the pack's first mode"`
}

func (s *Server) profile(_ context.Context, _ *mcp.CallToolRequest, in profileArgs) (*mcp.CallToolResult, any, error) {
	mode := domain.Mode(in.Mode)
	if in.Mode == "" {
		mode = domain.Mode(s.workspace().Info().Modes[0])
	}
	if !domain.ValidMode(mode) {
		return failed("unknown mode %q; use system_design, coding, or written", in.Mode)
	}
	p, err := s.opts.Store.Profile(mode)
	if err != nil {
		return failed("%v", err)
	}
	return jsonResult(p)
}

// ---------- exercises ----------

type exerciseArgs struct {
	Kind string `json:"kind" jsonschema:"the exercise kind from list_exercises, such as predict or build"`
	Lang string `json:"lang" jsonschema:"the language or group from list_exercises"`
	Name string `json:"name" jsonschema:"the exercise name from list_exercises, such as 01-dynamic-array"`
}

const noExercises = "this pack has no exercises that can run here: it needs to be opened from a folder on disk, with its practice CLI"

func (s *Server) listExercises(ctx context.Context, _ *mcp.CallToolRequest, _ noArgs) (*mcp.CallToolResult, any, error) {
	w := s.workspace()
	if w.Runner == nil {
		return jsonResult([]any{})
	}
	groups, err := w.Runner.List(ctx)
	if err != nil {
		return failed("%v", err)
	}
	return jsonResult(groups)
}

func (s *Server) openExercise(ctx context.Context, _ *mcp.CallToolRequest, in exerciseArgs) (*mcp.CallToolResult, any, error) {
	w := s.workspace()
	if w.Runner == nil {
		return failed(noExercises)
	}
	ws, err := w.Runner.Start(ctx, in.Kind, in.Lang, in.Name)
	if err != nil {
		return failed("%v", err)
	}
	return jsonResult(ws)
}

type writeArgs struct {
	exerciseArgs
	File    string `json:"file" jsonschema:"the file's path within the attempt, as returned by open_exercise"`
	Content string `json:"content" jsonschema:"the file's full new content"`
}

func (s *Server) writeExerciseFile(_ context.Context, _ *mcp.CallToolRequest, in writeArgs) (*mcp.CallToolResult, any, error) {
	w := s.workspace()
	if w.Runner == nil {
		return failed(noExercises)
	}
	if err := w.Runner.Save(in.Kind, in.Lang, in.Name, in.File, in.Content); err != nil {
		return failed("%v", err)
	}
	return jsonResult(map[string]any{"written": in.File})
}

func (s *Server) checkExercise(ctx context.Context, _ *mcp.CallToolRequest, in exerciseArgs) (*mcp.CallToolResult, any, error) {
	w := s.workspace()
	if w.Runner == nil {
		return failed(noExercises)
	}
	res, err := w.Runner.Check(ctx, in.Kind, in.Lang, in.Name)
	if err != nil {
		return failed("%v", err)
	}
	return jsonResult(res)
}
