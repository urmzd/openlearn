// Command openlearn is a desktop learning platform: it reads a course pack,
// generates timeboxed exercises from it, coaches while you work, and scores
// the result against a hidden rubric.
package main

import (
	"context"
	"embed"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"github.com/urmzd/openlearn/app/internal/mcpserver"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/mac"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	// `openlearn mcp [-pack DIR]` serves the course pack to an agent instead of
	// opening a window, so the one binary a learner installs is also the one
	// their agent connects to.
	if len(os.Args) > 1 && os.Args[1] == "mcp" {
		flags := flag.NewFlagSet("mcp", flag.ExitOnError)
		pack := flags.String("pack", "", "serve this pack directory instead of the one the app has open")
		_ = flags.Parse(os.Args[2:])
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		if err := mcpserver.Serve(ctx, *pack, "app"); err != nil {
			fmt.Fprintln(os.Stderr, "openlearn mcp:", err)
			os.Exit(1)
		}
		return
	}

	app := NewApp()

	err := wails.Run(&options.App{
		Title:     "OpenLearn",
		Width:     1440,
		Height:    920,
		MinWidth:  1100,
		MinHeight: 700,
		AssetServer: &assetserver.Options{
			Assets: assets,
			// Requests the bundled frontend does not answer are images out of
			// the open course pack.
			Handler: packAssets{app},
		},
		BackgroundColour: &options.RGBA{R: 11, G: 13, B: 18, A: 1},
		OnStartup:        app.startup,
		OnShutdown:       app.shutdown,
		Mac: &mac.Options{
			TitleBar: mac.TitleBarHiddenInset(),
			About: &mac.AboutInfo{
				Title:   "OpenLearn",
				Message: "Timeboxed system design and coding practice with a live coach.",
			},
		},
		Bind: []any{app},
	})
	if err != nil {
		println("Error:", err.Error())
	}
}
