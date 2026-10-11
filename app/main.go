// Command openlearn is a desktop learning platform: it reads a course pack,
// generates timeboxed exercises from it, coaches while you work, and scores
// the result against a hidden rubric.
package main

import (
	"embed"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/mac"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
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
