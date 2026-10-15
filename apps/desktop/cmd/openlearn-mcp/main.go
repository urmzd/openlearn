// Command openlearn-mcp serves the open course pack over the Model Context
// Protocol on stdin and stdout, for any agent that speaks it.
//
//	openlearn-mcp                 follow the pack the desktop app has open
//	openlearn-mcp -pack /a/pack   serve one pack directory, whatever the app does
//
// It is the same server `openlearn mcp` runs, built without the desktop app's
// webview dependencies so it can be installed on its own. It shares the app's
// store (~/.openlearn), so progress and sessions are the learner's own, and it
// works whether or not the app is running.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"github.com/urmzd/openlearn/app/internal/mcpserver"
)

// version is set at build time with -ldflags.
var version = "dev"

func main() {
	pack := flag.String("pack", "", "serve this pack directory instead of the one the app has open")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := mcpserver.Serve(ctx, *pack, version); err != nil {
		fmt.Fprintln(os.Stderr, "openlearn-mcp:", err)
		os.Exit(1)
	}
}
