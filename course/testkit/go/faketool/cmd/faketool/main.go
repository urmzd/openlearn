// Command faketool serves a rules file as a fake OpenAI-compatible provider
// (the `faketool` provider of MS-agent's PR runs, DESIGN B12):
//
//	go run openlearn.urmzd.com/tl/testkit/faketool/cmd/faketool --rules rules.json --port 0
//
// The first stdout line is `listening on 127.0.0.1:<port>`.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"openlearn.urmzd.com/tl/testkit/faketool"
)

func main() {
	rules := flag.String("rules", "", "rules JSON file")
	port := flag.Int("port", 0, "port (0: a free one)")
	flag.Parse()
	r, err := faketool.Load(*rules)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	s, err := faketool.Start(r, fmt.Sprintf("127.0.0.1:%d", *port))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("listening on %s\n", s.BaseURL()[len("http://"):len(s.BaseURL())-len("/v1")])
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	<-ch
	s.Close()
}
