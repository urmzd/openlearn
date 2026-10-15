// Pull vs push backpressure, in Go. Standalone, dependency-free.
//
//	go run backpressure-demo.go
//
// Same model as backpressure_demo.py: a producer offers LAMBDA msgs/tick to a
// single worker that can durably handle MU < LAMBDA (overload).
//
//   - PUSH: the broker shoves LAMBDA/tick at the worker; the surplus (LAMBDA-MU)
//     has nowhere safe to live and is dropped/overruns.
//   - PULL: the worker polls for <= MU/tick; the surplus stays in the log as a
//     growing backlog (consumer lag). Nothing is lost; the lag is the
//     backpressure signal KEDA scales on (topic 01).
//
// queue depth(t) = integral of (arrival - service); Little's Law L = lambda*W.
package main

import "fmt"

const (
	ticks  = 20
	lambda = 10.0 // arrivals per tick
	mu     = 6.0  // single worker service rate per tick
)

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// push: broker sets the pace. Returns (handled, dropped).
func push() (int, int) {
	handled, dropped := 0, 0
	for i := 0; i < ticks; i++ {
		done := min(lambda, mu)
		handled += int(done)
		dropped += int(lambda - done) // surplus overruns the worker
	}
	return handled, dropped
}

// pull: consumer sets the pace via poll(). Returns (handled, peakBacklog).
func pull() (int, int) {
	handled := 0
	backlog := 0.0 // = integral of (arrival - service) = consumer lag
	peak := 0.0
	for i := 0; i < ticks; i++ {
		backlog += lambda          // producer appends to the log
		pulled := min(backlog, mu) // poll() asks only for what we can handle
		backlog -= pulled          // rest stays in the log -> lag grows
		handled += int(pulled)
		if backlog > peak {
			peak = backlog
		}
	}
	return handled, int(peak)
}

func main() {
	pHandled, pDropped := push()
	lHandled, lBacklog := pull()

	fmt.Printf("params: lambda=%.0f/tick  mu=%.0f/tick  ticks=%d  (overload)\n\n", lambda, mu, ticks)
	fmt.Printf("PUSH: handled=%d  DROPPED=%d  <- worker overwhelmed, surplus lost\n", pHandled, pDropped)
	fmt.Printf("PULL: handled=%d  backlog(lag)=%d  <- nothing lost; lag is the signal\n\n", lHandled, lBacklog)

	if pDropped <= 0 {
		panic("push should overrun under overload")
	}
	if lHandled != pHandled {
		panic("both drain at mu; pull just retains the surplus")
	}

	w := 1.0 / mu
	fmt.Printf("Little's Law L = lambda*W = %.0f * %.3f = %.2f in-flight (steady-state)\n", lambda, w, lambda*w)
	fmt.Println("OK")
}
