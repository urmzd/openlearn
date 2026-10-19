// Package testkit is the fault and determinism kit the Go model test suites
// import (learner-tests/dur.*, agent and gateway tests).
//
//	clock       a fake Clock (Now, After, NewTimer, Sleep) driven by Advance
//	failpoint   named failpoints enabled by TL_FAILPOINTS="name=crash;other=error(msg)"
//	effects     an HTTP side-effect sink that records idempotency keys; AssertExactlyOnce
//	proc        KillLoop: start a binary under test, SIGKILL it at seeded offsets, restart, check
//	chaosproxy  a TCP proxy: latency, drop, reset mid-stream, half-open, bandwidth limit
//	otlpsink    an in-test OTLP/HTTP receiver (JSON and protobuf, gzip) with span-tree checks
//	promscrape  a Prometheus text-format scraper with metric and histogram lookups
//	faketool    a rule-based fake OpenAI-compatible provider keyed on the last message
package testkit
