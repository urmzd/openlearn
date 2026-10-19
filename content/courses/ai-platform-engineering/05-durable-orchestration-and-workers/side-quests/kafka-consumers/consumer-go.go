// Streamflow order-worker consumer (Go) -- the at-least-once + idempotent
// pattern for distributed workers, with graceful shutdown for Kubernetes
// (Containers, Kubernetes & Workloads -- ../../01-containers-kubernetes/).
//
// Illustrative: needs github.com/confluentinc/confluent-kafka-go/v2 and a broker.
//
//	go run consumer-go.go
//
// Key decisions shown:
//   - enable.auto.commit=false  -> we commit offsets AFTER durable work
//     (at-least-once; a crash before commit redelivers -> we must dedupe).
//   - cooperative-sticky rebalancing -> joins/leaves don't stop the whole group.
//   - SIGTERM -> leave the group cleanly so partitions reassign immediately.
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/confluentinc/confluent-kafka-go/v2/kafka"
)

// idempotent reports whether this message key was already processed.
// Real impl: Redis SETNX with a TTL (see the C4 component diagram in
// ../../01-containers-kubernetes/).
func idempotent(key string) bool { /* SETNX key -> false if it already existed */ return false }

func handle(m *kafka.Message) error {
	key := string(m.Key)
	if idempotent(key) {
		return nil // duplicate redelivery -- safe to skip, offset still advances
	}
	// ... business logic: persist order state in a DB transaction,
	//     emit settlement.requested via the producer (outbox pattern) ...
	return nil
}

func main() {
	c, err := kafka.NewConsumer(&kafka.ConfigMap{
		"bootstrap.servers":        os.Getenv("KAFKA_BROKERS"),
		"group.id":                 "order-workers",
		"enable.auto.commit":       false,                  // we own commits
		"partition.assignment.strategy": "cooperative-sticky", // no stop-the-world
		"auto.offset.reset":        "latest",
		"max.poll.interval.ms":     300000, // kicked if a handler exceeds this
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := c.Subscribe("orders", nil); err != nil {
		log.Fatal(err)
	}

	// Graceful shutdown: SIGTERM from K8s -> stop the loop, Close() leaves the
	// group so our partitions are reassigned without waiting for session timeout.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()

	for ctx.Err() == nil {
		ev := c.Poll(100)
		msg, ok := ev.(*kafka.Message)
		if !ok {
			continue // not a message (error/stats event) -- ignore for brevity
		}
		if err := handle(msg); err != nil {
			log.Printf("handler failed at offset %v: %v", msg.TopicPartition.Offset, err)
			continue // do NOT commit; message will be redelivered (retry/DLQ logic omitted)
		}
		// Commit ONLY after durable work. This is what makes it at-least-once.
		if _, err := c.CommitMessage(msg); err != nil {
			log.Printf("commit failed: %v", err)
		}
	}

	log.Println("SIGTERM received, draining...")
	time.Sleep(500 * time.Millisecond) // let in-flight work settle
	c.Close()                          // leaves the group cleanly
}
