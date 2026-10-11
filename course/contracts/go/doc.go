// Package contracts is the Go side of the course contracts (DESIGN 2.7,
// 3.2). The learner's go/go.mod requires it through
//
//	replace openlearn.urmzd.com/tl/contracts => ../contracts/go
//
// gen/ holds the protoc-gen-go and protoc-gen-go-grpc output for
// proto/tl/{engine,kv,control,durable,raft}/v1, committed so the learner
// never runs protoc. Interfaces and frozen test helpers (contracts/go/testing,
// D35) are added by the modules that own them.
package contracts
