// primers/lang.10/go: the Go half of the lang.10 primer (section 4).
//   kvstore/        a toy tl.kv.v1.KvTransferService over the vendored stubs
//   cmd/kvstore     its gRPC server binary
//   cmd/evolve      decodes and re-encodes a KvChunk: schema evolution
module lang10

go 1.25.0

require (
	google.golang.org/grpc v1.83.1
	google.golang.org/protobuf v1.36.11
	openlearn.urmzd.com/tl/contracts v0.0.0
)

require (
	golang.org/x/net v0.55.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	golang.org/x/text v0.37.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260526163538-3dc84a4a5aaa // indirect
)

replace openlearn.urmzd.com/tl/contracts => ../../../contracts/go
