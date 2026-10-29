// primers/lang.10/go: the Go half of the lang.10 primer (section 4).
//   kvstore/        a toy tl.kv.v1.KvTransferService over the vendored stubs
//   cmd/kvstore     its gRPC server binary
//   cmd/evolve      decodes and re-encodes a KvChunk: schema evolution
module lang10

go 1.26.0

require (
	google.golang.org/grpc v1.85.0
	google.golang.org/protobuf v1.36.12
	openlearn.urmzd.com/tl/contracts v0.0.0
)

require (
	golang.org/x/net v0.62.0 // indirect
	golang.org/x/sys v0.49.0 // indirect
	golang.org/x/text v0.43.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260817212433-ac3dfec99bb1 // indirect
)

replace openlearn.urmzd.com/tl/contracts => ../../../contracts/go
