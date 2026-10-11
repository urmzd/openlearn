// Package kvstore is a toy KV block store behind tl.kv.v1.KvTransferService
// (contracts/proto/tl/kv/v1/kv.proto), implemented against the vendored
// generated stubs (openlearn.urmzd.com/tl/contracts/gen/tl/kv/v1).
//
// It keeps each block's bytes in a map keyed by block hash, which is
// enough to practice the three kinds of RPC the engine's real KV transfer
// (L10.6, in Rust over the C pool) serves: a unary call (HasBlocks), a
// client stream (PushKv), and an idempotent cleanup (Release), with the
// gRPC status codes the contract names.
package kvstore

import (
	"context"
	"errors"
	"hash/crc32"
	"io"
	"sync"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	kvv1 "openlearn.urmzd.com/tl/contracts/gen/tl/kv/v1"
)

// Format is the only KV format this store reads (formats/kv-block.md v1).
const Format = 1

// castagnoli is the CRC-32C table: the KvChunk checksum (kv.proto).
var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// Store is safe for concurrent use: gRPC runs every call on its own goroutine.
type Store struct {
	kvv1.UnimplementedKvTransferServiceServer

	mu      sync.Mutex
	blocks  map[uint64][]byte   // block hash -> payload
	handles map[string][]uint64 // handle id -> hashes it references
}

// New returns an empty store.
func New() *Store {
	// SOLUTION-BEGIN lang.10
	return &Store{blocks: map[uint64][]byte{}, handles: map[string][]uint64{}}
	// SOLUTION-END
}

// HasBlocks reports, for each hash, whether the store holds that block.
func (s *Store) HasBlocks(ctx context.Context, req *kvv1.HasBlocksRequest) (*kvv1.HasBlocksResponse, error) {
	// SOLUTION-BEGIN lang.10
	if req.GetKvFormat() != Format {
		return nil, status.Errorf(codes.FailedPrecondition, "kv_format %d: this store reads format %d", req.GetKvFormat(), Format)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	present := make([]bool, len(req.GetBlockHashes()))
	for i, h := range req.GetBlockHashes() {
		_, present[i] = s.blocks[h]
	}
	return &kvv1.HasBlocksResponse{Present: present}, nil
	// SOLUTION-END
}

// PushKv receives the blocks of one handle in order. A chunk with a
// payload is checked (CRC-32C) and stored; an empty one must name a block
// the store already holds (deduplicated). The ack comes after the last
// chunk. On any error nothing of this handle is kept.
func (s *Store) PushKv(stream kvv1.KvTransferService_PushKvServer) error {
	// SOLUTION-BEGIN lang.10
	var ack kvv1.KvAck
	var hashes []uint64
	stored := map[uint64][]byte{}
	next := uint32(0)
	total := uint32(0)
	for {
		c, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if c.GetKvFormat() != Format {
			return status.Errorf(codes.FailedPrecondition, "kv_format %d: this store reads format %d", c.GetKvFormat(), Format)
		}
		if next == 0 {
			ack.HandleId, total = c.GetHandleId(), c.GetNBlocksTotal()
		}
		if c.GetHandleId() != ack.HandleId || c.GetNBlocksTotal() != total || c.GetBlockIndex() != next {
			return status.Errorf(codes.InvalidArgument, "chunk %d of %q out of order", c.GetBlockIndex(), c.GetHandleId())
		}
		next++
		if len(c.GetPayload()) == 0 {
			s.mu.Lock()
			_, ok := s.blocks[c.GetBlockHash()]
			s.mu.Unlock()
			if !ok {
				return status.Errorf(codes.NotFound, "block %016x is not cached here: resend it with its payload", c.GetBlockHash())
			}
			ack.BlocksDeduped++
		} else {
			if got := crc32.Checksum(c.GetPayload(), castagnoli); got != c.GetCrc32C() {
				return status.Errorf(codes.DataLoss, "block %d: crc32c %08x, chunk says %08x", c.GetBlockIndex(), got, c.GetCrc32C())
			}
			ack.BlocksReceived++
			if c.GetBlockHash() != 0 {
				stored[c.GetBlockHash()] = c.GetPayload()
			}
		}
		hashes = append(hashes, c.GetBlockHash())
	}
	if next != total {
		return status.Errorf(codes.InvalidArgument, "stream ended after %d of %d chunks", next, total)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for h, p := range stored {
		s.blocks[h] = p
	}
	s.handles[ack.HandleId] = hashes
	return stream.SendAndClose(&ack)
	// SOLUTION-END
}

// Release forgets a handle. Idempotent: an unknown handle is fine.
func (s *Store) Release(ctx context.Context, req *kvv1.ReleaseRequest) (*kvv1.ReleaseResponse, error) {
	// SOLUTION-BEGIN lang.10
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.handles, req.GetHandleId())
	return &kvv1.ReleaseResponse{}, nil
	// SOLUTION-END
}

// Handles is the number of handles held (tests).
func (s *Store) Handles() int {
	// SOLUTION-BEGIN lang.10
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.handles)
	// SOLUTION-END
}
