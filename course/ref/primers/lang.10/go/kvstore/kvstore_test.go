package kvstore

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	kvv1 "openlearn.urmzd.com/tl/contracts/gen/tl/kv/v1"
)

func TestHasBlocksRefusesAnotherFormat(t *testing.T) {
	_, err := New().HasBlocks(context.Background(), &kvv1.HasBlocksRequest{BlockHashes: []uint64{1}, KvFormat: 2})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("got %v, want FAILED_PRECONDITION", err)
	}
}

func TestReleaseIsIdempotent(t *testing.T) {
	s := New()
	for i := 0; i < 2; i++ {
		if _, err := s.Release(context.Background(), &kvv1.ReleaseRequest{HandleId: "nope"}); err != nil {
			t.Fatal(err)
		}
	}
}
