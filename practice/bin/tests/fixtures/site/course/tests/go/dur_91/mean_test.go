package dur_91

import (
	"errors"
	"testing"

	sst "openlearn.urmzd.com/tl/contracts/testing"
	"tinyllm/ds/demo"
)

func TestHandExample(t *testing.T) {
	// WHY: the chapter's worked example: mean of 1, 2, 3, 4 is 2.5.
	// KIND: unit
	got, err := demo.Mean([]float64{1, 2, 3, 4})
	if err != nil || !sst.Close(got, 2.5, 1e-12, 0) {
		t.Fatalf("Mean = %v, %v; want 2.5", got, err)
	}
}

func TestEmptyIsError(t *testing.T) {
	// WHY: the mean of nothing is undefined; callers must see that, not a 0.
	// KIND: boundary
	if _, err := demo.Mean(nil); !errors.Is(err, demo.ErrEmpty) {
		t.Fatalf("Mean(nil) err = %v, want ErrEmpty", err)
	}
}
