package dur_90

import (
	"math"
	"testing"

	sst "openlearn.urmzd.com/tl/contracts/testing"
	"tinyllm/ds/demo"
)

func TestHandExample(t *testing.T) {
	// WHY: the chapter's worked example: 1 + 2 + 3 = 6.
	// KIND: unit
	if got := demo.Sum([]float64{1, 2, 3}); !sst.Close(got, 6, 1e-12, 0) {
		t.Fatalf("Sum = %v, want 6", got)
	}
}

func TestSkipsNaN(t *testing.T) {
	// WHY: one NaN would otherwise poison every later total.
	// KIND: boundary
	if got := demo.Sum([]float64{1, math.NaN(), 2}); !sst.Close(got, 3, 1e-12, 0) {
		t.Fatalf("Sum = %v, want 3", got)
	}
}
