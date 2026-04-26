package tensor

import (
	"math"
	"math/rand"
)

// sampleNormal draws a single N(0,1) sample via Box-Muller.
func sampleNormal(rng *rand.Rand) float64 {
	// Add a small epsilon to avoid log(0).
	u1 := rng.Float64() + 1e-10
	u2 := rng.Float64() + 1e-10
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// NormalFill fills t.Data with samples from N(mean, std^2).
func NormalFill(t *Tensor, mean, std float32, rng *rand.Rand) {
	m := float64(mean)
	s := float64(std)
	for i := range t.Data {
		t.Data[i] = float32(m + s*sampleNormal(rng))
	}
}

// ZeroFill sets every element of t.Data to 0.
func ZeroFill(t *Tensor) {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// OneFill sets every element of t.Data to 1.
func OneFill(t *Tensor) {
	for i := range t.Data {
		t.Data[i] = 1
	}
}

// ScaledNormalFill fills t.Data with N(0, (std/sqrt(2*nLayer))^2). This is the
// GPT-2-style residual-projection initialization that keeps pre-activation
// variance stable as depth grows.
func ScaledNormalFill(t *Tensor, std float32, nLayer int, rng *rand.Rand) {
	scale := float64(std) / math.Sqrt(2*float64(nLayer))
	for i := range t.Data {
		t.Data[i] = float32(scale * sampleNormal(rng))
	}
}
