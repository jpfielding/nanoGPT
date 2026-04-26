// Package schedule provides learning-rate schedules for training loops.
package schedule

import "math"

// CosineSchedule is a cosine-decay learning rate with a linear warmup phase.
//
// - For step < WarmupIters, LR ramps linearly from ~0 to MaxLR.
// - For step > MaxIters, LR is pinned at MinLR.
// - Otherwise LR follows a half-cosine from MaxLR down to MinLR.
type CosineSchedule struct {
	WarmupIters int
	MaxIters    int
	MaxLR       float32
	MinLR       float32
}

// LRAt returns the learning rate for the given step (0-indexed).
func (s CosineSchedule) LRAt(step int) float32 {
	if step < s.WarmupIters {
		return s.MaxLR * float32(step+1) / float32(s.WarmupIters+1)
	}
	if step > s.MaxIters {
		return s.MinLR
	}
	denom := s.MaxIters - s.WarmupIters
	if denom <= 0 {
		// Degenerate schedule: warmup meets or exceeds max. Return MinLR.
		return s.MinLR
	}
	ratio := float64(step-s.WarmupIters) / float64(denom)
	coeff := 0.5 * (1.0 + math.Cos(math.Pi*ratio))
	return s.MinLR + float32(coeff)*(s.MaxLR-s.MinLR)
}
