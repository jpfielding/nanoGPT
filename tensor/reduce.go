package tensor

// SumAll returns the scalar sum of all elements in t.
func SumAll(t *Tensor) float32 {
	// Accumulate in float64 to reduce rounding drift on large tensors.
	var sum float64
	for _, v := range t.Data {
		sum += float64(v)
	}
	return float32(sum)
}

// MeanAll returns the scalar arithmetic mean of all elements in t.
// Returns 0 for an empty tensor.
func MeanAll(t *Tensor) float32 {
	n := len(t.Data)
	if n == 0 {
		return 0
	}
	var sum float64
	for _, v := range t.Data {
		sum += float64(v)
	}
	return float32(sum / float64(n))
}

// MaxAlong returns a new tensor containing the maximum values of t along dim.
// The output shape equals t.Shape with dim removed.
func MaxAlong(t *Tensor, dim int) *Tensor {
	if dim < 0 || dim >= len(t.Shape) {
		panic("tensor.MaxAlong: dim out of range")
	}
	outShape := make([]int, 0, len(t.Shape)-1)
	outShape = append(outShape, t.Shape[:dim]...)
	outShape = append(outShape, t.Shape[dim+1:]...)

	// Degenerate case: reducing a 1-D tensor yields a 0-D result held as a
	// single-element tensor (Shape may be empty).
	if len(outShape) == 0 {
		out := &Tensor{
			Shape:   nil,
			Strides: nil,
			Data:    make([]float32, 1),
		}
		if t.Shape[dim] == 0 {
			return out
		}
		m := t.Data[0]
		for _, v := range t.Data[1:] {
			if v > m {
				m = v
			}
		}
		out.Data[0] = m
		return out
	}

	out := New(outShape...)
	dimSize := t.Shape[dim]
	dimStride := t.Strides[dim]

	// Outer = product of dims before `dim`; Inner = product after.
	outer := 1
	for i := 0; i < dim; i++ {
		outer *= t.Shape[i]
	}
	inner := 1
	for i := dim + 1; i < len(t.Shape); i++ {
		inner *= t.Shape[i]
	}

	outerStride := dimSize * inner // stride in t.Data between outer indices
	for o := 0; o < outer; o++ {
		baseIn := o * outerStride
		baseOut := o * inner
		for i := 0; i < inner; i++ {
			start := baseIn + i
			m := t.Data[start]
			for d := 1; d < dimSize; d++ {
				v := t.Data[start+d*dimStride]
				if v > m {
					m = v
				}
			}
			out.Data[baseOut+i] = m
		}
	}
	return out
}

// SumSq returns the sum of squared elements of t. Useful for computing
// gradient norms without allocating a temporary.
func SumSq(t *Tensor) float32 {
	var sum float64
	for _, v := range t.Data {
		f := float64(v)
		sum += f * f
	}
	return float32(sum)
}
