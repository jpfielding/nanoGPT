// Package tensor provides float32 multi-dimensional arrays with autograd support
// via the closure/tape pattern. Each forward op returns a BackwardFn closure that,
// when called, accumulates gradients into operand .Grad slices.
package tensor

// BackwardFn is called during backpropagation to accumulate gradients into
// operand tensors' Grad slices. BackwardFns are always additive: they never
// zero existing gradient state.
type BackwardFn func()

// Tensor is a flat float32 multi-dimensional array with row-major strides and
// an optional gradient buffer.
type Tensor struct {
	Shape        []int
	Strides      []int     // row-major strides; Strides[last]=1
	Data         []float32 // contiguous, length == Numel()
	Grad         []float32 // same length as Data; nil if not needed
	RequiresGrad bool
}

// computeStrides returns row-major strides for the given shape.
// strides[ndim-1] = 1, strides[i] = strides[i+1] * shape[i+1].
func computeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// shapeNumel returns the product of dimensions. Returns 1 for a 0-D (scalar) shape.
func shapeNumel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// New allocates a tensor with the given shape. Data is zero-initialized;
// Grad is left nil.
func New(shape ...int) *Tensor {
	s := append([]int(nil), shape...)
	n := shapeNumel(s)
	return &Tensor{
		Shape:   s,
		Strides: computeStrides(s),
		Data:    make([]float32, n),
	}
}

// NewWithGrad allocates a tensor with both Data and Grad buffers, and marks
// it as RequiresGrad.
func NewWithGrad(shape ...int) *Tensor {
	t := New(shape...)
	t.Grad = make([]float32, len(t.Data))
	t.RequiresGrad = true
	return t
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int {
	return len(t.Data)
}

// ZeroGrad ensures Grad is allocated and fills it with zeros.
func (t *Tensor) ZeroGrad() {
	if t.Grad == nil {
		t.Grad = make([]float32, len(t.Data))
		return
	}
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

// Clone returns a deep copy of Data. Grad is not copied (result has nil Grad).
func (t *Tensor) Clone() *Tensor {
	shape := append([]int(nil), t.Shape...)
	out := &Tensor{
		Shape:   shape,
		Strides: computeStrides(shape),
		Data:    make([]float32, len(t.Data)),
	}
	copy(out.Data, t.Data)
	return out
}

// View returns a new tensor header that shares the underlying Data (and Grad,
// if present) with t but presents a different shape. Panics if the element
// count differs.
func (t *Tensor) View(shape ...int) *Tensor {
	s := append([]int(nil), shape...)
	if shapeNumel(s) != t.Numel() {
		panic("tensor.View: element count mismatch")
	}
	return &Tensor{
		Shape:        s,
		Strides:      computeStrides(s),
		Data:         t.Data,
		Grad:         t.Grad,
		RequiresGrad: t.RequiresGrad,
	}
}

// At returns the flat index for a multi-index into t.
func At(t *Tensor, idx ...int) int {
	if len(idx) != len(t.Shape) {
		panic("tensor.At: index arity mismatch")
	}
	off := 0
	for i, v := range idx {
		off += v * t.Strides[i]
	}
	return off
}
