package model

import (
	"math/rand"

	"nanogpt/optim"
	"nanogpt/tensor"
)

// Linear is an affine transform y = x @ W^T + b.
// Weight is stored as [OutFeatures, InFeatures] to match the PyTorch /
// nanoGPT convention; the forward uses MatMulTransB so the effective
// operation is still x @ W^T.
type Linear struct {
	Weight *tensor.Tensor // [OutFeatures, InFeatures]
	Bias   *tensor.Tensor // [OutFeatures] or nil
}

// NewLinear constructs a Linear layer with weight ~ N(0, 0.02^2) and zero
// bias (or no bias when bias=false).
func NewLinear(in, out int, bias bool, rng *rand.Rand) *Linear {
	w := tensor.NewWithGrad(out, in)
	tensor.NormalFill(w, 0, 0.02, rng)
	l := &Linear{Weight: w}
	if bias {
		b := tensor.NewWithGrad(out)
		tensor.ZeroFill(b)
		l.Bias = b
	}
	return l
}

// Forward computes x @ W^T (+ bias). x may be 2-D or higher; matmul is
// applied over the last two dimensions via a flattened 2-D view.
func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, tensor.BackwardFn) {
	out := l.Weight.Shape[0]
	in := l.Weight.Shape[1]
	if x.Shape[len(x.Shape)-1] != in {
		panic("model.Linear.Forward: last dim of x must equal in_features")
	}

	// Flatten leading dims to 2-D so MatMulTransB can be used.
	rows := x.Numel() / in
	x2 := x.View(rows, in)
	y2, mmBwd := tensor.MatMulTransB(x2, l.Weight)

	// Reshape output back to match x's leading dims with last replaced by out.
	outShape := append([]int(nil), x.Shape...)
	outShape[len(outShape)-1] = out
	y := y2.View(outShape...)

	if l.Bias == nil {
		return y, mmBwd
	}

	yb, addBwd := tensor.AddBias(y, l.Bias)
	// Wire upstream grad back through y's grad so mmBwd can see it.
	bwd := func() {
		addBwd()
		// AddBias writes x.Grad += upstream; x here is y, whose Grad is
		// shared with y2 through the view. So mmBwd reads y2.Grad which
		// already has the bias-add gradient accumulated.
		mmBwd()
	}
	return yb, bwd
}

// Parameters returns the trainable parameters. Weight is decayed; Bias is not.
func (l *Linear) Parameters() []*optim.Param {
	params := []*optim.Param{{
		Data:        l.Weight.Data,
		Grad:        l.Weight.Grad,
		DecayWeight: true,
	}}
	if l.Bias != nil {
		params = append(params, &optim.Param{
			Data:        l.Bias.Data,
			Grad:        l.Bias.Grad,
			DecayWeight: false,
		})
	}
	return params
}
