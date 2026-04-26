package model

import (
	"nanogpt/optim"
	"nanogpt/tensor"
)

// LayerNorm applies layer normalization over the last dimension with an
// affine transform (weight, optional bias).
type LayerNorm struct {
	Weight *tensor.Tensor // [NormDim] — initialized to ones
	Bias   *tensor.Tensor // [NormDim] — initialized to zeros (nil if !bias)
	Eps    float32
}

// NewLayerNorm constructs a LayerNorm layer for the given trailing dim.
// Weight is initialized to ones; Bias is zero-initialized when bias=true,
// otherwise it is nil.
func NewLayerNorm(dim int, bias bool) *LayerNorm {
	w := tensor.NewWithGrad(dim)
	tensor.OneFill(w)
	ln := &LayerNorm{Weight: w, Eps: 1e-5}
	if bias {
		b := tensor.NewWithGrad(dim)
		tensor.ZeroFill(b)
		ln.Bias = b
	}
	return ln
}

// Forward normalizes x over its last dim and applies the affine transform.
func (ln *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, tensor.BackwardFn) {
	return tensor.LayerNormForward(x, ln.Weight, ln.Bias, ln.Eps)
}

// Parameters returns the trainable parameters. Neither weight nor bias is
// weight-decayed.
func (ln *LayerNorm) Parameters() []*optim.Param {
	params := []*optim.Param{{
		Data:        ln.Weight.Data,
		Grad:        ln.Weight.Grad,
		DecayWeight: false,
	}}
	if ln.Bias != nil {
		params = append(params, &optim.Param{
			Data:        ln.Bias.Data,
			Grad:        ln.Bias.Grad,
			DecayWeight: false,
		})
	}
	return params
}
