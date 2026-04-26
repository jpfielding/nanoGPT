package model

import (
	"math/rand"

	"nanogpt/optim"
	"nanogpt/tensor"
)

// MLP is the position-wise feed-forward block: two linear layers with a
// GeLU between them and dropout on the output. The output projection's
// weight uses the depth-scaled init to match nanoGPT / GPT-2.
type MLP struct {
	FC1   *Linear // [4C, C] — out=4C, in=C
	FC2   *Linear // [C, 4C] — out=C, in=4C, ScaledNormalFill
	DropP float32
}

// NewMLP constructs an MLP per cfg.
func NewMLP(cfg GPTConfig, rng *rand.Rand) *MLP {
	fc1 := NewLinear(cfg.NEmbdg, 4*cfg.NEmbdg, cfg.Bias, rng)
	fc2 := NewLinear(4*cfg.NEmbdg, cfg.NEmbdg, cfg.Bias, rng)
	tensor.ScaledNormalFill(fc2.Weight, 0.02, cfg.NLayer, rng)
	return &MLP{FC1: fc1, FC2: fc2, DropP: cfg.Dropout}
}

// Forward computes FC2(GeLU(FC1(x))) with dropout on the output.
func (m *MLP) Forward(x *tensor.Tensor, training bool, rng *rand.Rand) (*tensor.Tensor, tensor.BackwardFn) {
	tape := make([]tensor.BackwardFn, 0, 4)

	h1, b1 := m.FC1.Forward(x)
	tape = append(tape, b1)

	h2, b2 := tensor.GeLU(h1)
	tape = append(tape, b2)

	h3, b3 := m.FC2.Forward(h2)
	tape = append(tape, b3)

	out, b4 := tensor.Dropout(h3, m.DropP, training, rng)
	tape = append(tape, b4)

	bwd := func() {
		for i := len(tape) - 1; i >= 0; i-- {
			tape[i]()
		}
	}
	return out, bwd
}

// Parameters returns the concatenated parameters of both linear layers.
func (m *MLP) Parameters() []*optim.Param {
	params := m.FC1.Parameters()
	params = append(params, m.FC2.Parameters()...)
	return params
}
