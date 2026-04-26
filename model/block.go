package model

import (
	"math/rand"

	"nanogpt/optim"
	"nanogpt/tensor"
)

// Block is a pre-norm transformer block:
//
//	x = x + Attn(LN1(x))
//	x = x + MLP(LN2(x))
type Block struct {
	LN1  *LayerNorm
	Attn *CausalSelfAttention
	LN2  *LayerNorm
	MLP  *MLP
}

// NewBlock constructs a transformer block per cfg.
func NewBlock(cfg GPTConfig, rng *rand.Rand) *Block {
	return &Block{
		LN1:  NewLayerNorm(cfg.NEmbdg, cfg.Bias),
		Attn: NewCausalSelfAttention(cfg, rng),
		LN2:  NewLayerNorm(cfg.NEmbdg, cfg.Bias),
		MLP:  NewMLP(cfg, rng),
	}
}

// Forward runs the block with pre-norm residual connections.
func (b *Block) Forward(x *tensor.Tensor, training bool, rng *rand.Rand) (*tensor.Tensor, tensor.BackwardFn) {
	tape := make([]tensor.BackwardFn, 0, 6)

	// Attn branch.
	n1, ln1Bwd := b.LN1.Forward(x)
	tape = append(tape, ln1Bwd)
	a, aBwd := b.Attn.Forward(n1, training, rng)
	tape = append(tape, aBwd)
	x2, add1Bwd := tensor.Add(x, a)
	tape = append(tape, add1Bwd)

	// MLP branch.
	n2, ln2Bwd := b.LN2.Forward(x2)
	tape = append(tape, ln2Bwd)
	m, mBwd := b.MLP.Forward(n2, training, rng)
	tape = append(tape, mBwd)
	out, add2Bwd := tensor.Add(x2, m)
	tape = append(tape, add2Bwd)

	bwd := func() {
		for i := len(tape) - 1; i >= 0; i-- {
			tape[i]()
		}
	}
	return out, bwd
}

// Parameters collects the parameters of every sub-layer.
func (b *Block) Parameters() []*optim.Param {
	params := b.LN1.Parameters()
	params = append(params, b.Attn.Parameters()...)
	params = append(params, b.LN2.Parameters()...)
	params = append(params, b.MLP.Parameters()...)
	return params
}
