package model

import (
	"math/rand"

	"nanogpt/optim"
	"nanogpt/tensor"
)

// Embedding is a lookup table mapping integer indices to dense vectors.
type Embedding struct {
	Weight *tensor.Tensor // [NumEmbeddings, EmbeddingDim]
}

// NewEmbedding constructs an embedding table with weights ~ N(0, 0.02^2).
func NewEmbedding(numEmb, embDim int, rng *rand.Rand) *Embedding {
	w := tensor.NewWithGrad(numEmb, embDim)
	tensor.NormalFill(w, 0, 0.02, rng)
	return &Embedding{Weight: w}
}

// Forward returns a [len(idx), EmbeddingDim] tensor by copying the selected
// rows from Weight. Backward scatter-adds upstream gradients back into the
// embedding table (multiple occurrences of the same id accumulate).
func (e *Embedding) Forward(idx []int32) (*tensor.Tensor, tensor.BackwardFn) {
	numEmb := e.Weight.Shape[0]
	dim := e.Weight.Shape[1]
	n := len(idx)

	out := tensor.New(n, dim)
	out.Grad = make([]float32, len(out.Data))

	for i, id := range idx {
		if id < 0 || int(id) >= numEmb {
			panic("model.Embedding.Forward: index out of range")
		}
		src := e.Weight.Data[int(id)*dim : (int(id)+1)*dim]
		dst := out.Data[i*dim : (i+1)*dim]
		copy(dst, src)
	}

	// Capture idx by value (slice header is fine — caller should not mutate
	// during a forward/backward cycle).
	captured := idx
	bwd := func() {
		if e.Weight.Grad == nil {
			return
		}
		for i, id := range captured {
			gSrc := out.Grad[i*dim : (i+1)*dim]
			gDst := e.Weight.Grad[int(id)*dim : (int(id)+1)*dim]
			for j, g := range gSrc {
				gDst[j] += g
			}
		}
	}
	return out, bwd
}

// Parameters returns the embedding weight; embeddings are NOT weight-decayed
// in nanoGPT.
func (e *Embedding) Parameters() []*optim.Param {
	return []*optim.Param{{
		Data:        e.Weight.Data,
		Grad:        e.Weight.Grad,
		DecayWeight: false,
	}}
}
