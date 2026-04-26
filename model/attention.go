package model

import (
	"math/rand"

	"nanogpt/optim"
	"nanogpt/tensor"
)

// CausalSelfAttention implements the multi-head causal self-attention block
// used by GPT. The Q/K/V projections are fused into a single linear (CAttn)
// and the output is projected back with CProj (whose weight gets the depth-
// scaled initialization).
type CausalSelfAttention struct {
	CAttn  *Linear        // [3C, C] weight (out=3C, in=C)
	CProj  *Linear        // [C, C] weight — ScaledNormalFill
	Mask   *tensor.Tensor // [BlockSize, BlockSize] causal mask (0 / -1e9)
	NHead  int
	NEmbdg int
	DropP  float32
}

// NewCausalSelfAttention constructs the attention block per cfg.
func NewCausalSelfAttention(cfg GPTConfig, rng *rand.Rand) *CausalSelfAttention {
	if cfg.NEmbdg%cfg.NHead != 0 {
		panic("model.NewCausalSelfAttention: NEmbdg must be divisible by NHead")
	}
	cAttn := NewLinear(cfg.NEmbdg, 3*cfg.NEmbdg, cfg.Bias, rng)
	cProj := NewLinear(cfg.NEmbdg, cfg.NEmbdg, cfg.Bias, rng)
	// Depth-scaled init for the residual output projection's weight.
	tensor.ScaledNormalFill(cProj.Weight, 0.02, cfg.NLayer, rng)

	// Causal mask: 0 at [i,j] if j<=i, else -1e9.
	mask := tensor.New(cfg.BlockSize, cfg.BlockSize)
	for i := 0; i < cfg.BlockSize; i++ {
		for j := 0; j < cfg.BlockSize; j++ {
			if j > i {
				mask.Data[i*cfg.BlockSize+j] = -1e9
			}
		}
	}

	return &CausalSelfAttention{
		CAttn:  cAttn,
		CProj:  cProj,
		Mask:   mask,
		NHead:  cfg.NHead,
		NEmbdg: cfg.NEmbdg,
		DropP:  cfg.Dropout,
	}
}

// splitQKV splits a [B, T, 3C] tensor into three [B, T, C] tensors by copying.
// Using fresh tensors (rather than views) ensures that each backward path
// accumulates into the original qkv.Grad without aliasing.
func splitQKV(qkv *tensor.Tensor) (q, k, v *tensor.Tensor, bwd tensor.BackwardFn) {
	if len(qkv.Shape) != 3 {
		panic("model.splitQKV: expected 3-D [B, T, 3C] tensor")
	}
	B := qkv.Shape[0]
	T := qkv.Shape[1]
	threeC := qkv.Shape[2]
	if threeC%3 != 0 {
		panic("model.splitQKV: last dim must be divisible by 3")
	}
	C := threeC / 3

	q = tensor.New(B, T, C)
	k = tensor.New(B, T, C)
	v = tensor.New(B, T, C)
	q.Grad = make([]float32, len(q.Data))
	k.Grad = make([]float32, len(k.Data))
	v.Grad = make([]float32, len(v.Data))

	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			srcBase := (b*T + t) * threeC
			dstBase := (b*T + t) * C
			copy(q.Data[dstBase:dstBase+C], qkv.Data[srcBase:srcBase+C])
			copy(k.Data[dstBase:dstBase+C], qkv.Data[srcBase+C:srcBase+2*C])
			copy(v.Data[dstBase:dstBase+C], qkv.Data[srcBase+2*C:srcBase+3*C])
		}
	}

	bwd = func() {
		if qkv.Grad == nil {
			return
		}
		for b := 0; b < B; b++ {
			for t := 0; t < T; t++ {
				srcBase := (b*T + t) * threeC
				dstBase := (b*T + t) * C
				gq := q.Grad[dstBase : dstBase+C]
				gk := k.Grad[dstBase : dstBase+C]
				gv := v.Grad[dstBase : dstBase+C]
				gRow := qkv.Grad[srcBase : srcBase+threeC]
				for i := 0; i < C; i++ {
					gRow[i] += gq[i]
					gRow[C+i] += gk[i]
					gRow[2*C+i] += gv[i]
				}
			}
		}
	}
	return q, k, v, bwd
}

// transposeHeads reshapes + transposes [B, T, nHead, headSize] to
// [B, nHead, T, headSize]. A copy is required because BLAS needs contiguous
// memory for each head.
func transposeHeads(x *tensor.Tensor, B, T, nHead, headSize int) (*tensor.Tensor, tensor.BackwardFn) {
	if x.Numel() != B*T*nHead*headSize {
		panic("model.transposeHeads: element count mismatch")
	}
	out := tensor.New(B, nHead, T, headSize)
	out.Grad = make([]float32, len(out.Data))

	// Source stride (row-major [B, T, nHead, headSize]):
	//   flat(b, t, h, d) = ((b*T + t)*nHead + h)*headSize + d
	// Dest stride [B, nHead, T, headSize]:
	//   flat(b, h, t, d) = ((b*nHead + h)*T + t)*headSize + d
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < nHead; h++ {
				srcBase := ((b*T+t)*nHead + h) * headSize
				dstBase := ((b*nHead+h)*T + t) * headSize
				copy(out.Data[dstBase:dstBase+headSize], x.Data[srcBase:srcBase+headSize])
			}
		}
	}

	bwd := func() {
		if x.Grad == nil {
			return
		}
		for b := 0; b < B; b++ {
			for t := 0; t < T; t++ {
				for h := 0; h < nHead; h++ {
					srcBase := ((b*T+t)*nHead + h) * headSize
					dstBase := ((b*nHead+h)*T + t) * headSize
					src := x.Grad[srcBase : srcBase+headSize]
					dst := out.Grad[dstBase : dstBase+headSize]
					for i := 0; i < headSize; i++ {
						src[i] += dst[i]
					}
				}
			}
		}
	}
	return out, bwd
}

// transposeHeadsBack reshapes [B, nHead, T, headSize] to [B, T, nHead, headSize]
// (the inverse permutation of transposeHeads).
func transposeHeadsBack(x *tensor.Tensor, B, T, nHead, headSize int) (*tensor.Tensor, tensor.BackwardFn) {
	if x.Numel() != B*T*nHead*headSize {
		panic("model.transposeHeadsBack: element count mismatch")
	}
	out := tensor.New(B, T, nHead, headSize)
	out.Grad = make([]float32, len(out.Data))

	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < nHead; h++ {
				// src: [B, nHead, T, headSize]
				srcBase := ((b*nHead+h)*T + t) * headSize
				// dst: [B, T, nHead, headSize]
				dstBase := ((b*T+t)*nHead + h) * headSize
				copy(out.Data[dstBase:dstBase+headSize], x.Data[srcBase:srcBase+headSize])
			}
		}
	}

	bwd := func() {
		if x.Grad == nil {
			return
		}
		for b := 0; b < B; b++ {
			for t := 0; t < T; t++ {
				for h := 0; h < nHead; h++ {
					srcBase := ((b*nHead+h)*T + t) * headSize
					dstBase := ((b*T+t)*nHead + h) * headSize
					src := x.Grad[srcBase : srcBase+headSize]
					dst := out.Grad[dstBase : dstBase+headSize]
					for i := 0; i < headSize; i++ {
						src[i] += dst[i]
					}
				}
			}
		}
	}
	return out, bwd
}

// sliceMask returns a [T, T] view over the [BlockSize, BlockSize] mask at
// the top-left corner. The returned tensor shares the same Data slice so it
// is a read-only snapshot for this forward call.
func sliceMask(mask *tensor.Tensor, T int) *tensor.Tensor {
	block := mask.Shape[0]
	if T == block {
		return mask
	}
	if T > block {
		panic("model.sliceMask: T larger than mask block size")
	}
	sub := tensor.New(T, T)
	for i := 0; i < T; i++ {
		copy(sub.Data[i*T:(i+1)*T], mask.Data[i*block:i*block+T])
	}
	return sub
}

// Forward runs causal multi-head self-attention over x [B, T, C].
func (a *CausalSelfAttention) Forward(x *tensor.Tensor, training bool, rng *rand.Rand) (*tensor.Tensor, tensor.BackwardFn) {
	if len(x.Shape) != 3 {
		panic("model.CausalSelfAttention.Forward: expected [B, T, C] input")
	}
	B := x.Shape[0]
	T := x.Shape[1]
	C := x.Shape[2]
	if C != a.NEmbdg {
		panic("model.CausalSelfAttention.Forward: input C does not match NEmbdg")
	}
	headSize := C / a.NHead

	tape := make([]tensor.BackwardFn, 0, 8)

	// 1. qkv = CAttn(x) -> [B, T, 3C]
	qkv, attnBwd := a.CAttn.Forward(x)
	tape = append(tape, attnBwd)

	// 2. Split into q, k, v each [B, T, C].
	q, k, v, splitBwd := splitQKV(qkv)
	tape = append(tape, splitBwd)

	// 3. View as [B, T, nHead, headSize] then transpose to [B, nHead, T, headSize].
	qView := q.View(B, T, a.NHead, headSize)
	kView := k.View(B, T, a.NHead, headSize)
	vView := v.View(B, T, a.NHead, headSize)

	qH, qHBwd := transposeHeads(qView, B, T, a.NHead, headSize)
	tape = append(tape, qHBwd)
	kH, kHBwd := transposeHeads(kView, B, T, a.NHead, headSize)
	tape = append(tape, kHBwd)
	vH, vHBwd := transposeHeads(vView, B, T, a.NHead, headSize)
	tape = append(tape, vHBwd)

	// 4. Self-attention over the T slice of the causal mask.
	mask := sliceMask(a.Mask, T)
	attOut, attOutBwd := tensor.SelfAttentionForward(qH, kH, vH, mask, a.DropP, training, rng)
	tape = append(tape, attOutBwd)

	// 5. Transpose back [B, nHead, T, headSize] -> [B, T, nHead, headSize],
	//    then reshape to [B, T, C].
	back, backBwd := transposeHeadsBack(attOut, B, T, a.NHead, headSize)
	tape = append(tape, backBwd)
	y := back.View(B, T, C)

	// 6. Output projection -> [B, T, C].
	proj, projBwd := a.CProj.Forward(y)
	tape = append(tape, projBwd)

	// 7. Residual dropout.
	out, dropBwd := tensor.Dropout(proj, a.DropP, training, rng)
	tape = append(tape, dropBwd)

	bwd := func() {
		for i := len(tape) - 1; i >= 0; i-- {
			tape[i]()
		}
	}
	return out, bwd
}

// Parameters returns the trainable parameters of both inner linear layers.
func (a *CausalSelfAttention) Parameters() []*optim.Param {
	params := a.CAttn.Parameters()
	params = append(params, a.CProj.Parameters()...)
	return params
}
