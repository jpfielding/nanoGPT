package tensor

import (
	"math"
	"math/rand"

	"nanogpt/blas"
)

// SelfAttentionForward computes batched, multi-head, scaled dot-product
// attention with an additive mask.
//
// Shapes:
//
//	q, k, v: [B, nHead, T, headSize]
//	mask:    [T, T] (0 for allowed positions, large-negative for masked)
//	out:     [B, nHead, T, headSize]
//
// Per (b, h):
//
//	att = q @ k^T * (1/sqrt(headSize))      [T, T]
//	att += mask
//	att = softmax(att, dim=-1)
//	att = dropout(att, dropoutP)
//	out = att @ v                            [T, headSize]
func SelfAttentionForward(
	q, k, v *Tensor,
	mask *Tensor,
	dropoutP float32,
	training bool,
	rng *rand.Rand,
) (*Tensor, BackwardFn) {
	if len(q.Shape) != 4 || len(k.Shape) != 4 || len(v.Shape) != 4 {
		panic("tensor.SelfAttentionForward: q/k/v must be 4-D [B, H, T, D]")
	}
	B := q.Shape[0]
	H := q.Shape[1]
	T := q.Shape[2]
	D := q.Shape[3]
	if k.Shape[0] != B || k.Shape[1] != H || k.Shape[2] != T || k.Shape[3] != D {
		panic("tensor.SelfAttentionForward: k shape mismatch")
	}
	if v.Shape[0] != B || v.Shape[1] != H || v.Shape[2] != T || v.Shape[3] != D {
		panic("tensor.SelfAttentionForward: v shape mismatch")
	}
	if mask == nil || len(mask.Shape) != 2 || mask.Shape[0] != T || mask.Shape[1] != T {
		panic("tensor.SelfAttentionForward: mask must be [T, T]")
	}

	out := New(B, H, T, D)
	out.Grad = make([]float32, len(out.Data))

	// attProb holds the softmaxed (and dropout-scaled) attention weights used
	// by the backward. dropMask stores the per-element scale (0 or 1/(1-p))
	// applied by dropout; nil means dropout was a no-op.
	attProb := make([]float32, B*H*T*T)
	var dropMask []float32
	applyDropout := training && dropoutP > 0
	if applyDropout {
		dropMask = make([]float32, B*H*T*T)
	}

	scale := float32(1.0 / math.Sqrt(float64(D)))
	pf := float64(dropoutP)
	dropScale := float32(1.0 / (1.0 - float64(dropoutP)))

	qhSize := T * D
	vhSize := T * D
	ohSize := T * D
	ahSize := T * T

	// Per-head scratch for att pre-softmax.
	att := make([]float32, T*T)

	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			qBase := ((b * H) + h) * qhSize
			kBase := ((b * H) + h) * qhSize
			vBase := ((b * H) + h) * vhSize
			oBase := ((b * H) + h) * ohSize
			aBase := ((b * H) + h) * ahSize

			qSlice := q.Data[qBase : qBase+qhSize]
			kSlice := k.Data[kBase : kBase+qhSize]
			vSlice := v.Data[vBase : vBase+vhSize]

			// att = q @ k^T * scale  ([T,D] @ [D,T] = [T,T])
			blas.Sgemm(false, true, T, T, D,
				scale, qSlice, D,
				kSlice, D,
				0.0, att, T)

			// Add mask.
			for i := 0; i < T; i++ {
				row := att[i*T : (i+1)*T]
				mRow := mask.Data[i*T : (i+1)*T]
				for j := 0; j < T; j++ {
					row[j] += mRow[j]
				}
			}

			// Row-wise softmax (numerically stable).
			for i := 0; i < T; i++ {
				row := att[i*T : (i+1)*T]
				maxv := row[0]
				for _, v := range row[1:] {
					if v > maxv {
						maxv = v
					}
				}
				var sum float64
				for j, v := range row {
					e := math.Exp(float64(v - maxv))
					row[j] = float32(e)
					sum += e
				}
				inv := float32(1.0 / sum)
				for j := range row {
					row[j] *= inv
				}
			}

			// Dropout (on the attention weights themselves).
			if applyDropout {
				dm := dropMask[aBase : aBase+ahSize]
				for i := range att {
					if rng.Float64() < pf {
						dm[i] = 0
						att[i] = 0
					} else {
						dm[i] = dropScale
						att[i] *= dropScale
					}
				}
			}

			// Save attProb for backward.
			copy(attProb[aBase:aBase+ahSize], att)

			// out = att @ v  ([T,T] @ [T,D] = [T,D])
			oSlice := out.Data[oBase : oBase+ohSize]
			blas.Sgemm(false, false, T, D, T,
				1.0, att, T,
				vSlice, D,
				0.0, oSlice, D)
		}
	}

	bw := func() {
		// Scratch buffers reused across (b, h).
		dAtt := make([]float32, T*T)

		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				qBase := ((b * H) + h) * qhSize
				kBase := ((b * H) + h) * qhSize
				vBase := ((b * H) + h) * vhSize
				oBase := ((b * H) + h) * ohSize
				aBase := ((b * H) + h) * ahSize

				qSlice := q.Data[qBase : qBase+qhSize]
				kSlice := k.Data[kBase : kBase+qhSize]
				vSlice := v.Data[vBase : vBase+vhSize]
				doSlice := out.Grad[oBase : oBase+ohSize]
				att := attProb[aBase : aBase+ahSize]

				// dv[b,h] += att^T @ dout  ([T,T]^T @ [T,D] = [T,D])
				if v.Grad != nil {
					dvSlice := v.Grad[vBase : vBase+vhSize]
					blas.Sgemm(true, false, T, D, T,
						1.0, att, T,
						doSlice, D,
						1.0, dvSlice, D)
				}

				// dAtt = dout @ v^T  ([T,D] @ [D,T] = [T,T])
				blas.Sgemm(false, true, T, T, D,
					1.0, doSlice, D,
					vSlice, D,
					0.0, dAtt, T)

				// Undo dropout on dAtt: multiply by the same per-element scale
				// (0 for dropped, 1/(1-p) for kept).
				if applyDropout {
					dm := dropMask[aBase : aBase+ahSize]
					for i := range dAtt {
						dAtt[i] *= dm[i]
					}
				}

				// Softmax backward, row-wise:
				//   dAttRaw[i] = att[i] * (dAtt[i] - sum_j(att[j]*dAtt[j]))
				for i := 0; i < T; i++ {
					row := att[i*T : (i+1)*T]
					dRow := dAtt[i*T : (i+1)*T]
					var dot float64
					for j := 0; j < T; j++ {
						dot += float64(row[j]) * float64(dRow[j])
					}
					dotF := float32(dot)
					for j := 0; j < T; j++ {
						dRow[j] = row[j] * (dRow[j] - dotF)
					}
				}

				// Scale by 1/sqrt(D) (matches forward's pre-softmax scaling).
				for i := range dAtt {
					dAtt[i] *= scale
				}

				// dq[b,h] += dAttRaw @ k       ([T,T] @ [T,D] = [T,D])
				if q.Grad != nil {
					dqSlice := q.Grad[qBase : qBase+qhSize]
					blas.Sgemm(false, false, T, D, T,
						1.0, dAtt, T,
						kSlice, D,
						1.0, dqSlice, D)
				}
				// dk[b,h] += dAttRaw^T @ q     ([T,T]^T @ [T,D] = [T,D])
				if k.Grad != nil {
					dkSlice := k.Grad[kBase : kBase+qhSize]
					blas.Sgemm(true, false, T, D, T,
						1.0, dAtt, T,
						qSlice, D,
						1.0, dkSlice, D)
				}
			}
		}
	}

	return out, bw
}
