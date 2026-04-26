package tensor

import "nanogpt/blas"

// MatMul computes out = x @ w where w is 2-D of shape [K, N] and x is either
// 2-D [M, K] or 3-D [B, M, K]. The output shape mirrors x with the last
// dimension replaced by N.
//
// Backward (per leading batch slice):
//
//	dx += dout @ w^T
//	dw += x^T  @ dout   (accumulated across all batches)
func MatMul(x, w *Tensor) (*Tensor, BackwardFn) {
	if len(w.Shape) != 2 {
		panic("tensor.MatMul: w must be 2-D")
	}
	if len(x.Shape) < 2 {
		panic("tensor.MatMul: x must be at least 2-D")
	}
	K := w.Shape[0]
	N := w.Shape[1]
	M := x.Shape[len(x.Shape)-2]
	if x.Shape[len(x.Shape)-1] != K {
		panic("tensor.MatMul: inner dim mismatch")
	}

	// Flatten leading dims into B.
	B := 1
	for i := 0; i < len(x.Shape)-2; i++ {
		B *= x.Shape[i]
	}

	outShape := append([]int(nil), x.Shape...)
	outShape[len(outShape)-1] = N
	out := New(outShape...)
	out.Grad = make([]float32, len(out.Data))

	// Forward: for each batch slice call sgemm (beta=0: overwrite).
	for b := 0; b < B; b++ {
		xSlice := x.Data[b*M*K : (b+1)*M*K]
		oSlice := out.Data[b*M*N : (b+1)*M*N]
		blas.Sgemm(false, false, M, N, K,
			1.0, xSlice, K,
			w.Data, N,
			0.0, oSlice, N)
	}

	bw := func() {
		// dx[b] = dout[b] @ w^T  (shape [M,N] @ [N,K] = [M,K])
		if x.Grad != nil {
			for b := 0; b < B; b++ {
				doSlice := out.Grad[b*M*N : (b+1)*M*N]
				dxSlice := x.Grad[b*M*K : (b+1)*M*K]
				blas.Sgemm(false, true, M, K, N,
					1.0, doSlice, N,
					w.Data, N,
					1.0, dxSlice, K)
			}
		}
		// dw += sum_b x[b]^T @ dout[b]  (shape [K,M] @ [M,N] = [K,N])
		if w.Grad != nil {
			for b := 0; b < B; b++ {
				xSlice := x.Data[b*M*K : (b+1)*M*K]
				doSlice := out.Grad[b*M*N : (b+1)*M*N]
				blas.Sgemm(true, false, K, N, M,
					1.0, xSlice, K,
					doSlice, N,
					1.0, w.Grad, N)
			}
		}
	}
	return out, bw
}

// MatMulTransB computes out = x @ w^T for 2-D operands: x [M, K], w [N, K].
// The result is [M, N]. This is the shape used when Q @ K^T is computed in
// self-attention after transposing the last two dims.
//
// Backward:
//
//	dx += dout   @ w      (both not transposed, shape [M,N] @ [N,K] = [M,K])
//	dw += dout^T @ x      (shape [N,M] @ [M,K] = [N,K])
func MatMulTransB(x, w *Tensor) (*Tensor, BackwardFn) {
	if len(x.Shape) != 2 || len(w.Shape) != 2 {
		panic("tensor.MatMulTransB: x and w must be 2-D")
	}
	M := x.Shape[0]
	K := x.Shape[1]
	N := w.Shape[0]
	if w.Shape[1] != K {
		panic("tensor.MatMulTransB: inner dim mismatch")
	}

	out := New(M, N)
	out.Grad = make([]float32, len(out.Data))

	blas.Sgemm(false, true, M, N, K,
		1.0, x.Data, K,
		w.Data, K,
		0.0, out.Data, N)

	bw := func() {
		if x.Grad != nil {
			// dx += dout @ w  ([M,N] @ [N,K] = [M,K])
			blas.Sgemm(false, false, M, K, N,
				1.0, out.Grad, N,
				w.Data, K,
				1.0, x.Grad, K)
		}
		if w.Grad != nil {
			// dw += dout^T @ x  ([N,M] @ [M,K] = [N,K])
			blas.Sgemm(true, false, N, K, M,
				1.0, out.Grad, N,
				x.Data, K,
				1.0, w.Grad, K)
		}
	}
	return out, bw
}
