//go:build !darwin && !linux

package blas

// Sgemm computes C = alpha*A*B + beta*C in row-major layout using a naive
// O(M*N*K) triple loop. Correct but slow; intended only for portability testing.
func Sgemm(transA, transB bool, M, N, K int,
	alpha float32, A []float32, lda int,
	B []float32, ldb int,
	beta float32, C []float32, ldc int) {
	if M == 0 || N == 0 {
		return
	}

	// Scale C by beta first: C = beta*C.
	for i := 0; i < M; i++ {
		row := C[i*ldc : i*ldc+N]
		switch beta {
		case 1:
			// no-op
		case 0:
			for j := range row {
				row[j] = 0
			}
		default:
			for j := range row {
				row[j] *= beta
			}
		}
	}

	if K == 0 || alpha == 0 {
		return
	}

	// aAt returns A[i, k] honoring transA.
	aAt := func(i, k int) float32 {
		if transA {
			// A is (K x M), row-major, stride lda.
			return A[k*lda+i]
		}
		return A[i*lda+k]
	}
	// bAt returns B[k, j] honoring transB.
	bAt := func(k, j int) float32 {
		if transB {
			// B is (N x K), row-major, stride ldb.
			return B[j*ldb+k]
		}
		return B[k*ldb+j]
	}

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += aAt(i, k) * bAt(k, j)
			}
			C[i*ldc+j] += alpha * sum
		}
	}
}

// Saxpy computes y += alpha*x over n elements using a simple loop.
func Saxpy(n int, alpha float32, x, y []float32) {
	if n <= 0 || alpha == 0 {
		return
	}
	if n > len(x) {
		n = len(x)
	}
	if n > len(y) {
		n = len(y)
	}
	for i := 0; i < n; i++ {
		y[i] += alpha * x[i]
	}
}
