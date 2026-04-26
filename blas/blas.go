// Package blas provides BLAS SGEMM via CGo (Accelerate on macOS, OpenBLAS on Linux)
// with a pure-Go fallback for other platforms.
//
// All platform files in this package implement the same two functions:
//
//	Sgemm(transA, transB bool, M, N, K int,
//	      alpha float32, A []float32, lda int,
//	      B []float32, ldb int,
//	      beta float32, C []float32, ldc int)
//
//	   Computes C = alpha*A*B + beta*C where A is (M x K), B is (K x N), and
//	   C is (M x N), all stored in row-major order. If transA/transB is true,
//	   that operand is transposed before the multiply. lda/ldb/ldc are the
//	   row strides (leading dimensions) of A, B, C respectively.
//
//	Saxpy(n int, alpha float32, x, y []float32)
//
//	   Computes Y += alpha*X elementwise over n elements (unit stride).
package blas
