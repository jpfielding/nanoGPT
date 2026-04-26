//go:build darwin

package blas

/*
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

static void nanogpt_sgemm(int transA, int transB,
                          int M, int N, int K,
                          float alpha, const float *A, int lda,
                          const float *B, int ldb,
                          float beta, float *C, int ldc) {
    enum CBLAS_TRANSPOSE ta = transA ? CblasTrans : CblasNoTrans;
    enum CBLAS_TRANSPOSE tb = transB ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasRowMajor, ta, tb,
                M, N, K,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
}

static void nanogpt_saxpy(int n, float alpha, const float *x, float *y) {
    cblas_saxpy(n, alpha, x, 1, y, 1);
}
*/
import "C"

import "unsafe"

// Sgemm computes C = alpha*A*B + beta*C via Accelerate's cblas_sgemm (row-major).
func Sgemm(transA, transB bool, M, N, K int,
	alpha float32, A []float32, lda int,
	B []float32, ldb int,
	beta float32, C []float32, ldc int) {
	if M == 0 || N == 0 {
		return
	}
	if len(C) == 0 {
		return
	}
	// When K == 0, cblas_sgemm still needs to scale C by beta; Accelerate handles this.
	if K > 0 && (len(A) == 0 || len(B) == 0) {
		return
	}

	var aPtr, bPtr *C.float
	if len(A) > 0 {
		aPtr = (*C.float)(unsafe.Pointer(&A[0]))
	}
	if len(B) > 0 {
		bPtr = (*C.float)(unsafe.Pointer(&B[0]))
	}
	cPtr := (*C.float)(unsafe.Pointer(&C[0]))

	var ta, tb C.int
	if transA {
		ta = 1
	}
	if transB {
		tb = 1
	}

	C.nanogpt_sgemm(ta, tb,
		C.int(M), C.int(N), C.int(K),
		C.float(alpha), aPtr, C.int(lda),
		bPtr, C.int(ldb),
		C.float(beta), cPtr, C.int(ldc))
}

// Saxpy computes y += alpha*x over n elements via Accelerate's cblas_saxpy.
func Saxpy(n int, alpha float32, x, y []float32) {
	if n <= 0 || len(x) == 0 || len(y) == 0 {
		return
	}
	C.nanogpt_saxpy(C.int(n), C.float(alpha),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])))
}
