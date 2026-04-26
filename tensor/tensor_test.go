package tensor_test

import (
	"math"
	"math/rand"
	"testing"

	"nanogpt/tensor"
)

// gradCheck verifies that the analytical backward pass produced by fwd matches
// a central finite-difference estimate of the gradient.
//
// The scalar loss used for finite differencing is SumAll(out). Equivalently,
// the upstream gradient is all-ones, which is what we seed into out.Grad
// before invoking the analytical backward.
func gradCheck(
	t *testing.T,
	name string,
	inputs []*tensor.Tensor,
	fwd func() (*tensor.Tensor, tensor.BackwardFn),
	eps, tol float32,
) {
	t.Helper()

	// Analytical gradient: forward, seed dOut=1, backward.
	for _, in := range inputs {
		in.ZeroGrad()
	}
	out, bw := fwd()
	for i := range out.Grad {
		out.Grad[i] = 1
	}
	bw()

	// Snapshot analytical grads before we start perturbing inputs.
	analytical := make([][]float32, len(inputs))
	for i, in := range inputs {
		analytical[i] = append([]float32(nil), in.Grad...)
	}

	epsF := float64(eps)

	for idx, in := range inputs {
		numerical := make([]float32, len(in.Data))
		for i := range in.Data {
			orig := in.Data[i]

			in.Data[i] = orig + eps
			outPlus, _ := fwd()
			lossPlus := float64(tensor.SumAll(outPlus))

			in.Data[i] = orig - eps
			outMinus, _ := fwd()
			lossMinus := float64(tensor.SumAll(outMinus))

			in.Data[i] = orig
			numerical[i] = float32((lossPlus - lossMinus) / (2 * epsF))
		}

		checkClose(t, name, idx, numerical, analytical[idx], tol)
	}
}

// checkClose compares numerical vs analytical gradients using a mixed absolute
// and relative tolerance. Reports at most a few mismatches so failures stay
// readable.
func checkClose(t *testing.T, name string, inputIdx int, numerical, analytical []float32, tol float32) {
	t.Helper()
	if len(numerical) != len(analytical) {
		t.Fatalf("%s input %d: length mismatch num=%d ana=%d", name, inputIdx, len(numerical), len(analytical))
	}

	var maxAbsDiff, maxAbsNum float64
	for i := range numerical {
		d := math.Abs(float64(numerical[i]) - float64(analytical[i]))
		if d > maxAbsDiff {
			maxAbsDiff = d
		}
		if a := math.Abs(float64(numerical[i])); a > maxAbsNum {
			maxAbsNum = a
		}
	}

	rel := maxAbsDiff / (maxAbsNum + 1e-8)
	if rel > float64(tol) {
		// Surface a handful of worst offenders for debugging.
		reported := 0
		for i := range numerical {
			d := math.Abs(float64(numerical[i]) - float64(analytical[i]))
			if d > float64(tol)*(math.Abs(float64(numerical[i]))+1e-3) {
				t.Logf("%s input %d [%d]: numerical=%g analytical=%g diff=%g",
					name, inputIdx, i, numerical[i], analytical[i], d)
				reported++
				if reported >= 5 {
					break
				}
			}
		}
		t.Fatalf("%s input %d: max|num-ana|=%g max|num|=%g rel=%g > tol=%g",
			name, inputIdx, maxAbsDiff, maxAbsNum, rel, tol)
	}
}

// randTensor fills a grad-tracked tensor with Uniform(lo, hi) values from r.
func randTensor(r *rand.Rand, lo, hi float32, shape ...int) *tensor.Tensor {
	t := tensor.NewWithGrad(shape...)
	span := hi - lo
	for i := range t.Data {
		t.Data[i] = lo + r.Float32()*span
	}
	return t
}

const (
	gcEps = float32(1e-3)
	gcTol = float32(5e-3)
)

func TestGeLUGrad(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	x := randTensor(r, -2, 2, 4, 8)

	fwd := func() (*tensor.Tensor, tensor.BackwardFn) {
		x.ZeroGrad()
		return tensor.GeLU(x)
	}
	gradCheck(t, "GeLU", []*tensor.Tensor{x}, fwd, gcEps, gcTol)
}

func TestLayerNormGrad(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	x := randTensor(r, -1, 1, 2, 6)
	weight := randTensor(r, -1, 1, 6)
	bias := randTensor(r, -1, 1, 6)

	fwd := func() (*tensor.Tensor, tensor.BackwardFn) {
		x.ZeroGrad()
		weight.ZeroGrad()
		bias.ZeroGrad()
		return tensor.LayerNormForward(x, weight, bias, 1e-5)
	}
	gradCheck(t, "LayerNorm", []*tensor.Tensor{x, weight, bias}, fwd, gcEps, gcTol)
}

func TestMatMulGrad(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	// MatMulTransB: x [M, K], w [N, K] -> out [M, N]
	x := randTensor(r, -1, 1, 3, 4)
	w := randTensor(r, -1, 1, 5, 4)

	fwd := func() (*tensor.Tensor, tensor.BackwardFn) {
		x.ZeroGrad()
		w.ZeroGrad()
		return tensor.MatMulTransB(x, w)
	}
	gradCheck(t, "MatMulTransB", []*tensor.Tensor{x, w}, fwd, gcEps, gcTol)
}

func TestCrossEntropyGrad(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	logits := randTensor(r, -1, 1, 4, 8)
	targets := []int32{1, 3, 0, 2}

	// Analytical grad.
	logits.ZeroGrad()
	_, bw := tensor.CrossEntropyLoss(logits, targets)
	bw()
	analytical := append([]float32(nil), logits.Grad...)

	// Numerical grad via central differences against the scalar loss directly.
	numerical := make([]float32, len(logits.Data))
	epsF := float64(gcEps)
	for i := range logits.Data {
		orig := logits.Data[i]

		logits.Data[i] = orig + gcEps
		lossPlus, _ := tensor.CrossEntropyLoss(logits, targets)

		logits.Data[i] = orig - gcEps
		lossMinus, _ := tensor.CrossEntropyLoss(logits, targets)

		logits.Data[i] = orig
		numerical[i] = float32((float64(lossPlus) - float64(lossMinus)) / (2 * epsF))
	}

	checkClose(t, "CrossEntropy", 0, numerical, analytical, gcTol)
}

func TestAddBiasGrad(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	x := randTensor(r, -1, 1, 2, 3, 4)
	bias := randTensor(r, -1, 1, 4)

	fwd := func() (*tensor.Tensor, tensor.BackwardFn) {
		x.ZeroGrad()
		bias.ZeroGrad()
		return tensor.AddBias(x, bias)
	}
	gradCheck(t, "AddBias", []*tensor.Tensor{x, bias}, fwd, gcEps, gcTol)
}

func TestSelfAttentionGrad(t *testing.T) {
	r := rand.New(rand.NewSource(42))

	const (
		B        = 1
		nHead    = 2
		T        = 4
		headSize = 4
	)

	q := randTensor(r, -1, 1, B, nHead, T, headSize)
	k := randTensor(r, -1, 1, B, nHead, T, headSize)
	v := randTensor(r, -1, 1, B, nHead, T, headSize)

	// Causal mask: 0 on/below diagonal, -1e9 above.
	mask := tensor.New(T, T)
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			if j > i {
				mask.Data[i*T+j] = -1e9
			}
		}
	}

	fwd := func() (*tensor.Tensor, tensor.BackwardFn) {
		q.ZeroGrad()
		k.ZeroGrad()
		v.ZeroGrad()
		// Dropout disabled: training=false, p=0, rng unused.
		return tensor.SelfAttentionForward(q, k, v, mask, 0, false, nil)
	}
	gradCheck(t, "SelfAttention", []*tensor.Tensor{q, k, v}, fwd, gcEps, gcTol)
}
