package tensor

import (
	"math"
	"math/rand"
)

// AddInplace computes dst += src element-wise. Shapes must have equal element
// counts; broadcasting is not supported. No gradient tracking.
func AddInplace(dst, src *Tensor) {
	if len(dst.Data) != len(src.Data) {
		panic("tensor.AddInplace: length mismatch")
	}
	for i, v := range src.Data {
		dst.Data[i] += v
	}
}

// Add returns a+b (element-wise) with a backward that accumulates the upstream
// gradient into both a.Grad and b.Grad.
func Add(a, b *Tensor) (*Tensor, BackwardFn) {
	if len(a.Data) != len(b.Data) {
		panic("tensor.Add: length mismatch")
	}
	out := New(a.Shape...)
	for i, v := range a.Data {
		out.Data[i] = v + b.Data[i]
	}
	out.Grad = make([]float32, len(out.Data))

	bw := func() {
		if a.Grad != nil {
			for i, g := range out.Grad {
				a.Grad[i] += g
			}
		}
		if b.Grad != nil {
			for i, g := range out.Grad {
				b.Grad[i] += g
			}
		}
	}
	return out, bw
}

// AddBias adds a 1-D bias onto the last dimension of x (broadcasting across
// all leading dimensions). x has shape [*, N]; bias has shape [N].
func AddBias(x, bias *Tensor) (*Tensor, BackwardFn) {
	if len(bias.Shape) != 1 {
		panic("tensor.AddBias: bias must be 1-D")
	}
	n := bias.Shape[0]
	if len(x.Shape) == 0 || x.Shape[len(x.Shape)-1] != n {
		panic("tensor.AddBias: last dim of x must match bias length")
	}

	out := New(x.Shape...)
	rows := len(x.Data) / n
	for r := 0; r < rows; r++ {
		base := r * n
		for j := 0; j < n; j++ {
			out.Data[base+j] = x.Data[base+j] + bias.Data[j]
		}
	}
	out.Grad = make([]float32, len(out.Data))

	bw := func() {
		if x.Grad != nil {
			for i, g := range out.Grad {
				x.Grad[i] += g
			}
		}
		if bias.Grad != nil {
			for r := 0; r < rows; r++ {
				base := r * n
				for j := 0; j < n; j++ {
					bias.Grad[j] += out.Grad[base+j]
				}
			}
		}
	}
	return out, bw
}

// geluConst = sqrt(2/pi), precomputed in float64 for accuracy.
var geluConst = math.Sqrt(2.0 / math.Pi)

// GeLU applies the tanh-approximation GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
// Saves tanhVal and the inner kappa derivative to avoid recomputing x^3 in backward.
func GeLU(x *Tensor) (*Tensor, BackwardFn) {
	n := len(x.Data)
	out := New(x.Shape...)
	out.Grad = make([]float32, n)

	// Cache tanh(kappa) and the derivative of kappa wrt x per element.
	// kappa = sqrt(2/pi) * (x + 0.044715*x^3)
	// kappaDeriv = sqrt(2/pi) * (1 + 3*0.044715*x^2)
	tanhVals := make([]float64, n)
	kappaDeriv := make([]float64, n)

	for i, v := range x.Data {
		xf := float64(v)
		xf2 := xf * xf
		xf3 := xf2 * xf
		kappa := geluConst * (xf + 0.044715*xf3)
		th := math.Tanh(kappa)
		tanhVals[i] = th
		kappaDeriv[i] = geluConst * (1.0 + 3.0*0.044715*xf2)
		out.Data[i] = float32(0.5 * xf * (1.0 + th))
	}

	bw := func() {
		if x.Grad == nil {
			return
		}
		for i, g := range out.Grad {
			xf := float64(x.Data[i])
			th := tanhVals[i]
			// d/dx GeLU(x) = 0.5*(1+th) + 0.5*x*(1-th^2)*kappaDeriv
			deriv := 0.5*(1.0+th) + 0.5*xf*(1.0-th*th)*kappaDeriv[i]
			x.Grad[i] += float32(float64(g) * deriv)
		}
	}
	return out, bw
}

// Dropout applies inverted dropout. With probability p each element is zeroed,
// otherwise it is scaled by 1/(1-p). If training is false or p==0 it is a
// no-op: the returned tensor is x itself and the backward is a no-op.
func Dropout(x *Tensor, p float32, training bool, rng *rand.Rand) (*Tensor, BackwardFn) {
	if !training || p == 0 {
		return x, func() {}
	}
	if p < 0 || p >= 1 {
		panic("tensor.Dropout: p must be in [0, 1)")
	}

	n := len(x.Data)
	out := New(x.Shape...)
	out.Grad = make([]float32, n)
	mask := make([]float32, n) // 0 or scale
	scale := 1.0 / (1.0 - float64(p))
	scaleF := float32(scale)
	pf := float64(p)

	for i, v := range x.Data {
		if rng.Float64() < pf {
			mask[i] = 0
			out.Data[i] = 0
		} else {
			mask[i] = scaleF
			out.Data[i] = v * scaleF
		}
	}

	bw := func() {
		if x.Grad == nil {
			return
		}
		for i, g := range out.Grad {
			x.Grad[i] += g * mask[i]
		}
	}
	return out, bw
}
