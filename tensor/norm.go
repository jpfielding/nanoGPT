package tensor

import "math"

// LayerNormForward normalizes the last dimension of x and applies an affine
// transform with weight (required) and bias (optional, may be nil).
// x: [*, C]; weight: [C]; bias: [C] or nil.
//
// Forward (per row):
//
//	mean = sum(x)/C
//	var  = sum((x-mean)^2)/C
//	rstd = 1/sqrt(var+eps)
//	xHat = (x-mean) * rstd
//	out  = xHat*weight + bias
func LayerNormForward(x, weight, bias *Tensor, eps float32) (*Tensor, BackwardFn) {
	if len(x.Shape) == 0 {
		panic("tensor.LayerNormForward: x must have at least 1 dim")
	}
	c := x.Shape[len(x.Shape)-1]
	if len(weight.Shape) != 1 || weight.Shape[0] != c {
		panic("tensor.LayerNormForward: weight shape must be [C]")
	}
	if bias != nil && (len(bias.Shape) != 1 || bias.Shape[0] != c) {
		panic("tensor.LayerNormForward: bias shape must be [C] or nil")
	}

	rows := len(x.Data) / c
	out := New(x.Shape...)
	out.Grad = make([]float32, len(out.Data))

	// Save per-row mean and rstd, and per-element xHat for backward.
	means := make([]float64, rows)
	rstds := make([]float64, rows)
	xHat := make([]float32, len(x.Data))

	invC := 1.0 / float64(c)
	epsF := float64(eps)

	for r := 0; r < rows; r++ {
		base := r * c
		// mean
		var sum float64
		for i := 0; i < c; i++ {
			sum += float64(x.Data[base+i])
		}
		mean := sum * invC
		// variance
		var vsum float64
		for i := 0; i < c; i++ {
			d := float64(x.Data[base+i]) - mean
			vsum += d * d
		}
		variance := vsum * invC
		rstd := 1.0 / math.Sqrt(variance+epsF)
		means[r] = mean
		rstds[r] = rstd

		for i := 0; i < c; i++ {
			h := (float64(x.Data[base+i]) - mean) * rstd
			xHat[base+i] = float32(h)
			o := h * float64(weight.Data[i])
			if bias != nil {
				o += float64(bias.Data[i])
			}
			out.Data[base+i] = float32(o)
		}
	}

	bw := func() {
		// Gradient wrt weight and bias: sum over all rows.
		if weight.Grad != nil {
			for r := 0; r < rows; r++ {
				base := r * c
				for i := 0; i < c; i++ {
					weight.Grad[i] += out.Grad[base+i] * xHat[base+i]
				}
			}
		}
		if bias != nil && bias.Grad != nil {
			for r := 0; r < rows; r++ {
				base := r * c
				for i := 0; i < c; i++ {
					bias.Grad[i] += out.Grad[base+i]
				}
			}
		}

		if x.Grad == nil {
			return
		}

		// Per-row dx using the three-term LN backward.
		for r := 0; r < rows; r++ {
			base := r * c
			rstd := rstds[r]
			mean := means[r]

			// Compute sum(dxHat) and sum(dxHat * xHat) over the row.
			var sumDxHat, sumDxHatXHat float64
			for i := 0; i < c; i++ {
				dy := float64(out.Grad[base+i])
				w := float64(weight.Data[i])
				dxHat := dy * w
				sumDxHat += dxHat
				sumDxHatXHat += dxHat * float64(xHat[base+i])
			}

			// Closed-form LN backward: for each j,
			//   dx_j = rstd * (dxHat_j - mean(dxHat) - xHat_j * mean(dxHat*xHat))
			// which is algebraically equivalent to the dvar/dmean expansion in
			// the spec but avoids redundant passes over the row.
			_ = mean // kept for clarity; not needed in this closed form.
			meanDxHat := sumDxHat * invC
			meanDxHatXHat := sumDxHatXHat * invC
			for i := 0; i < c; i++ {
				dy := float64(out.Grad[base+i])
				w := float64(weight.Data[i])
				dxHat := dy * w
				dx := rstd * (dxHat - meanDxHat - float64(xHat[base+i])*meanDxHatXHat)
				x.Grad[base+i] += float32(dx)
			}
		}
	}

	return out, bw
}
