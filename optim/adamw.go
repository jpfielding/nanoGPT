// Package optim provides optimizers for nanoGPT training. Currently only
// AdamW with decoupled weight decay is implemented.
package optim

import (
	"fmt"
	"math"
)

// Param wraps a flat parameter tensor alongside its gradient buffer and a
// flag indicating whether weight decay should be applied. Data and Grad must
// have the same length.
type Param struct {
	Data        []float32
	Grad        []float32
	DecayWeight bool
}

// AdamWConfig holds the hyperparameters for the AdamW optimizer.
type AdamWConfig struct {
	LR          float32
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32
}

// AdamW implements AdamW with decoupled weight decay. First and second moment
// buffers are lazily allocated on the first Update call in which a parameter
// is seen, keyed by the address of the parameter's first data element.
type AdamW struct {
	Config AdamWConfig
	Step   int

	// m1/m2 are keyed by &param.Data[0] so they survive across calls without
	// requiring the caller to hold stable *Param pointers. The ordered slice
	// of keys preserves insertion order for deterministic checkpointing.
	m1     map[*float32][]float32
	m2     map[*float32][]float32
	order  []*float32
	orderI map[*float32]int
}

// NewAdamW returns an AdamW ready for its first Update call. Step starts at 0
// and is incremented to 1 on the first update.
func NewAdamW(cfg AdamWConfig) *AdamW {
	return &AdamW{
		Config: cfg,
		m1:     make(map[*float32][]float32),
		m2:     make(map[*float32][]float32),
		orderI: make(map[*float32]int),
	}
}

// SetLR updates the learning rate used by subsequent Update calls. The LR
// scheduler should call this once per step.
func (o *AdamW) SetLR(lr float32) {
	o.Config.LR = lr
}

// Update performs one AdamW step across params. Step is incremented first so
// that the very first update uses step=1 for bias correction.
func (o *AdamW) Update(params []*Param) {
	o.Step++
	cfg := o.Config

	// Bias-correction terms: compute in float64 to avoid precision loss at
	// large step counts.
	stepF := float64(o.Step)
	bc1 := 1.0 - math.Pow(float64(cfg.Beta1), stepF)
	bc2 := 1.0 - math.Pow(float64(cfg.Beta2), stepF)

	beta1 := cfg.Beta1
	beta2 := cfg.Beta2
	oneMinusBeta1 := 1 - beta1
	oneMinusBeta2 := 1 - beta2
	lr := cfg.LR
	eps := cfg.Eps

	for _, p := range params {
		if len(p.Data) == 0 {
			continue
		}
		if len(p.Data) != len(p.Grad) {
			panic(fmt.Sprintf("adamw: param size mismatch: data=%d grad=%d", len(p.Data), len(p.Grad)))
		}
		key := &p.Data[0]
		m1, ok := o.m1[key]
		if !ok {
			m1 = make([]float32, len(p.Data))
			m2 := make([]float32, len(p.Data))
			o.m1[key] = m1
			o.m2[key] = m2
			o.orderI[key] = len(o.order)
			o.order = append(o.order, key)
		}
		m2 := o.m2[key]

		// Pre-hoist the decoupled weight-decay multiplier for this param.
		var decay float32 = 1
		if p.DecayWeight && cfg.WeightDecay != 0 {
			decay = 1 - lr*cfg.WeightDecay
		}

		data := p.Data
		grad := p.Grad
		// All four slices share the same length; range over one to let the
		// compiler elide bounds checks on the others.
		for i := range data {
			g := grad[i]

			// Update biased moments.
			m1i := beta1*m1[i] + oneMinusBeta1*g
			m2i := beta2*m2[i] + oneMinusBeta2*g*g
			m1[i] = m1i
			m2[i] = m2i

			// Bias-corrected estimates, performed in float32 after the
			// float64 division above.
			m1Hat := float32(float64(m1i) / bc1)
			m2Hat := float32(float64(m2i) / bc2)

			w := data[i] * decay
			w -= lr * m1Hat / (float32(math.Sqrt(float64(m2Hat))) + eps)
			data[i] = w
		}
	}
}

// StateTensors returns the optimizer's m1/m2 buffers keyed by insertion
// order for checkpointing. Modifying the returned slices mutates optimizer
// state.
func (o *AdamW) StateTensors() map[string][]float32 {
	out := make(map[string][]float32, 2*len(o.order))
	for idx, key := range o.order {
		out[fmt.Sprintf("m1_%d", idx)] = o.m1[key]
		out[fmt.Sprintf("m2_%d", idx)] = o.m2[key]
	}
	return out
}

// LoadState restores m1/m2 buffers previously returned by StateTensors. The
// optimizer must have already been Update-d with the matching params (so that
// the insertion order matches), or the buffers will be ignored.
func (o *AdamW) LoadState(state map[string][]float32) {
	for idx, key := range o.order {
		if v, ok := state[fmt.Sprintf("m1_%d", idx)]; ok {
			if len(v) != len(o.m1[key]) {
				panic(fmt.Sprintf("adamw: m1_%d size mismatch: got %d want %d", idx, len(v), len(o.m1[key])))
			}
			copy(o.m1[key], v)
		}
		if v, ok := state[fmt.Sprintf("m2_%d", idx)]; ok {
			if len(v) != len(o.m2[key]) {
				panic(fmt.Sprintf("adamw: m2_%d size mismatch: got %d want %d", idx, len(v), len(o.m2[key])))
			}
			copy(o.m2[key], v)
		}
	}
}
