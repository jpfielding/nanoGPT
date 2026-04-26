package model

import (
	"fmt"
	"math"
	"math/rand"

	"nanogpt/optim"
	"nanogpt/tensor"
)

// GPT is a decoder-only transformer language model.
type GPT struct {
	Config GPTConfig
	WTE    *Embedding // token embedding [VocabSize, C]
	WPE    *Embedding // position embedding [BlockSize, C]
	DropE  float32    // embedding dropout probability
	Blocks []*Block   // NLayer transformer blocks
	LNF    *LayerNorm // final layer norm
	// LMHead's weight is TIED to WTE.Weight (same *Tensor pointer);
	// LMHead.Bias is always nil.
	LMHead *Linear
}

// NewGPT constructs a GPT model per cfg. The language-model head's weight is
// tied to the token-embedding weight (they are the same *Tensor).
func NewGPT(cfg GPTConfig, rng *rand.Rand) *GPT {
	wte := NewEmbedding(cfg.VocabSize, cfg.NEmbdg, rng)
	wpe := NewEmbedding(cfg.BlockSize, cfg.NEmbdg, rng)

	blocks := make([]*Block, cfg.NLayer)
	for i := range blocks {
		blocks[i] = NewBlock(cfg, rng)
	}

	lnf := NewLayerNorm(cfg.NEmbdg, cfg.Bias)

	// Weight tying: share the embedding weight tensor with the LM head.
	lmHead := &Linear{Weight: wte.Weight, Bias: nil}

	return &GPT{
		Config: cfg,
		WTE:    wte,
		WPE:    wpe,
		DropE:  cfg.Dropout,
		Blocks: blocks,
		LNF:    lnf,
		LMHead: lmHead,
	}
}

// ForwardBT runs the full GPT forward pass.
//
//	idx:      flat token ids of shape [B*T]
//	targets:  optional flat target ids of shape [B*T] (nil for inference);
//	          target -1 marks ignored positions.
//
// When targets is nil, the returned loss is 0 and the backward is a no-op.
// In both cases the logits tensor is computed internally; for inference it
// is only kept for the last time step to save memory.
func (g *GPT) ForwardBT(idx []int32, B, T int, targets []int32, training bool, rng *rand.Rand) (float32, tensor.BackwardFn) {
	if len(idx) != B*T {
		panic("model.GPT.ForwardBT: len(idx) must equal B*T")
	}
	if T > g.Config.BlockSize {
		panic("model.GPT.ForwardBT: T exceeds BlockSize")
	}
	C := g.Config.NEmbdg

	tape := make([]tensor.BackwardFn, 0, 4+4*len(g.Blocks))

	// 1. Token embeddings: [B*T, C] -> [B, T, C]
	tokFlat, tokBwd := g.WTE.Forward(idx)
	tape = append(tape, tokBwd)
	tokEmb := tokFlat.View(B, T, C)

	// 2. Position embeddings.
	pos := make([]int32, B*T)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			pos[b*T+t] = int32(t)
		}
	}
	posFlat, posBwd := g.WPE.Forward(pos)
	tape = append(tape, posBwd)
	posEmb := posFlat.View(B, T, C)

	// 3. x = tok + pos
	x, addBwd := tensor.Add(tokEmb, posEmb)
	tape = append(tape, addBwd)

	// 4. Embedding dropout.
	x, dropBwd := tensor.Dropout(x, g.DropE, training, rng)
	tape = append(tape, dropBwd)

	// 5. Transformer blocks.
	for _, blk := range g.Blocks {
		var bBwd tensor.BackwardFn
		x, bBwd = blk.Forward(x, training, rng)
		tape = append(tape, bBwd)
	}

	// 6. Final LayerNorm.
	x, lnfBwd := g.LNF.Forward(x)
	tape = append(tape, lnfBwd)

	// 7. Language-model head + loss.
	if targets == nil {
		// Inference: compute logits on the last time step of each batch.
		lastX := tensor.New(B, 1, C)
		for b := 0; b < B; b++ {
			srcBase := (b*T + (T - 1)) * C
			dstBase := b * C
			copy(lastX.Data[dstBase:dstBase+C], x.Data[srcBase:srcBase+C])
		}
		// LMHead.Forward needs grads? No — inference only.
		logits, _ := g.LMHead.Forward(lastX)
		_ = logits
		return 0, func() {}
	}

	// Training: run LM head over all positions and compute loss.
	if len(targets) != B*T {
		panic("model.GPT.ForwardBT: len(targets) must equal B*T")
	}
	logits, lmBwd := g.LMHead.Forward(x) // [B, T, VocabSize]
	tape = append(tape, lmBwd)

	// Flatten logits to 2-D for the loss.
	logits2 := logits.View(B*T, g.Config.VocabSize)
	loss, lossBwd := tensor.CrossEntropyLoss(logits2, targets)
	tape = append(tape, lossBwd)

	bwd := func() {
		for i := len(tape) - 1; i >= 0; i-- {
			tape[i]()
		}
	}
	return loss, bwd
}

// Parameters returns the full list of trainable parameters, deduplicated so
// that tied weights (WTE.Weight == LMHead.Weight) appear exactly once.
func (g *GPT) Parameters() []*optim.Param {
	var params []*optim.Param
	params = append(params, g.WTE.Parameters()...)
	params = append(params, g.WPE.Parameters()...)
	for _, blk := range g.Blocks {
		params = append(params, blk.Parameters()...)
	}
	params = append(params, g.LNF.Parameters()...)
	// LMHead shares weight with WTE; skip its weight here. LMHead has no bias.

	return dedupParams(params)
}

// dedupParams removes duplicate params keyed by the address of Data[0].
func dedupParams(in []*optim.Param) []*optim.Param {
	seen := make(map[*float32]bool, len(in))
	out := make([]*optim.Param, 0, len(in))
	for _, p := range in {
		if len(p.Data) == 0 {
			continue
		}
		key := &p.Data[0]
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, p)
	}
	return out
}

// ZeroGrad zeros the gradient buffer of every trainable parameter.
func (g *GPT) ZeroGrad() {
	for _, p := range g.Parameters() {
		for i := range p.Grad {
			p.Grad[i] = 0
		}
	}
}

// NumParameters returns the total number of unique scalar parameters.
func (g *GPT) NumParameters() int {
	n := 0
	for _, p := range g.Parameters() {
		n += len(p.Data)
	}
	return n
}

// NamedParameters returns a map of qualified parameter name to tensor for
// checkpoint serialization. The naming scheme mirrors PyTorch's state_dict.
func (g *GPT) NamedParameters() map[string]*tensor.Tensor {
	out := make(map[string]*tensor.Tensor)
	out["wte.weight"] = g.WTE.Weight
	out["wpe.weight"] = g.WPE.Weight

	for i, blk := range g.Blocks {
		prefix := fmt.Sprintf("blocks.%d.", i)
		out[prefix+"ln1.weight"] = blk.LN1.Weight
		if blk.LN1.Bias != nil {
			out[prefix+"ln1.bias"] = blk.LN1.Bias
		}
		out[prefix+"attn.c_attn.weight"] = blk.Attn.CAttn.Weight
		if blk.Attn.CAttn.Bias != nil {
			out[prefix+"attn.c_attn.bias"] = blk.Attn.CAttn.Bias
		}
		out[prefix+"attn.c_proj.weight"] = blk.Attn.CProj.Weight
		if blk.Attn.CProj.Bias != nil {
			out[prefix+"attn.c_proj.bias"] = blk.Attn.CProj.Bias
		}
		out[prefix+"ln2.weight"] = blk.LN2.Weight
		if blk.LN2.Bias != nil {
			out[prefix+"ln2.bias"] = blk.LN2.Bias
		}
		out[prefix+"mlp.fc1.weight"] = blk.MLP.FC1.Weight
		if blk.MLP.FC1.Bias != nil {
			out[prefix+"mlp.fc1.bias"] = blk.MLP.FC1.Bias
		}
		out[prefix+"mlp.fc2.weight"] = blk.MLP.FC2.Weight
		if blk.MLP.FC2.Bias != nil {
			out[prefix+"mlp.fc2.bias"] = blk.MLP.FC2.Bias
		}
	}

	out["lnf.weight"] = g.LNF.Weight
	if g.LNF.Bias != nil {
		out["lnf.bias"] = g.LNF.Bias
	}
	// LMHead weight is tied to wte.weight; no entry needed.
	return out
}

// LoadNamedParameters copies tensor data from params into the model's
// existing parameter tensors. Missing keys are ignored; shape mismatches
// panic.
func (g *GPT) LoadNamedParameters(params map[string]*tensor.Tensor) {
	current := g.NamedParameters()
	for name, dst := range current {
		src, ok := params[name]
		if !ok {
			continue
		}
		if len(src.Data) != len(dst.Data) {
			panic(fmt.Sprintf("model.GPT.LoadNamedParameters: shape mismatch for %s: got %d want %d",
				name, len(src.Data), len(dst.Data)))
		}
		copy(dst.Data, src.Data)
	}
}

// Generate autoregressively samples up to maxNewTokens from the model given
// an initial context. Sampling uses temperature scaling and top-k filtering;
// set topK <= 0 to disable top-k.
func (g *GPT) Generate(ctx []int32, _ int, maxNewTokens int, temperature float32, topK int, rng *rand.Rand) []int32 {
	out := append([]int32(nil), ctx...)
	V := g.Config.VocabSize
	bs := g.Config.BlockSize

	for i := 0; i < maxNewTokens; i++ {
		cond := out
		if len(cond) > bs {
			cond = cond[len(cond)-bs:]
		}
		T := len(cond)

		// Inference forward with B=1. ForwardBT with targets=nil only computes
		// logits for the last position internally, but the returned tensor is
		// discarded — so recompute here using the same path but keep logits.
		logits := g.forwardLogitsLast(cond, T, rng)

		// Temperature.
		if temperature > 0 && temperature != 1 {
			inv := 1.0 / float64(temperature)
			for j := range logits {
				logits[j] = float32(float64(logits[j]) * inv)
			}
		}

		// Top-k: set all but the top-k logits to -1e9.
		if topK > 0 && topK < V {
			applyTopK(logits, topK)
		}

		// Softmax.
		probs := softmax(logits)

		// Sample.
		next := sampleCategorical(probs, rng)
		out = append(out, next)
	}
	return out
}

// forwardLogitsLast runs the forward pass for B=1 and returns the logits for
// the last time step. Training is false; no gradient tape is kept.
func (g *GPT) forwardLogitsLast(idx []int32, T int, rng *rand.Rand) []float32 {
	C := g.Config.NEmbdg
	V := g.Config.VocabSize

	tokFlat, _ := g.WTE.Forward(idx) // [T, C]
	tokEmb := tokFlat.View(1, T, C)

	pos := make([]int32, T)
	for t := 0; t < T; t++ {
		pos[t] = int32(t)
	}
	posFlat, _ := g.WPE.Forward(pos)
	posEmb := posFlat.View(1, T, C)

	x, _ := tensor.Add(tokEmb, posEmb)
	x, _ = tensor.Dropout(x, 0, false, rng)

	for _, blk := range g.Blocks {
		x, _ = blk.Forward(x, false, rng)
	}

	x, _ = g.LNF.Forward(x)

	// Extract last position: x is [1, T, C].
	lastX := tensor.New(1, 1, C)
	copy(lastX.Data, x.Data[(T-1)*C:T*C])

	logits, _ := g.LMHead.Forward(lastX) // [1, 1, V]
	out := make([]float32, V)
	copy(out, logits.Data)
	return out
}

// applyTopK keeps the top-k values in logits and sets the rest to -1e9.
// k must satisfy 0 < k < len(logits).
func applyTopK(logits []float32, k int) {
	if k <= 0 || k >= len(logits) {
		return
	}
	// Find the k-th largest value by a simple partial sort (O(n*k)).
	// For typical k in [40, 200] and V in [65, 50257] this is fine.
	cp := make([]float32, len(logits))
	copy(cp, logits)
	// Partial selection: k passes of bubble-max.
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(cp); j++ {
			if cp[j] > cp[maxIdx] {
				maxIdx = j
			}
		}
		cp[i], cp[maxIdx] = cp[maxIdx], cp[i]
	}
	threshold := cp[k-1]
	for i, v := range logits {
		if v < threshold {
			logits[i] = -1e9
		}
	}
}

// softmax returns softmax(logits). Numerically stable via max subtraction.
func softmax(logits []float32) []float32 {
	maxv := logits[0]
	for _, v := range logits[1:] {
		if v > maxv {
			maxv = v
		}
	}
	out := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - maxv))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}

// sampleCategorical samples an index proportional to probs (which must sum
// to ~1). Uses inverse-CDF sampling.
func sampleCategorical(probs []float32, rng *rand.Rand) int32 {
	r := rng.Float32()
	var cum float32
	for i, p := range probs {
		cum += p
		if r < cum {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}

// ClipGradNorm clips gradients to a maximum global L2 norm. If the total
// norm exceeds maxNorm, every gradient is scaled by maxNorm/totalNorm.
func ClipGradNorm(params []*optim.Param, maxNorm float32) {
	var sumSq float64
	for _, p := range params {
		for _, g := range p.Grad {
			f := float64(g)
			sumSq += f * f
		}
	}
	totalNorm := float32(math.Sqrt(sumSq))
	if totalNorm <= maxNorm || totalNorm == 0 {
		return
	}
	scale := maxNorm / totalNorm
	for _, p := range params {
		for i := range p.Grad {
			p.Grad[i] *= scale
		}
	}
}
