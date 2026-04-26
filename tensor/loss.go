package tensor

import "math"

// CrossEntropyLoss computes the mean cross-entropy loss of logits against
// integer targets. A target value of -1 marks that row as ignored: it
// contributes neither to the loss nor to the gradient, and the denominator
// is the count of non-ignored rows.
//
// logits: [N, vocabSize] (e.g. N = B*T for a flattened language-model batch)
// targets: length N, int32 class ids (-1 to ignore).
//
// The forward uses logsumexp for numerical stability. The backward fills
// logits.Grad with (softmax - one_hot) / count for every non-ignored row
// (ignored rows receive zero gradient).
func CrossEntropyLoss(logits *Tensor, targets []int32) (float32, BackwardFn) {
	if len(logits.Shape) != 2 {
		panic("tensor.CrossEntropyLoss: logits must be 2-D [N, V]")
	}
	n := logits.Shape[0]
	v := logits.Shape[1]
	if len(targets) != n {
		panic("tensor.CrossEntropyLoss: targets length must equal logits.Shape[0]")
	}

	// First pass: compute loss and stash per-row (max, logZ) for the backward.
	maxes := make([]float32, n)
	logZ := make([]float64, n)
	var total float64
	count := 0

	for i := 0; i < n; i++ {
		tgt := targets[i]
		if tgt == -1 {
			continue
		}
		if tgt < 0 || int(tgt) >= v {
			panic("tensor.CrossEntropyLoss: target out of range")
		}
		row := logits.Data[i*v : (i+1)*v]
		maxv := row[0]
		for _, x := range row[1:] {
			if x > maxv {
				maxv = x
			}
		}
		maxes[i] = maxv
		var sum float64
		for _, x := range row {
			sum += math.Exp(float64(x - maxv))
		}
		lse := float64(maxv) + math.Log(sum)
		logZ[i] = lse
		total += lse - float64(row[tgt])
		count++
	}

	var loss float32
	if count > 0 {
		loss = float32(total / float64(count))
	}

	bw := func() {
		logits.ZeroGrad()
		if count == 0 {
			return
		}
		invCount := 1.0 / float64(count)
		for i := 0; i < n; i++ {
			tgt := targets[i]
			if tgt == -1 {
				continue
			}
			row := logits.Data[i*v : (i+1)*v]
			gRow := logits.Grad[i*v : (i+1)*v]
			maxv := maxes[i]
			lse := logZ[i]
			// softmax[j] = exp(row[j]-maxv) / (exp(lse-maxv))
			//            = exp(row[j] - lse)
			for j := 0; j < v; j++ {
				_ = maxv // maxes retained in case future code wants per-row reuse.
				p := math.Exp(float64(row[j]) - lse)
				gRow[j] = float32(p * invCount)
			}
			gRow[tgt] -= float32(invCount)
		}
	}

	return loss, bw
}
