// Package train implements the `nanogpt train` subcommand.
package train

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"nanogpt/checkpoint"
	"nanogpt/data"
	"nanogpt/model"
	"nanogpt/optim"
	"nanogpt/schedule"
)

// Run executes the train subcommand.
func Run(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)

	dataDir := fs.String("data-dir", "data/shakespeare_char", "directory containing train.bin, val.bin, vocab.json")
	outDir := fs.String("out-dir", "out", "directory to write checkpoints")

	nLayer := fs.Int("n-layer", 6, "number of transformer blocks")
	nHead := fs.Int("n-head", 6, "number of attention heads")
	nEmbd := fs.Int("n-embd", 384, "embedding dimension")
	blockSize := fs.Int("block-size", 256, "context length T")
	vocabSize := fs.Int("vocab-size", 0, "vocab size (0 = autodetect from vocab.json)")
	dropout := fs.Float64("dropout", 0.2, "dropout probability")
	bias := fs.Bool("bias", true, "use bias in linear/layernorm")

	batchSize := fs.Int("batch-size", 64, "micro-batch size B")
	maxIters := fs.Int("max-iters", 5000, "total training steps")
	evalInterval := fs.Int("eval-interval", 250, "steps between val-loss evaluations")
	evalIters := fs.Int("eval-iters", 200, "batches averaged in a val-loss estimate")
	logInterval := fs.Int("log-interval", 10, "steps between training log lines")

	lr := fs.Float64("lr", 1e-3, "peak learning rate")
	minLR := fs.Float64("min-lr", 1e-4, "floor learning rate")
	warmupIters := fs.Int("warmup-iters", 100, "warmup steps")
	weightDecay := fs.Float64("weight-decay", 0.1, "AdamW weight decay")
	beta1 := fs.Float64("beta1", 0.9, "AdamW beta1")
	beta2 := fs.Float64("beta2", 0.99, "AdamW beta2")
	eps := fs.Float64("eps", 1e-8, "AdamW epsilon")
	gradClip := fs.Float64("grad-clip", 1.0, "global gradient-norm clip (<=0 disables)")
	gradAccumSteps := fs.Int("grad-accum-steps", 1, "gradient accumulation steps")
	resume := fs.String("resume", "", "path to checkpoint to resume from (empty = fresh)")
	seed := fs.Int64("seed", 1337, "RNG seed")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "train: %v\n", err)
		os.Exit(1)
	}

	if *gradAccumSteps < 1 {
		fmt.Fprintf(os.Stderr, "train: --grad-accum-steps must be >= 1, got %d\n", *gradAccumSteps)
		os.Exit(1)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "train: mkdir %q: %v\n", *outDir, err)
		os.Exit(1)
	}

	// Load data.
	trainPath := filepath.Join(*dataDir, "train.bin")
	valPath := filepath.Join(*dataDir, "val.bin")
	vocabPath := filepath.Join(*dataDir, "vocab.json")

	trainSet, err := data.LoadDataset(trainPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "train: %v\n", err)
		os.Exit(1)
	}
	valSet, err := data.LoadDataset(valPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "train: %v\n", err)
		os.Exit(1)
	}

	vocab, err := data.LoadVocab(vocabPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "train: %v\n", err)
		os.Exit(1)
	}

	vs := *vocabSize
	if vs == 0 {
		vs = vocab.VocabSize
	}

	cfg := model.GPTConfig{
		NLayer:    *nLayer,
		NHead:     *nHead,
		NEmbdg:    *nEmbd,
		BlockSize: *blockSize,
		VocabSize: vs,
		Dropout:   float32(*dropout),
		Bias:      *bias,
	}

	optCfg := optim.AdamWConfig{
		LR:          float32(*lr),
		Beta1:       float32(*beta1),
		Beta2:       float32(*beta2),
		Eps:         float32(*eps),
		WeightDecay: float32(*weightDecay),
	}

	rng := rand.New(rand.NewSource(*seed))
	valRng := rand.New(rand.NewSource(*seed + 1))

	g := model.NewGPT(cfg, rng)
	opt := optim.NewAdamW(optCfg)

	startStep := 0
	bestValLoss := float32(1e9)

	if *resume != "" {
		ckpt, err := checkpoint.Load(*resume)
		if err != nil {
			fmt.Fprintf(os.Stderr, "train: resume: %v\n", err)
			os.Exit(1)
		}
		// Rebuild model and optimizer to match the checkpoint's config.
		cfg = ckpt.Config
		optCfg = ckpt.OptimizerCfg
		g = model.NewGPT(cfg, rng)
		opt = optim.NewAdamW(optCfg)
		// Optimizer must see Update() with matching params before LoadState
		// will actually populate its moments. StateTensors/LoadState use
		// insertion order. Easiest: do one dummy Update with zero grads.
		primeOptimizer(opt, g)
		checkpoint.UnpackModel(ckpt, g, opt)
		startStep = ckpt.Step
		bestValLoss = ckpt.BestValLoss
		fmt.Printf("resumed from %s at step %d (best val loss %.4f)\n", *resume, startStep, bestValLoss)
	}

	fmt.Printf("model: vocab=%d block=%d n_layer=%d n_head=%d n_embd=%d params=%d\n",
		cfg.VocabSize, cfg.BlockSize, cfg.NLayer, cfg.NHead, cfg.NEmbdg, g.NumParameters())
	fmt.Printf("data: train=%d val=%d tokens\n", trainSet.Len(), valSet.Len())

	sched := schedule.CosineSchedule{
		WarmupIters: *warmupIters,
		MaxIters:    *maxIters,
		MaxLR:       float32(*lr),
		MinLR:       float32(*minLR),
	}

	B, T := *batchSize, *blockSize

	lastLogTime := time.Now()
	var lastLoss float32

	for step := startStep; step < *maxIters; step++ {
		// Evaluation (step 0 or every eval-interval).
		if step == startStep || (step > 0 && step%*evalInterval == 0) {
			valLoss := estimateLoss(g, valSet, B, T, *evalIters, valRng)
			trainLoss := estimateLoss(g, trainSet, B, T, *evalIters, valRng)
			stepMS := float32(time.Since(lastLogTime).Milliseconds())
			currentLR := sched.LRAt(step)
			fmt.Printf("step %d: train %.4f, val %.4f, lr %.2e, %.1f ms/step\n",
				step, trainLoss, valLoss, currentLR, stepMS)

			if step == startStep || valLoss < bestValLoss {
				bestValLoss = valLoss
				ckptPath := filepath.Join(*outDir, "ckpt.ckpt")
				ckpt := checkpoint.PackModel(g, opt, step, bestValLoss, cfg, optCfg)
				if err := checkpoint.Save(ckptPath, ckpt); err != nil {
					fmt.Fprintf(os.Stderr, "train: save checkpoint: %v\n", err)
					os.Exit(1)
				}
				// Also copy vocab next to the checkpoint so sample can find it.
				sideVocab := filepath.Join(*outDir, "vocab.json")
				if err := data.SaveVocab(sideVocab, vocab); err != nil {
					fmt.Fprintf(os.Stderr, "train: save vocab: %v\n", err)
				}
			}
			lastLogTime = time.Now()
		}

		// LR schedule.
		opt.SetLR(sched.LRAt(step))

		stepStart := time.Now()

		// Gradient accumulation: zero grads ONCE, then accumulate grads over
		// multiple forward/backward passes before a single Update.
		g.ZeroGrad()
		var accumLoss float32
		for micro := 0; micro < *gradAccumSteps; micro++ {
			x, y := trainSet.RandomBatch(B, T, rng)
			loss, bwd := g.ForwardBT(x, B, T, y, true, rng)
			bwd()
			accumLoss += loss
		}
		if *gradAccumSteps > 0 {
			lastLoss = accumLoss / float32(*gradAccumSteps)
		}

		if *gradClip > 0 {
			model.ClipGradNorm(g.Parameters(), float32(*gradClip))
		}
		opt.Update(g.Parameters())

		if step%*logInterval == 0 {
			dt := time.Since(stepStart)
			fmt.Printf("step %d: loss %.4f, lr %.2e, dt %.1fms\n",
				step, lastLoss, opt.Config.LR, float32(dt.Microseconds())/1000.0)
		}
	}

	// Final save.
	finalLoss := estimateLoss(g, valSet, B, T, *evalIters, valRng)
	if finalLoss < bestValLoss {
		bestValLoss = finalLoss
	}
	ckptPath := filepath.Join(*outDir, "ckpt.ckpt")
	ckpt := checkpoint.PackModel(g, opt, *maxIters, bestValLoss, cfg, optCfg)
	if err := checkpoint.Save(ckptPath, ckpt); err != nil {
		fmt.Fprintf(os.Stderr, "train: save final checkpoint: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("done. final val %.4f, best val %.4f, wrote %s\n", finalLoss, bestValLoss, ckptPath)
}

// estimateLoss averages ForwardBT loss over n random batches with
// training=false. Uses a separate rng to avoid perturbing training state.
func estimateLoss(g *model.GPT, ds *data.Dataset, B, T, n int, rng *rand.Rand) float32 {
	if n <= 0 {
		return 0
	}
	var sum float32
	for i := 0; i < n; i++ {
		x, y := ds.RandomBatch(B, T, rng)
		loss, _ := g.ForwardBT(x, B, T, y, false, rng)
		sum += loss
	}
	return sum / float32(n)
}

// primeOptimizer runs a single no-op Update so the optimizer allocates and
// orders m1/m2 buffers for every parameter before LoadState attempts to copy
// them in. Grads are already zero on a freshly-built model, so this is a
// safe no-op mathematically (first-moment and second-moment updates with
// zero grads leave data unchanged under AdamW's update rule? — no, the
// weight-decay term still fires. We therefore snapshot and restore data.)
func primeOptimizer(opt *optim.AdamW, g *model.GPT) {
	params := g.Parameters()
	snapshots := make([][]float32, len(params))
	for i, p := range params {
		s := make([]float32, len(p.Data))
		copy(s, p.Data)
		snapshots[i] = s
		// Ensure grads are zero.
		for j := range p.Grad {
			p.Grad[j] = 0
		}
	}
	opt.Update(params)
	// Undo weight-decay drift so the subsequent LoadNamedParameters replaces
	// pristine values.
	for i, p := range params {
		copy(p.Data, snapshots[i])
	}
	// Reset optimizer step so resumed training's step counting (independent
	// of this prime) stays clean.
	opt.Step = 0
}
