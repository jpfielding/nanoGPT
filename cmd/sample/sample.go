// Package sample implements the `nanogpt sample` subcommand: load a
// checkpoint, reconstruct the model, and generate text.
package sample

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"nanogpt/checkpoint"
	"nanogpt/data"
	"nanogpt/model"
	"nanogpt/optim"
)

// Run executes the sample subcommand.
func Run(args []string) {
	fs := flag.NewFlagSet("sample", flag.ExitOnError)

	ckptPath := fs.String("ckpt", "", "path to .ckpt file (required)")
	vocabPath := fs.String("vocab", "", "path to vocab.json (default: <ckpt dir>/vocab.json)")
	prompt := fs.String("prompt", "\n", "prompt string to condition on")
	maxNewTokens := fs.Int("max-new-tokens", 500, "number of tokens to generate per sample")
	temperature := fs.Float64("temperature", 0.8, "sampling temperature")
	topK := fs.Int("top-k", 200, "top-k truncation (<=0 disables)")
	numSamples := fs.Int("num-samples", 1, "number of samples to generate")
	seed := fs.Int64("seed", 1337, "RNG seed")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "sample: %v\n", err)
		os.Exit(1)
	}
	if *ckptPath == "" {
		fmt.Fprintln(os.Stderr, "sample: --ckpt is required")
		os.Exit(1)
	}

	ckpt, err := checkpoint.Load(*ckptPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sample: %v\n", err)
		os.Exit(1)
	}

	vp := *vocabPath
	if vp == "" {
		vp = filepath.Join(filepath.Dir(*ckptPath), "vocab.json")
	}
	vocab, err := data.LoadVocab(vp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sample: %v\n", err)
		os.Exit(1)
	}

	// Rebuild the model per the checkpoint's config and load weights.
	rng := rand.New(rand.NewSource(*seed))
	g := model.NewGPT(ckpt.Config, rng)

	// UnpackModel also touches optimizer state. We do not need it for
	// inference, so provide a throwaway optimizer whose LoadState is a no-op
	// on unknown keys (the standard AdamW is a no-op when order is empty).
	opt := optim.NewAdamW(ckpt.OptimizerCfg)
	checkpoint.UnpackModel(ckpt, g, opt)

	// Support FILE:<path> syntax to load prompt from a text file.
	promptStr := *prompt
	if len(promptStr) > 5 && promptStr[:5] == "FILE:" {
		raw, err := os.ReadFile(promptStr[5:])
		if err != nil {
			fmt.Fprintf(os.Stderr, "sample: read prompt file: %v\n", err)
			os.Exit(1)
		}
		promptStr = string(raw)
	}

	encodedPrompt := vocab.Encode(promptStr)
	ctx := make([]int32, len(encodedPrompt))
	for i, tok := range encodedPrompt {
		ctx[i] = int32(tok)
	}

	for s := 0; s < *numSamples; s++ {
		out := g.Generate(ctx, ckpt.Config.BlockSize, *maxNewTokens, float32(*temperature), *topK, rng)
		tokens := make([]uint16, len(out))
		for i, v := range out {
			tokens[i] = uint16(v)
		}
		fmt.Println(vocab.Decode(tokens))
		fmt.Println("---")
	}
}
