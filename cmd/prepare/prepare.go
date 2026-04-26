// Package prepare implements the `nanogpt prepare` subcommand: it reads a
// plain-text corpus, builds a character-level vocabulary, and emits
// train.bin / val.bin / vocab.json artifacts compatible with `nanogpt train`.
package prepare

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"nanogpt/data"
)

// Run executes the prepare subcommand with the given argv (without the
// program name or subcommand token).
func Run(args []string) {
	fs := flag.NewFlagSet("prepare", flag.ExitOnError)
	input := fs.String("input", "", "input text file (required)")
	outDir := fs.String("out-dir", "data/shakespeare_char", "output directory")
	valSplit := fs.Float64("val-split", 0.1, "fraction for validation")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "prepare: %v\n", err)
		os.Exit(1)
	}

	if *input == "" {
		fmt.Fprintln(os.Stderr, "prepare: --input is required")
		os.Exit(1)
	}
	if *valSplit < 0 || *valSplit >= 1 {
		fmt.Fprintf(os.Stderr, "prepare: --val-split must be in [0, 1), got %v\n", *valSplit)
		os.Exit(1)
	}

	text, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "prepare: read input %q: %v\n", *input, err)
		os.Exit(1)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "prepare: mkdir %q: %v\n", *outDir, err)
		os.Exit(1)
	}

	vocab := data.BuildVocab(string(text))
	tokens := vocab.Encode(string(text))

	// Split: train first (1 - valSplit), val last (valSplit).
	n := len(tokens)
	nTrain := n - int(float64(n)*(*valSplit))
	if nTrain < 0 {
		nTrain = 0
	}
	if nTrain > n {
		nTrain = n
	}
	trainTokens := tokens[:nTrain]
	valTokens := tokens[nTrain:]

	trainPath := filepath.Join(*outDir, "train.bin")
	valPath := filepath.Join(*outDir, "val.bin")
	vocabPath := filepath.Join(*outDir, "vocab.json")

	if err := writeTokens(trainPath, trainTokens); err != nil {
		fmt.Fprintf(os.Stderr, "prepare: %v\n", err)
		os.Exit(1)
	}
	if err := writeTokens(valPath, valTokens); err != nil {
		fmt.Fprintf(os.Stderr, "prepare: %v\n", err)
		os.Exit(1)
	}
	if err := data.SaveVocab(vocabPath, vocab); err != nil {
		fmt.Fprintf(os.Stderr, "prepare: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("vocab_size: %d\n", vocab.VocabSize)
	fmt.Printf("train tokens: %d\n", len(trainTokens))
	fmt.Printf("val tokens:   %d\n", len(valTokens))
	fmt.Printf("wrote %s\n", trainPath)
	fmt.Printf("wrote %s\n", valPath)
	fmt.Printf("wrote %s\n", vocabPath)
}

// writeTokens writes tokens as raw little-endian uint16 to path.
func writeTokens(path string, tokens []uint16) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %q: %w", path, err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, tokens); err != nil {
		return fmt.Errorf("write %q: %w", path, err)
	}
	return nil
}
