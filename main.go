// nanogpt is a minimal character-level GPT implementation in Go. The binary
// has three subcommands:
//
//	nanogpt prepare   build vocab + train/val binaries from a text file
//	nanogpt train     train a GPT model, writing checkpoints to --out-dir
//	nanogpt sample    generate text from a checkpoint
package main

import (
	"fmt"
	"os"

	"nanogpt/cmd/prepare"
	"nanogpt/cmd/sample"
	"nanogpt/cmd/train"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: nanogpt <prepare|train|sample> [flags]")
		os.Exit(1)
	}
	switch os.Args[1] {
	case "prepare":
		prepare.Run(os.Args[2:])
	case "train":
		train.Run(os.Args[2:])
	case "sample":
		sample.Run(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Println("usage: nanogpt <prepare|train|sample> [flags]")
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}
