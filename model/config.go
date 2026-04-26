// Package model implements the GPT transformer architecture on top of the
// nanogpt/tensor autograd primitives. All layers expose a Forward method that
// returns the output tensor together with a backward closure, and a
// Parameters method that lists trainable tensors for the optimizer.
package model

// GPTConfig configures the GPT model's architecture.
type GPTConfig struct {
	NLayer    int
	NHead     int
	NEmbdg    int // embedding dimension C
	BlockSize int // max sequence length T
	VocabSize int
	Dropout   float32
	Bias      bool // include bias in linear layers and layernorm
}

// ShakespeareConfig returns the default char-level Shakespeare config.
func ShakespeareConfig() GPTConfig {
	return GPTConfig{
		NLayer:    6,
		NHead:     6,
		NEmbdg:    384,
		BlockSize: 256,
		VocabSize: 65,
		Dropout:   0.2,
		Bias:      true,
	}
}
