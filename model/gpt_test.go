package model

import (
	"math/rand"
	"testing"
)

func TestCropBlockSize(t *testing.T) {
	cfg := GPTConfig{
		NLayer:    2,
		NHead:     2,
		NEmbdg:    16,
		BlockSize: 32,
		VocabSize: 20,
		Dropout:   0.0,
		Bias:      true,
	}
	rng := rand.New(rand.NewSource(1))
	g := NewGPT(cfg, rng)

	g.CropBlockSize(8)

	if g.Config.BlockSize != 8 {
		t.Fatalf("BlockSize = %d, want 8", g.Config.BlockSize)
	}
	if g.WPE.Weight.Shape[0] != 8 {
		t.Fatalf("WPE rows = %d, want 8", g.WPE.Weight.Shape[0])
	}
	for i, blk := range g.Blocks {
		if blk.Attn.Mask.Shape[0] != 8 || blk.Attn.Mask.Shape[1] != 8 {
			t.Fatalf("block %d mask shape = %v, want [8 8]", i, blk.Attn.Mask.Shape)
		}
	}

	B, T := 2, 6
	idx := make([]int32, B*T)
	for i := range idx {
		idx[i] = int32(rng.Intn(cfg.VocabSize))
	}
	targets := make([]int32, B*T)
	for i := range targets {
		targets[i] = int32(rng.Intn(cfg.VocabSize))
	}
	loss, bwd := g.ForwardBT(idx, B, T, targets, true, rng)
	if loss <= 0 {
		t.Fatalf("loss = %v, want > 0", loss)
	}
	bwd()
}
