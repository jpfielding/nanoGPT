package data

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
)

// Dataset holds a flat sequence of uint16 token ids, mirroring the
// numpy `dtype=uint16` binary format used by the Python reference.
type Dataset struct {
	Tokens []uint16
}

// LoadDataset reads a little-endian uint16 binary file into memory.
// The file length must be a multiple of 2.
func LoadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset %q: %w", path, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat dataset %q: %w", path, err)
	}
	size := info.Size()
	if size%2 != 0 {
		return nil, fmt.Errorf("dataset %q: size %d is not a multiple of 2", path, size)
	}
	n := int(size / 2)
	tokens := make([]uint16, n)

	// binary.Read with a typed slice handles little-endian uint16 decoding
	// directly; for larger files a raw read + unsafe cast would be faster,
	// but clarity wins here.
	if err := binary.Read(f, binary.LittleEndian, tokens); err != nil && err != io.EOF {
		return nil, fmt.Errorf("read dataset %q: %w", path, err)
	}
	return &Dataset{Tokens: tokens}, nil
}

// Len returns the number of tokens in the dataset.
func (d *Dataset) Len() int { return len(d.Tokens) }

// RandomBatch draws B random contiguous windows of length T+1 and returns
// flat row-major input/target slices of length B*T. y is x shifted by one.
//
// Each start position is drawn from [0, len(tokens)-T-1).
func (d *Dataset) RandomBatch(B, T int, rng *rand.Rand) (x, y []int32) {
	if B <= 0 || T <= 0 {
		panic(fmt.Sprintf("RandomBatch: B and T must be positive, got B=%d T=%d", B, T))
	}
	hi := len(d.Tokens) - T - 1
	if hi <= 0 {
		panic(fmt.Sprintf("RandomBatch: dataset too small (len=%d) for T=%d", len(d.Tokens), T))
	}

	x = make([]int32, B*T)
	y = make([]int32, B*T)
	for b := 0; b < B; b++ {
		start := rng.Intn(hi)
		base := b * T
		// Hoist the per-batch slice bound-check out of the inner loop.
		src := d.Tokens[start : start+T+1]
		for t := 0; t < T; t++ {
			x[base+t] = int32(src[t])
			y[base+t] = int32(src[t+1])
		}
	}
	return x, y
}
