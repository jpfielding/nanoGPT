// Package data provides character-level tokenization and binary dataset loading
// for nanoGPT-style training.
package data

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// Vocab is a character-level vocabulary mapping single UTF-8 characters to
// integer token ids and back. It is JSON-serializable via SaveVocab/LoadVocab.
type Vocab struct {
	VocabSize int
	Chars     string         // sorted unique chars as a string
	StoI      map[string]int // single UTF-8 char -> token id
	ItoS      map[int]string // token id -> single UTF-8 char
}

// vocabJSON is the on-disk JSON shape. Keys use snake_case to match the
// Python reference implementation.
type vocabJSON struct {
	VocabSize int               `json:"vocab_size"`
	Chars     string            `json:"chars"`
	StoI      map[string]int    `json:"stoi"`
	ItoS      map[string]string `json:"itos"` // keys are stringified ints
}

// BuildVocab scans text, collects unique runes, sorts them, and assigns
// sequential token ids. The returned Vocab's Chars field is the sorted
// concatenation of all unique runes.
func BuildVocab(text string) *Vocab {
	seen := make(map[rune]struct{})
	for _, r := range text {
		seen[r] = struct{}{}
	}
	runes := make([]rune, 0, len(seen))
	for r := range seen {
		runes = append(runes, r)
	}
	sort.Slice(runes, func(i, j int) bool { return runes[i] < runes[j] })

	stoi := make(map[string]int, len(runes))
	itos := make(map[int]string, len(runes))
	for i, r := range runes {
		s := string(r)
		stoi[s] = i
		itos[i] = s
	}
	return &Vocab{
		VocabSize: len(runes),
		Chars:     string(runes),
		StoI:      stoi,
		ItoS:      itos,
	}
}

// LoadVocab reads a JSON vocabulary file previously written by SaveVocab.
func LoadVocab(path string) (*Vocab, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load vocab %q: %w", path, err)
	}
	var raw vocabJSON
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("decode vocab %q: %w", path, err)
	}
	itos := make(map[int]string, len(raw.ItoS))
	for k, v := range raw.ItoS {
		var id int
		if _, err := fmt.Sscanf(k, "%d", &id); err != nil {
			return nil, fmt.Errorf("decode vocab %q: bad itos key %q: %w", path, k, err)
		}
		itos[id] = v
	}
	return &Vocab{
		VocabSize: raw.VocabSize,
		Chars:     raw.Chars,
		StoI:      raw.StoI,
		ItoS:      itos,
	}, nil
}

// SaveVocab writes v to path as JSON. The itos map's int keys are serialized
// as their decimal string representations.
func SaveVocab(path string, v *Vocab) error {
	itos := make(map[string]string, len(v.ItoS))
	for k, s := range v.ItoS {
		itos[fmt.Sprintf("%d", k)] = s
	}
	raw := vocabJSON{
		VocabSize: v.VocabSize,
		Chars:     v.Chars,
		StoI:      v.StoI,
		ItoS:      itos,
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return fmt.Errorf("encode vocab: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("write vocab %q: %w", path, err)
	}
	return nil
}

// Encode converts s into a slice of token ids. An unknown rune panics with a
// helpful message identifying the offending character.
func (v *Vocab) Encode(s string) []uint16 {
	out := make([]uint16, 0, len(s))
	for _, r := range s {
		id, ok := v.StoI[string(r)]
		if !ok {
			panic(fmt.Sprintf("vocab: unknown rune %q (U+%04X) not present in vocabulary", r, r))
		}
		out = append(out, uint16(id))
	}
	return out
}

// Decode converts a slice of token ids back into a string by concatenating
// ItoS lookups. Unknown ids panic with a helpful message.
func (v *Vocab) Decode(tokens []uint16) string {
	// Pre-size: most tokens map to a 1-byte rune, so len(tokens) is a good
	// lower bound.
	buf := make([]byte, 0, len(tokens))
	for _, t := range tokens {
		s, ok := v.ItoS[int(t)]
		if !ok {
			panic(fmt.Sprintf("vocab: unknown token id %d not present in vocabulary", t))
		}
		buf = append(buf, s...)
	}
	return string(buf)
}
