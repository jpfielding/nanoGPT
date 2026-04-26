// Package checkpoint implements a custom binary format for saving and loading
// nanoGPT training state. A checkpoint contains JSON metadata (step, best
// validation loss, model and optimizer config) followed by a sequence of
// named float32 tensors (model weights and optimizer state).
package checkpoint

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"

	"nanogpt/model"
	"nanogpt/optim"
	"nanogpt/tensor"
)

// magic is the 8-byte file signature.
var magic = [8]byte{'N', 'A', 'N', 'O', 'G', 'P', 'T', 0x00}

// version is the checkpoint format version.
const version uint32 = 1

// Checkpoint is an in-memory representation of a nanoGPT checkpoint.
type Checkpoint struct {
	Step         int
	BestValLoss  float32
	Config       model.GPTConfig
	OptimizerCfg optim.AdamWConfig
	Tensors      map[string]*tensor.Tensor
}

// metadata is the JSON-serialized header portion of a checkpoint.
type metadata struct {
	Step            int               `json:"step"`
	BestValLoss     float32           `json:"best_val_loss"`
	Config          model.GPTConfig   `json:"config"`
	OptimizerConfig optim.AdamWConfig `json:"optimizer_config"`
}

// Save writes ckpt to path in the nanoGPT binary checkpoint format.
// Tensor keys are written in sorted order for deterministic output.
func Save(path string, ckpt *Checkpoint) error {
	if ckpt == nil {
		return errors.New("checkpoint.Save: nil checkpoint")
	}

	meta := metadata{
		Step:            ckpt.Step,
		BestValLoss:     ckpt.BestValLoss,
		Config:          ckpt.Config,
		OptimizerConfig: ckpt.OptimizerCfg,
	}
	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("checkpoint.Save: marshal metadata: %w", err)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("checkpoint.Save: create file: %w", err)
	}
	defer f.Close()

	bw := bufio.NewWriter(f)

	// Header: magic + version + MetaLen + JSON bytes.
	if _, err := bw.Write(magic[:]); err != nil {
		return fmt.Errorf("checkpoint.Save: write magic: %w", err)
	}
	if err := binary.Write(bw, binary.LittleEndian, version); err != nil {
		return fmt.Errorf("checkpoint.Save: write version: %w", err)
	}
	if err := binary.Write(bw, binary.LittleEndian, uint32(len(metaBytes))); err != nil {
		return fmt.Errorf("checkpoint.Save: write meta length: %w", err)
	}
	if _, err := bw.Write(metaBytes); err != nil {
		return fmt.Errorf("checkpoint.Save: write metadata: %w", err)
	}

	// Pad to 8-byte boundary.
	// Bytes written so far: 8 (magic) + 4 (version) + 4 (metalen) + len(metaBytes).
	written := 8 + 4 + 4 + len(metaBytes)
	pad := (8 - (written % 8)) % 8
	if pad > 0 {
		if _, err := bw.Write(make([]byte, pad)); err != nil {
			return fmt.Errorf("checkpoint.Save: write padding: %w", err)
		}
	}

	// Tensors (sorted for determinism).
	names := make([]string, 0, len(ckpt.Tensors))
	for name := range ckpt.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		t := ckpt.Tensors[name]
		if t == nil {
			return fmt.Errorf("checkpoint.Save: tensor %q is nil", name)
		}
		if err := writeTensor(bw, name, t); err != nil {
			return fmt.Errorf("checkpoint.Save: write tensor %q: %w", name, err)
		}
	}

	if err := bw.Flush(); err != nil {
		return fmt.Errorf("checkpoint.Save: flush: %w", err)
	}
	return nil
}

// writeTensor writes a single named tensor record.
func writeTensor(w io.Writer, name string, t *tensor.Tensor) error {
	nameBytes := []byte(name)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(nameBytes))); err != nil {
		return err
	}
	if _, err := w.Write(nameBytes); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(t.Shape))); err != nil {
		return err
	}
	shape32 := make([]int32, len(t.Shape))
	for i, d := range t.Shape {
		shape32[i] = int32(d)
	}
	if err := binary.Write(w, binary.LittleEndian, shape32); err != nil {
		return err
	}
	dataBytes := uint64(len(t.Data)) * 4
	if err := binary.Write(w, binary.LittleEndian, dataBytes); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, t.Data); err != nil {
		return err
	}
	return nil
}

// Load reads a checkpoint from path.
func Load(path string) (*Checkpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("checkpoint.Load: open file: %w", err)
	}
	defer f.Close()

	br := bufio.NewReader(f)

	// Magic.
	var gotMagic [8]byte
	if _, err := io.ReadFull(br, gotMagic[:]); err != nil {
		return nil, fmt.Errorf("checkpoint.Load: read magic: %w", err)
	}
	if gotMagic != magic {
		return nil, fmt.Errorf("checkpoint.Load: bad magic %q", gotMagic[:])
	}

	// Version.
	var gotVersion uint32
	if err := binary.Read(br, binary.LittleEndian, &gotVersion); err != nil {
		return nil, fmt.Errorf("checkpoint.Load: read version: %w", err)
	}
	if gotVersion != version {
		return nil, fmt.Errorf("checkpoint.Load: unsupported version %d (want %d)", gotVersion, version)
	}

	// Metadata.
	var metaLen uint32
	if err := binary.Read(br, binary.LittleEndian, &metaLen); err != nil {
		return nil, fmt.Errorf("checkpoint.Load: read meta length: %w", err)
	}
	metaBytes := make([]byte, metaLen)
	if _, err := io.ReadFull(br, metaBytes); err != nil {
		return nil, fmt.Errorf("checkpoint.Load: read metadata: %w", err)
	}
	var meta metadata
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return nil, fmt.Errorf("checkpoint.Load: unmarshal metadata: %w", err)
	}

	// Skip padding to 8-byte boundary.
	headerLen := 8 + 4 + 4 + int(metaLen)
	pad := (8 - (headerLen % 8)) % 8
	if pad > 0 {
		if _, err := io.CopyN(io.Discard, br, int64(pad)); err != nil {
			return nil, fmt.Errorf("checkpoint.Load: skip padding: %w", err)
		}
	}

	ckpt := &Checkpoint{
		Step:         meta.Step,
		BestValLoss:  meta.BestValLoss,
		Config:       meta.Config,
		OptimizerCfg: meta.OptimizerConfig,
		Tensors:      make(map[string]*tensor.Tensor),
	}

	// Tensors until EOF.
	for {
		name, t, err := readTensor(br)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("checkpoint.Load: read tensor: %w", err)
		}
		ckpt.Tensors[name] = t
	}

	return ckpt, nil
}

// readTensor reads a single named tensor record. Returns io.EOF cleanly when
// positioned at the end of the stream with no bytes remaining.
func readTensor(r io.Reader) (string, *tensor.Tensor, error) {
	var nameLen uint32
	if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
		// Clean EOF before any bytes of this record were read.
		if errors.Is(err, io.EOF) {
			return "", nil, io.EOF
		}
		return "", nil, fmt.Errorf("read name length: %w", err)
	}
	nameBytes := make([]byte, nameLen)
	if _, err := io.ReadFull(r, nameBytes); err != nil {
		return "", nil, fmt.Errorf("read name: %w", err)
	}
	name := string(nameBytes)

	var rank uint32
	if err := binary.Read(r, binary.LittleEndian, &rank); err != nil {
		return "", nil, fmt.Errorf("read rank: %w", err)
	}
	shape32 := make([]int32, rank)
	if err := binary.Read(r, binary.LittleEndian, shape32); err != nil {
		return "", nil, fmt.Errorf("read shape: %w", err)
	}
	shape := make([]int, rank)
	numel := 1
	for i, d := range shape32 {
		shape[i] = int(d)
		numel *= int(d)
	}

	var dataBytes uint64
	if err := binary.Read(r, binary.LittleEndian, &dataBytes); err != nil {
		return "", nil, fmt.Errorf("read data length: %w", err)
	}
	expected := uint64(numel) * 4
	if dataBytes != expected {
		return "", nil, fmt.Errorf("data length %d != expected %d (shape %v)", dataBytes, expected, shape)
	}

	t := tensor.New(shape...)
	if err := binary.Read(r, binary.LittleEndian, t.Data); err != nil {
		return "", nil, fmt.Errorf("read data: %w", err)
	}
	return name, t, nil
}

// PackModel packs model weights and optimizer state into a Checkpoint.
// Optimizer state (float32 slices) is wrapped as 1-D tensors.
func PackModel(
	g interface {
		NamedParameters() map[string]*tensor.Tensor
	},
	o interface {
		StateTensors() map[string][]float32
	},
	step int, bestValLoss float32,
	cfg model.GPTConfig, optCfg optim.AdamWConfig,
) *Checkpoint {
	tensors := make(map[string]*tensor.Tensor)
	for name, t := range g.NamedParameters() {
		tensors["model."+name] = t
	}
	for name, data := range o.StateTensors() {
		t := tensor.New(len(data))
		copy(t.Data, data)
		tensors["optim."+name] = t
	}
	return &Checkpoint{
		Step:         step,
		BestValLoss:  bestValLoss,
		Config:       cfg,
		OptimizerCfg: optCfg,
		Tensors:      tensors,
	}
}

// UnpackModel restores model weights and optimizer state from a Checkpoint,
// reversing the naming prefixes applied by PackModel.
func UnpackModel(ckpt *Checkpoint,
	g interface {
		LoadNamedParameters(map[string]*tensor.Tensor)
	},
	o interface {
		LoadState(map[string][]float32)
	},
) {
	modelParams := make(map[string]*tensor.Tensor)
	optState := make(map[string][]float32)
	for name, t := range ckpt.Tensors {
		switch {
		case len(name) > 6 && name[:6] == "model.":
			modelParams[name[6:]] = t
		case len(name) > 6 && name[:6] == "optim.":
			data := make([]float32, len(t.Data))
			copy(data, t.Data)
			optState[name[6:]] = data
		}
	}
	g.LoadNamedParameters(modelParams)
	o.LoadState(optState)
}
