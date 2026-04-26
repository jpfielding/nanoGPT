# nanoGPT (Go)

A faithful Go port of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT): a
minimal, hackable character-level GPT with training, sampling, and data
preparation in a single binary. Written in Go with flat `float32` tensors and a
manual forward+backward tape (no autograd library), dispatching matrix multiply
to BLAS via CGo (Accelerate on macOS, OpenBLAS on Linux, pure-Go fallback
elsewhere).

## Status

- End-to-end training and sampling on CPU.
- Initial loss on Shakespeare char-level matches `ln(65) ≈ 4.17`; loss descends
  steadily (validated on short runs; full 5000-iter convergence to ~1.5 val
  loss is the reference target from Python nanoGPT but has not been
  benchmarked yet in Go).
- Resume-from-checkpoint verified end-to-end.
- Gradient correctness covered by finite-difference tests on every op
  (`go test ./tensor/...`).

## Install / Build

Requires Go 1.23+. On macOS, CGo links against the Accelerate framework (no
install step). On Linux, `-lopenblas` is expected; elsewhere a pure-Go naive
SGEMM is used via build tags.

```
git clone https://github.com/jpfielding/nanoGPT.git
cd nanoGPT
go build ./...
```

## Quickstart: Shakespeare char-level

```
# 1. Download and tokenize Tiny Shakespeare.
go run . prepare --shakespeare

# 2. Train (defaults match Python nanoGPT's Shakespeare config).
go run . train --data-dir data/shakespeare_char --out-dir out --max-iters 5000

# 3. Sample from the resulting checkpoint.
go run . sample --ckpt out/ckpt.ckpt --prompt $'\n' --max-new-tokens 500
```

## Subcommands

### `prepare`

Tokenizes a plain-text corpus into character-level `train.bin` / `val.bin` and
writes a `vocab.json`.

```
go run . prepare --input input.txt --out-dir data/mycorpus --val-split 0.1
go run . prepare --shakespeare   # auto-downloads Tiny Shakespeare
```

### `train`

| Flag | Default | Notes |
| --- | --- | --- |
| `--data-dir` | `data/shakespeare_char` | directory with `train.bin`, `val.bin`, `vocab.json` |
| `--out-dir` | `out` | where checkpoints land |
| `--n-layer`, `--n-head`, `--n-embd` | 6, 6, 384 | Shakespeare-sized defaults |
| `--block-size` | 256 | context length T |
| `--batch-size` | 64 | micro-batch |
| `--max-iters` | 5000 | total steps |
| `--lr`, `--min-lr`, `--warmup-iters` | 1e-3, 1e-4, 100 | cosine schedule |
| `--beta1`, `--beta2`, `--weight-decay`, `--eps` | 0.9, 0.95, 0.1, 1e-8 | AdamW |
| `--grad-clip` | 1.0 | global L2 norm; ≤0 disables |
| `--grad-accum-steps` | 1 | micro-batches per optimizer step |
| `--eval-interval`, `--eval-iters` | 250, 200 | val-loss cadence / averaging |
| `--log-interval` | 10 | training-loss log cadence |
| `--resume` | `""` | path to `.ckpt` to resume from |
| `--eval-only` | false | one val-loss pass then exit |
| `--always-save-checkpoint` | false | save each eval, not only on improvement |
| `--seed` | 1337 | RNG seed |

Training log lines report loss, LR, wall time, tokens/sec, and an A100-peak
MFU estimate (informational on CPU).

### `sample`

```
go run . sample \
    --ckpt out/ckpt.ckpt \
    --prompt $'\n' \
    --temperature 0.8 \
    --top-k 200 \
    --max-new-tokens 500 \
    --num-samples 1
```

Use `--prompt FILE:prompt.txt` to read the prompt from a file.

## Repository layout

```
blas/         CGo BLAS wrappers (Accelerate / OpenBLAS / pure-Go fallback)
tensor/       flat float32 tensors + hand-written forward/backward ops
model/        Embedding, Linear, LayerNorm, Attention, MLP, Block, GPT
optim/        AdamW with decoupled weight decay
schedule/     Cosine LR schedule with linear warmup
data/         binary uint16 dataset loader + char-level tokenizer
checkpoint/   custom binary format (magic + JSON meta + tensor records)
cmd/          prepare, train, sample subcommand packages
main.go       CLI dispatch
```

## Design notes

- **No autograd.** Each op returns a `BackwardFn` closure that captures exactly
  the intermediates it needs; the model builds a linear tape during forward
  and walks it in reverse for backward. Inspired by Karpathy's `llm.c`.
- **BLAS for matmul, pure Go for everything else.** `cblas_sgemm` is called
  per-head in attention so each call operates on L2-resident sub-matrices.
- **Weight tying.** `LMHead.Weight == WTE.Weight` (same `*Tensor`).
  `Parameters()` deduplicates via the address of `Data[0]` so AdamW updates
  tied weights exactly once.
- **Numerical stability.** Softmax is max-subtracted, cross-entropy uses the
  logsumexp formulation, LayerNorm backward uses the three-term form with
  `x_hat` saved during forward.
- **Checkpoint format.** Custom binary: `"NANOGPT\x00"` magic + version +
  JSON metadata (model config, step, best val loss, AdamW config) + per-tensor
  records (name, shape, float32 data). Optimizer `m1`/`m2` moments are saved
  alongside weights for bit-identical resume.

## Tests

```
go test ./...
```

Covers finite-difference gradient checks on `Add`, `MatMul`, `LayerNorm`,
`SelfAttention`, `CrossEntropyLoss`, and a shape/forward check for
`GPT.CropBlockSize`.

## Not implemented

- Distributed training (DDP), mixed precision, `torch.compile` equivalents.
- Loading pretrained GPT-2 weights from HuggingFace.
- wandb / tensorboard integration.
- GPU execution (CPU-only; BLAS is CPU-only in this port).

## License

MIT, matching the upstream nanoGPT project.
