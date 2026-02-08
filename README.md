# deepfilter-rt

Real-time speech enhancement using [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) with ONNX Runtime in Rust.

## Overview

This crate provides frame-by-frame audio denoising using the DeepFilterNet neural network. It runs a single merged ONNX model per frame for minimal inference overhead, combining:

- **df crate**: STFT/ISTFT and feature extraction (from DeepFilterNet)
- **ort**: ONNX Runtime for neural network inference (CPU, CUDA, NNAPI, CoreML)

## Processing Pipeline

```
Audio Frame (480 samples @ 48kHz = 10ms)
    │
    ▼
┌─────────────────────────────────────┐
│  1. STFT Analysis                   │
│     Vorbis window, 960-pt FFT       │
│     Output: 481 complex bins        │
└─────────────────────────────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌──────────────┐  ┌──────────────┐
│ 2. ERB Feat  │  │ 3. Spec Feat │
│ 481→32 bands │  │ First 96 bins│
│ dB + norm    │  │ Unit norm    │
└──────────────┘  └──────────────┘
    │                  │
    └────────┬─────────┘
             ▼
┌─────────────────────────────────────┐
│  4. COMBINED MODEL (combined.onnx)  │
│     Single ONNX session:            │
│     encoder + ERB decoder + DF dec  │
│     → mask [32], coefs [5,96,2],    │
│       lsnr [1]                      │
└─────────────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────────┐  ┌───────────────┐
│ 5. Apply Mask│  │ 6. Deep Filter│
│ Expand→481   │  │ Convolve 5 taps│
└──────────────┘  └───────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  7. ISTFT Synthesis                 │
│     → Enhanced audio (480 samples)  │
└─────────────────────────────────────┘
```

The three original ONNX models (encoder, ERB decoder, DF decoder) are pre-merged into a single `combined.onnx` per variant. This reduces per-frame overhead from 3 ORT dispatches to 1.

## Model Variants

All models from [DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/tree/main/models), pre-merged.

| Model | Lookahead | Latency | Mode | Description |
|-------|-----------|---------|------|-------------|
| `dfn2` | 2 frames | ~30ms | Stateless | DeepFilterNet2, standard |
| `dfn2_ll` | 0 | ~10ms | Stateless | DeepFilterNet2, low latency |
| `dfn2_h0` | 2 frames | ~30ms | Stateful | DeepFilterNet2, GRU states |
| `dfn3` | 2 frames | ~30ms | Stateless | DeepFilterNet3, improved |
| `dfn3_ll` | 0 | ~10ms | Stateless | DeepFilterNet3-LL, best real-time |
| `dfn3_h0` | 2 frames | ~30ms | Stateful | DeepFilterNet3, best quality |

**Common parameters** (all models):
- Sample rate: 48000 Hz
- Frame size: 480 samples (10ms)
- FFT size: 960
- ERB bands: 32
- DF bins: 96
- DF order: 5 taps

## Installation

```toml
[dependencies]
deepfilter-rt = { git = "https://github.com/shimondoodkin/deepfilter-rt" }
```

### Requirements

- ONNX Runtime dynamic library (auto-downloaded by `ort` on desktop, or provide `libonnxruntime.so` for Android)
- Model files: `combined.onnx` + `config.ini` per variant (included in `models/`)

## Usage

### Streaming API (Recommended)

```rust
use deepfilter_rt::DeepFilterStream;
use std::path::Path;

let mut stream = DeepFilterStream::new(Path::new("models/dfn3_ll"))?;
stream.warmup()?;  // Avoid cold-start latency

// Process audio chunks (any size, 48kHz mono f32)
let enhanced = stream.process(&input_samples)?;

// At end of stream
let remaining = stream.flush()?;
```

### Frame-by-Frame (Real-time Callbacks)

```rust
use deepfilter_rt::{DeepFilterProcessor, HOP_SIZE};
use std::path::Path;

let mut processor = DeepFilterProcessor::new(Path::new("models/dfn3_ll"))?;
processor.warmup()?;

// In your audio callback (must be exactly 480 samples)
let mut output = vec![0.0f32; HOP_SIZE];
processor.process_frame(&input, &mut output)?;
```

### Pipelined Threading

For real-time apps where ONNX inference is too heavy for the audio callback thread, run the denoiser on a dedicated thread with a bounded channel:

```rust
use std::sync::mpsc;

let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(500); // ~5s buffer at 10ms frames

// DeepFilter thread
std::thread::spawn(move || {
    let mut proc = DeepFilterProcessor::new(Path::new("models/dfn3_ll")).unwrap();
    proc.warmup().unwrap();
    let mut out = vec![0.0f32; 480];
    while let Ok(frame) = rx.recv() {
        proc.process_frame(&frame, &mut out).unwrap();
        // use denoised output...
    }
});

// Audio thread - send frames without blocking on inference
tx.send(audio_frame.to_vec()).ok();
```

See `examples/pipelined.rs` for a complete working example.

### Check Model Info

```rust
let stream = DeepFilterStream::new(model_path)?;

println!("Model: {}", stream.variant().name());            // "DeepFilterNet3-LL"
println!("Low latency: {}", stream.variant().is_low_latency()); // true
println!("Stateful: {}", stream.variant().is_stateful());  // false
println!("Latency: {}ms", stream.latency_ms());            // 10.0
println!("Sample rate: {}", stream.sample_rate());          // 48000
```

## Hardware Acceleration

Enable via Cargo features:

```toml
[dependencies]
# NVIDIA GPU (CUDA) — default
deepfilter-rt = { git = "..." }

# Android GPU/NPU (NNAPI)
deepfilter-rt = { git = "...", features = ["nnapi"] }

# Android with fp16 relaxation (faster, slightly lower quality)
deepfilter-rt = { git = "...", features = ["nnapi", "fp16"] }

# iOS/macOS (CoreML)
deepfilter-rt = { git = "...", features = ["coreml"] }

# CPU only (no GPU features)
deepfilter-rt = { git = "...", default-features = false }
```

### fp16 Pipeline

With `fp16` enabled:

```
Audio → f32 → [STFT f32] → [NNAPI: f32→fp16→inference→fp16→f32] → [ISTFT f32] → f32
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                             Only this part uses fp16 internally
```

The API remains f32 throughout. The fp16 benefit is faster matrix ops on NPU/GPU.

## Examples

```bash
# Process a WAV file (high-level streaming API)
cargo run --example process_file -- input.wav output.wav models/dfn3_ll

# Simulated realtime streaming
cargo run --example realtime -- input.wav output.wav models/dfn3_ll

# Pipelined threading (producer/consumer pattern)
cargo run --example pipelined -- input.wav output.wav models/dfn3_ll
```

## Merging Models

Pre-merged `combined.onnx` files are included for all 6 variants. If you need to re-merge (e.g. after updating source models):

```bash
pip install onnx onnxsim

# Merge a single variant
python scripts/merge_onnx.py models/dfn3_ll

# Merge all variants
for dir in models/dfn2 models/dfn2_h0 models/dfn2_ll models/dfn3 models/dfn3_h0 models/dfn3_ll; do
    python scripts/merge_onnx.py "$dir"
done
```

The script merges `enc.onnx`, `erb_dec.onnx`, and `df_dec.onnx` into a single `combined.onnx` with proper tensor name prefixing to avoid ORT buffer collisions.

## Performance

Single merged model reduces per-frame ORT overhead from 3 dispatches to 1. Combined with flat ring buffers and pre-allocated I/O, the hot path has zero heap allocations.

| Model | CPU | CUDA | Notes |
|-------|-----|------|-------|
| dfn2_ll | ~0.1-0.2x RTF | faster | Fast, good for embedded |
| dfn3_ll | ~0.3-0.5x RTF | faster | Larger but better quality |

RTF = Real-Time Factor (< 1.0 means faster than real-time).

## Thread Count

The `with_threads` constructors control ONNX Runtime's intra-op parallelism:

- **Real-time audio**: Use 1-2 threads to minimize latency jitter
- **Batch/offline**: Use more threads (4-8) for throughput
- **Default**: 2 threads

## Android Setup

See `ONNX_RUNTIME_ANDROID_SETUP.md` for complete instructions:

1. Download `libonnxruntime.so` from Maven Central (version 1.23.x for ort 2.0.0-rc.11)
2. Place in `android/app/src/main/jniLibs/{arch}/`
3. Build with: `cargo ndk -t arm64-v8a build --release --features "nnapi,fp16"`

## API Reference

### `DeepFilterStream`

High-level streaming wrapper with internal buffering.

| Method | Description |
|--------|-------------|
| `new(model_dir)` | Load from directory (auto-detect variant) |
| `with_threads(model_dir, n)` | With explicit thread count |
| `process(input) -> Vec<f32>` | Process any chunk size |
| `flush() -> Vec<f32>` | Get remaining samples at end of stream |
| `warmup()` | Warm up inference engine |
| `reset()` | Reset state between streams |
| `latency_ms() -> f32` | Total algorithmic latency |
| `sample_rate() -> usize` | Always 48000 |
| `variant() -> ModelVariant` | Get model type |

### `DeepFilterProcessor`

Low-level frame processor for audio callbacks.

| Method | Description |
|--------|-------------|
| `new(model_dir)` | Load from directory |
| `with_threads(model_dir, n)` | With explicit thread count |
| `process_frame(input, output)` | Process one 480-sample frame |
| `warmup()` | Warm up inference engine |
| `reset()` | Reset all states |
| `variant() -> ModelVariant` | Get model type |

### `ModelVariant`

Auto-detected from folder name and `config.ini`.

| Method | Description |
|--------|-------------|
| `name() -> &str` | e.g. "DeepFilterNet3-LL" |
| `is_low_latency() -> bool` | 10ms vs 30ms |
| `is_stateful() -> bool` | Has GRU hidden state |

## More Docs

- `ONNX_RUNTIME_ANDROID_SETUP.md` - Complete Android/NNAPI setup guide
- `MODES.md` - Runtime mode selection and LL vs non-LL behavior
- `STATEFUL_ONNX.md` - How to export stateful ONNX models
- `API.md` - Detailed usage notes

## License

MIT / Apache-2.0 (same as DeepFilterNet)

## References

- [DeepFilterNet Paper](https://arxiv.org/abs/2110.05588)
- [DeepFilterNet GitHub](https://github.com/Rikorose/DeepFilterNet)
- [ONNX Runtime Rust](https://github.com/pykeio/ort)
