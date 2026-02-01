# deepfilter-rt

Real-time speech enhancement using DeepFilterNet with ONNX Runtime in Rust.

## Overview

This crate provides frame-by-frame audio denoising using the [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) neural network. It combines:

- **df crate**: STFT/ISTFT and feature extraction (from DeepFilterNet)
- **ort**: ONNX Runtime for neural network inference

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
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  │
┌──────────────┐  ┌──────────────┐        │
│ 2. ERB Feat  │  │ 3. Spec Feat │        │
│ 481→32 bands │  │ First 96 bins│        │
│ dB + norm    │  │ Unit norm    │        │
└──────────────┘  └──────────────┘        │
    │                  │                  │
    └────────┬─────────┘                  │
             ▼                            │
┌─────────────────────────────────────┐   │
│  4. ENCODER (enc.onnx)              │   │
│     → embeddings, skip connections  │   │
└─────────────────────────────────────┘   │
             │                            │
    ┌────────┴────────┐                   │
    ▼                 ▼                   │
┌──────────────┐  ┌──────────────┐        │
│ 5. ERB DEC   │  │ 6. DF DEC    │        │
│ → mask [32]  │  │ → coefs [5,96,2]      │
└──────────────┘  └──────────────┘        │
    │                 │                   │
    ▼                 ▼                   │
┌──────────────┐  ┌──────────────┐        │
│ 7. Apply Mask│  │ 8. Deep Filter│◄──────┘
│ Expand→481   │  │ Convolve 5 taps│
└──────────────┘  └──────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  9. ISTFT Synthesis                 │
│     → Enhanced audio (480 samples)  │
└─────────────────────────────────────┘
```

## Model Variants

All models downloaded from [DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/tree/main/models).

| Model | Lookahead | Latency | Size | Description |
|-------|-----------|---------|------|-------------|
| `dfn2` | 2 frames | ~30ms | 9MB | DeepFilterNet2, standard |
| `dfn2_ll` | 0 | ~10ms | 9MB | DeepFilterNet2, low latency |
| `dfn3` | 2 frames | ~30ms | 8MB | DeepFilterNet3, improved |
| `dfn3_ll` | 0 | ~10ms | 39MB | DeepFilterNet3-LL, best real-time |

**Common parameters** (all models):
- Sample rate: 48000 Hz
- Frame size: 480 samples (10ms)
- FFT size: 960
- ERB bands: 32
- DF bins: 96
- DF order: 5 taps

## Usage

### Basic

```rust
use deepfilter_rt::{DeepFilterStream, HOP_SIZE};
use std::path::Path;

// Load model (auto-detects variant from config.ini)
let mut stream = DeepFilterStream::new(Path::new("models/dfn3_ll"))?;

// Process audio chunks (any size)
let enhanced = stream.process(&input_samples)?;

// At end of stream
let remaining = stream.flush()?;
```

### Basic (explicit mode)

```rust
use deepfilter_rt::{DeepFilterStream, InferenceMode};
use std::path::Path;

let mut stream = DeepFilterStream::new_with_mode(
    Path::new("models/dfn3_h0"),
    InferenceMode::StatefulH0,
)?;
```

### Frame-by-Frame (Real-time)

```rust
use deepfilter_rt::{DeepFilterProcessor, HOP_SIZE};

let mut processor = DeepFilterProcessor::new(Path::new("models/dfn2_ll"))?;

// In your audio callback (must be exactly 480 samples)
fn audio_callback(input: &[f32; 480], output: &mut [f32; 480]) {
    processor.process_frame(input, output).unwrap();
}
```

### Inference Modes (folder-based)

The runtime mode is selected by the **model folder name**:

- `*_h0` → **stateful** (uses GRU `h0/h1`, best quality)
- otherwise → **stateless** (growing window, take last)

Examples:
- `models/dfn3_h0` (stateful)
- `models/dfn3` (stateless window)

If your deployment does not use folder names (e.g. Android assets), use the
explicit constructors:
- `DeepFilterProcessor::new_with_mode(...)`
- `DeepFilterStream::new_with_mode(...)`

See `MODES.md` and `STATEFUL_ONNX.md` for details.

### Check Model Info

```rust
let stream = DeepFilterStream::new(model_path)?;

println!("Model: {}", stream.variant().name());     // "DeepFilterNet3-LL"
println!("Low latency: {}", stream.variant().is_low_latency()); // true
println!("Latency: {}ms", stream.latency_ms());     // 10.0
println!("Sample rate: {}", stream.sample_rate());  // 48000
println!("Frame size: {}", stream.frame_size());    // 480
```

## Installation

```toml
[dependencies]
deepfilter-rt = { path = "../deepfilter_rt" }
```

### Requirements

- ONNX Runtime (automatically downloaded by `ort` crate)
- Model files: `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, `config.ini`

### Download Models

Models are pre-downloaded in `models/` directory:
- `models/dfn2/` - DeepFilterNet2
- `models/dfn2_ll/` - DeepFilterNet2 Low Latency
- `models/dfn3/` - DeepFilterNet3
- `models/dfn3_ll/` - DeepFilterNet3 Low Latency

Or download manually:
```bash
curl -LO https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3_ll_onnx.tar.gz
tar -xzf DeepFilterNet3_ll_onnx.tar.gz
```

## Examples

```bash
# Process a WAV file
cargo run --example process_file -- input.wav output.wav models/dfn3_ll

# Simulated realtime (streaming) from WAV to WAV
cargo run --example realtime -- input.wav output.wav models/dfn3_ll
```

## Performance

Target: Real-time factor (RTF) < 1.0

| Model | CPU (typical) | Notes |
|-------|---------------|-------|
| dfn2_ll | ~0.1-0.2x | Fast, good for embedded |
| dfn3_ll | ~0.3-0.5x | Larger but better quality |

## API Reference

### `DeepFilterProcessor`

Low-level frame processor.

- `new(model_dir: &Path)` - Load from directory
- `process_frame(&mut self, input: &[f32], output: &mut [f32])` - Process one frame
- `reset(&mut self)` - Reset all states
- `variant(&self) -> ModelVariant` - Get model type

### `DeepFilterStream`

High-level streaming wrapper with buffering.

- `new(model_dir: &Path)` - Load from directory
- `process(&mut self, input: &[f32]) -> Vec<f32>` - Process any chunk size
- `flush(&mut self) -> Vec<f32>` - Get remaining samples
- `reset(&mut self)` - Reset state
- `latency_ms(&self) -> f32` - Get total latency
- `sample_rate(&self) -> usize` - Always 48000
- `frame_size(&self) -> usize` - Always 480

### `ModelVariant`

```rust
pub enum ModelVariant {
    DeepFilterNet2,
    DeepFilterNet2LL,
    DeepFilterNet3,
    DeepFilterNet3LL,
}
```

## More Docs

- `MODES.md` - runtime mode selection and LL vs non-LL behavior
- `STATEFUL_ONNX.md` - how to export stateful ONNX models
- `API.md` - detailed usage notes

## License

MIT / Apache-2.0 (same as DeepFilterNet)

## References

- [DeepFilterNet Paper](https://arxiv.org/abs/2110.05588)
- [DeepFilterNet GitHub](https://github.com/Rikorose/DeepFilterNet)
- [ONNX Runtime Rust](https://github.com/pykeio/ort)
