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

## Usage

### Streaming API (Recommended)

```rust
use deepfilter_rt::DeepFilterStream;
use std::path::Path;

// Load model (auto-detects variant from folder name + config.ini)
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

let mut processor = DeepFilterProcessor::new(Path::new("models/dfn2_ll"))?;

// In your audio callback (must be exactly 480 samples)
fn audio_callback(input: &[f32; 480], output: &mut [f32; 480]) {
    processor.process_frame(input, output).unwrap();
}
```

### Check Model Info

```rust
let stream = DeepFilterStream::new(model_path)?;

println!("Model: {}", stream.variant().name());           // "DeepFilterNet3-LL"
println!("Low latency: {}", stream.variant().is_low_latency()); // true
println!("Stateful: {}", stream.variant().is_stateful()); // false
println!("Latency: {}ms", stream.latency_ms());           // 10.0
println!("Sample rate: {}", stream.sample_rate());        // 48000
```

## Hardware Acceleration

Enable via Cargo features:

```toml
[dependencies]
# Android GPU/NPU (NNAPI)
deepfilter-rt = { path = "...", features = ["nnapi"] }

# Android with fp16 relaxation (faster, slightly lower quality)
deepfilter-rt = { path = "...", features = ["nnapi", "fp16"] }

# iOS/macOS (CoreML)
deepfilter-rt = { path = "...", features = ["coreml"] }

# NVIDIA GPU (CUDA)
deepfilter-rt = { path = "...", features = ["cuda"] }
```

### fp16 Pipeline

With `fp16` enabled, the processing pipeline looks like:

```
Audio (i16) → f32 → [STFT f32] → [NNAPI: f32→fp16→inference→fp16→f32] → [ISTFT f32] → f32
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  Only this part uses fp16 internally
```

The API remains f32 throughout. The fp16 benefit is faster matrix ops on NPU/GPU.

## Android Setup

See `ONNX_RUNTIME_ANDROID_SETUP.md` for complete instructions:

1. Download `libonnxruntime.so` from Maven Central (version 1.23.x for ort 2.0.0-rc.11)
2. Place in `android/app/src/main/jniLibs/{arch}/`
3. Build with: `cargo ndk -t arm64-v8a build --release --features "nnapi,fp16"`

### Logging

This crate uses the `log` crate. Configure `android_logger` once in your app:

```rust
android_logger::init_once(
    android_logger::Config::default()
        .with_max_level(log::LevelFilter::Debug)
        .with_tag("MyApp"),
);
```

View logs: `adb logcat | grep MyApp`

## Installation

```toml
[dependencies]
deepfilter-rt = { git = "https://github.com/shimondoodkin/deepfilter-rt" }
```

### Requirements

- ONNX Runtime (automatically downloaded by `ort` crate, or provide `libonnxruntime.so` for Android)
- Model files: `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, `config.ini`

### Download Models

Models are included in `models/` directory:
- `models/dfn2/` - DeepFilterNet2
- `models/dfn2_ll/` - DeepFilterNet2 Low Latency
- `models/dfn2_h0/` - DeepFilterNet2 Stateful
- `models/dfn3/` - DeepFilterNet3
- `models/dfn3_ll/` - DeepFilterNet3 Low Latency
- `models/dfn3_h0/` - DeepFilterNet3 Stateful

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

### `DeepFilterStream`

High-level streaming wrapper with buffering.

- `new(model_dir: &Path)` - Load from directory (auto-detect variant)
- `with_threads(model_dir, threads)` - With explicit thread count
- `process(&mut self, input: &[f32]) -> Vec<f32>` - Process any chunk size
- `flush(&mut self) -> Vec<f32>` - Get remaining samples
- `warmup(&mut self)` - Warm up inference engine
- `reset(&mut self)` - Reset state
- `latency_ms(&self) -> f32` - Get total latency
- `sample_rate(&self) -> usize` - Always 48000
- `variant(&self) -> ModelVariant` - Get model type

### `DeepFilterProcessor`

Low-level frame processor (for audio callbacks).

- `new(model_dir: &Path)` - Load from directory
- `with_threads(model_dir, threads)` - With explicit thread count
- `process_frame(&mut self, input: &[f32], output: &mut [f32])` - Process one 480-sample frame
- `warmup(&mut self)` - Warm up inference engine
- `reset(&mut self)` - Reset all states
- `variant(&self) -> ModelVariant` - Get model type

### `ModelVariant`

```rust
pub enum ModelVariant {
    DeepFilterNet2,
    DeepFilterNet2LL,
    DeepFilterNet2H0,
    DeepFilterNet3,
    DeepFilterNet3LL,
    DeepFilterNet3H0,
}

impl ModelVariant {
    fn name(&self) -> &'static str;
    fn is_low_latency(&self) -> bool;
    fn is_stateful(&self) -> bool;
}
```

## Thread Count

The `with_threads` constructors control ONNX Runtime's intra-op parallelism:

- **Real-time audio**: Use 1-2 threads to minimize latency jitter
- **Batch/offline**: Use more threads (4-8) for throughput
- **Default**: ONNX Runtime picks based on CPU cores

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
