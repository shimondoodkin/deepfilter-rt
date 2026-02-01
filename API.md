# deepfilter-rt API Notes

This document explains the primary API surface and common usage patterns.

## Core Types

### `DeepFilterProcessor`
Low-level, frame-by-frame processor. Use this for realtime callbacks.

Key methods:

- `new(model_dir: &Path) -> Result<Self>`
  Load models from a directory containing `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, and `config.ini`.

- `new_with_mode(model_dir: &Path, mode: InferenceMode) -> Result<Self>`
  Same as `new`, but forces an explicit inference mode (useful for Android/asset bundles).

- `process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()>`
  Process exactly one frame (480 samples at 48 kHz). Output must be the same size.

- `reset(&mut self)`
  Clears all state and rolling buffers.

- `variant(&self) -> ModelVariant`
  Returns detected model variant based on `config.ini`.

### `DeepFilterStream`
High-level streaming wrapper that accepts arbitrary chunk sizes.

Key methods:

- `new(model_dir: &Path) -> Result<Self>`
- `process(&mut self, input: &[f32]) -> Result<Vec<f32>>`
- `flush(&mut self) -> Result<Vec<f32>>`
- `reset(&mut self)`
- `variant(&self) -> ModelVariant`
- `latency_ms(&self) -> f32`
- `new_with_mode(model_dir: &Path, mode: InferenceMode) -> Result<Self>`
  Force an explicit inference mode for streaming.

## Inference Modes

The runtime mode is chosen by the **model folder name**:

- `*_h0` ? **Stateful (h0)**
  - Uses GRU hidden state inputs/outputs (`h0` / `h1`)
  - Best quality, closest to Python

- Otherwise ? **Stateless (window last)**
  - Uses a growing temporal window and takes the last frame
  - Works for regular ONNX without `h0`

If a folder ends in `_h0` but `enc.onnx` has no `h0` input, initialization fails.
Use `new_with_mode` when you can’t rely on folder names (e.g. Android assets).

## LL vs non-LL

Low-latency variants are selected via `config.ini` (lookahead=0). This affects
algorithmic delay but **not** the runtime mode selection. You can have both:

- `dfn3_h0` (stateful, standard latency)
- `dfn3_ll_h0` (stateful, low latency)
- `dfn3` (stateless, standard latency)
- `dfn3_ll` (stateless, low latency)

## Example Usage

### Realtime callback

```rust
use deepfilter_rt::{DeepFilterProcessor, HOP_SIZE};
use std::path::Path;

let mut processor = DeepFilterProcessor::new(Path::new("models/dfn3_h0"))?;

fn audio_callback(input: &[f32; 480], output: &mut [f32; 480]) {
    processor.process_frame(input, output).unwrap();
}
```

### Streaming chunks

```rust
use deepfilter_rt::DeepFilterStream;
use std::path::Path;

let mut stream = DeepFilterStream::new(Path::new("models/dfn3"))?;

let enhanced = stream.process(&input_samples)?;
let remaining = stream.flush()?;
```

## Troubleshooting

- **Choppy output with stateless models**: use a stateful `_h0` model if available.
- **Model init fails**: check that `enc.onnx` has `h0` if using `_h0` folder.
- **Wrong sample rate**: resample to 48 kHz before processing.

