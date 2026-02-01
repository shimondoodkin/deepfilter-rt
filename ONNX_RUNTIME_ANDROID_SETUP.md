# ONNX Runtime on Android with Rust - Full Setup Guide

## 1. Cargo.toml Dependencies

```toml
[dependencies]
ort = { version = "2.0.0-rc.9", default-features = false, features = [
    "ndarray",      # For tensor operations
    "load-dynamic", # Load libonnxruntime.so at runtime (required for Android)
    "nnapi"         # Android NNAPI for GPU/NPU acceleration
] }
ndarray = "0.17"    # Tensor array operations (ort 2.0.0-rc.9 uses ndarray 0.17)
```

## 2. Download libonnxruntime.so from Maven Central

**Important:** ort crate 2.0.0-rc.9/rc.11 requires ONNX Runtime 1.23.x

```bash
# Download from Maven Central (NOT GitHub releases!)
curl -sL "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar" -o /tmp/onnxruntime.aar

# Extract .so files (AAR is just a zip)
cd /tmp && unzip -o onnxruntime.aar "jni/*"

# Copy to jniLibs for each architecture you need (in final poduct all of them)
cp jni/arm64-v8a/libonnxruntime.so /path/to/flutter_app/android/app/src/main/jniLibs/arm64-v8a/
cp jni/armeabi-v7a/libonnxruntime.so /path/to/flutter_app/android/app/src/main/jniLibs/armeabi-v7a/
cp jni/x86_64/libonnxruntime.so /path/to/flutter_app/android/app/src/main/jniLibs/x86_64/
```

**Version compatibility table:**
| ort crate | ONNX Runtime |
|-----------|--------------|
| 2.0.0-rc.9/rc.11 | 1.23.x |
| 1.16.x | 1.16.x |

## 3. Rust Initialization Code

```rust
use std::sync::Once;

static ORT_INIT: Once = Once::new();

fn init_ort() -> anyhow::Result<()> {
    let mut result: anyhow::Result<()> = Ok(());

    ORT_INIT.call_once(|| {
        #[cfg(target_os = "android")]
        {
            // Load from the .so bundled in jniLibs
            match ort::init_from("libonnxruntime.so") {
                Ok(_) => log::info!("ONNX Runtime initialized"),
                Err(e) => {
                    log::error!("ORT init failed: {}", e);
                    result = Err(anyhow::anyhow!("ORT init failed: {}", e));
                }
            }
        }

        #[cfg(not(target_os = "android"))]
        {
            // Desktop: ort finds the library automatically
            let _ = ort::init();
        }
    });

    result
}
```

## 4. Create Session with NNAPI (GPU/NPU)

```rust
use ort::session::{Session, builder::GraphOptimizationLevel};

fn create_session(model_bytes: &[u8]) -> anyhow::Result<Session> {
    // Must init first
    init_ort()?;

    let builder = Session::builder()?;

    // Try NNAPI for hardware acceleration
    #[cfg(target_os = "android")]
    let builder = {
        use ort::execution_providers::NNAPIExecutionProvider;

        let nnapi = NNAPIExecutionProvider::default();
        match builder.with_execution_providers([nnapi.build()]) {
            Ok(b) => {
                log::info!("NNAPI enabled (GPU/NPU acceleration)");
                b
            }
            Err(e) => {
                log::warn!("NNAPI failed: {}, using CPU", e);
                Session::builder()?
            }
        }
    };

    let session = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_memory(model_bytes)?;

    Ok(session)
}
```

## 5. Run Inference

```rust
use ort::value::Tensor;

fn run_inference(session: &mut Session, input_data: Vec<f32>) -> anyhow::Result<Vec<f32>> {
    // Create input tensor [batch=1, features=128]
    let input_tensor = Tensor::from_array(([1usize, 128], input_data))?;

    // Run inference
    let outputs = session.run(ort::inputs![
        "input_name" => input_tensor,
    ])?;

    // Extract output
    let (shape, output_data) = outputs["output_name"].try_extract_tensor::<f32>()?;

    Ok(output_data.to_vec())
}
```

## 6. Android Gradle Config (if using Flutter)

In `android/app/build.gradle`:
```gradle
android {
    // Ensure jniLibs are included
    sourceSets {
        main {
            jniLibs.srcDirs = ['src/main/jniLibs']
        }
    }
}
```

## 7. Project Structure

```
your_project/
├── rust/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
└── android/
    └── app/
        └── src/
            └── main/
                └── jniLibs/
                    └── arm64-v8a/
                        └── libonnxruntime.so   # ~19MB from Maven AAR
```

## 8. Logging Setup for Android

```toml
# In Cargo.toml
[target.'cfg(target_os = "android")'.dependencies]
android_logger = "0.14"
log = "0.4"
```

```rust
// In lib.rs
#[cfg(target_os = "android")]
fn init_logging() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug)
            .with_tag("MyRustLib"),
    );
}
```

## 9. Thread Safety for Shared Sessions

```rust
use std::sync::Arc;
use parking_lot::Mutex;

// Session::run() requires &mut self, so wrap in Mutex
pub struct SharedModel {
    session: Mutex<Session>,
}

impl SharedModel {
    pub fn new(model_bytes: &[u8]) -> anyhow::Result<Arc<Self>> {
        let session = create_session(model_bytes)?;
        Ok(Arc::new(Self {
            session: Mutex::new(session),
        }))
    }

    pub fn infer(&self, input: Vec<f32>) -> anyhow::Result<Vec<f32>> {
        let mut session = self.session.lock();
        run_inference(&mut session, input)
    }
}
```

## 10. Loading Models with Config from Archive

For models that include configuration (like DeepFilterNet), you can bundle them in a tar.gz archive.

**Additional dependencies:**
```toml
flate2 = "1.0"      # Gzip decompression
tar = "0.4"         # Tar archive extraction
rust-ini = "0.21"   # INI config parsing
```

**Archive structure:**
```
model.tar.gz
├── config.ini      # Model configuration
├── encoder.onnx    # ONNX model file(s)
├── decoder.onnx
└── ...
```

**Example config.ini:**
```ini
[model]
sr = 48000
hop_size = 480
fft_size = 960

[features]
nb_erb = 32
nb_df = 96
```

**Loading code:**
```rust
use std::io::{Cursor, Read};
use flate2::read::GzDecoder;
use tar::Archive;
use ini::Ini;

pub struct ModelConfig {
    pub sample_rate: usize,
    pub hop_size: usize,
    pub fft_size: usize,
}

pub fn load_model_archive(archive_bytes: &[u8]) -> anyhow::Result<(ModelConfig, Vec<u8>)> {
    // Decompress gzip
    let tar = GzDecoder::new(Cursor::new(archive_bytes));
    let mut archive = Archive::new(tar);

    let mut model_bytes = Vec::new();
    let mut config_str = String::new();

    // Extract files from archive
    for entry in archive.entries()? {
        let mut file = entry?;
        let path = file.path()?.to_path_buf();
        let filename = path.file_name()
            .map(|f| f.to_string_lossy())
            .unwrap_or_default();

        // Skip macOS metadata files
        if filename.starts_with("._") {
            continue;
        }

        if filename == "config.ini" {
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            config_str = String::from_utf8_lossy(&bytes).into_owned();
        } else if filename.ends_with(".onnx") {
            file.read_to_end(&mut model_bytes)?;
        }
    }

    // Parse config
    let ini = Ini::load_from_str(&config_str)?;
    let model_section = ini.section(Some("model"))
        .ok_or_else(|| anyhow::anyhow!("Missing [model] section"))?;

    let config = ModelConfig {
        sample_rate: model_section.get("sr")
            .ok_or_else(|| anyhow::anyhow!("Missing sr"))?.parse()?,
        hop_size: model_section.get("hop_size")
            .ok_or_else(|| anyhow::anyhow!("Missing hop_size"))?.parse()?,
        fft_size: model_section.get("fft_size")
            .ok_or_else(|| anyhow::anyhow!("Missing fft_size"))?.parse()?,
    };

    Ok((config, model_bytes))
}
```

**Loading from Flutter assets:**
```dart
// In Dart/Flutter
final modelData = await rootBundle.load('assets/model.tar.gz');
await initModel(modelData: modelData.buffer.asUint8List());
```

```rust
// In Rust (called from Flutter)
pub fn init_model(model_data: Vec<u8>) -> Result<(), String> {
    let (config, onnx_bytes) = load_model_archive(&model_data)
        .map_err(|e| e.to_string())?;

    log::info!("Model config: sr={}, hop_size={}",
        config.sample_rate, config.hop_size);

    let session = create_session(&onnx_bytes)
        .map_err(|e| e.to_string())?;

    // Store session for later use...
    Ok(())
}
```

## 11. Build Commands for Android

**Prerequisites:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install cargo-ndk
cargo install cargo-ndk

# Add Android targets
rustup target add aarch64-linux-android    # ARM64 (most devices)
rustup target add armv7-linux-androideabi  # ARM32 (older devices)
rustup target add x86_64-linux-android     # x86_64 (emulators)
```

**Set NDK path:**
```bash
export ANDROID_NDK_HOME=/path/to/android-sdk/ndk/27.0.12077973
# Or add to ~/.bashrc / ~/.zshrc
```

**Build for Android:**
```bash
# Single architecture (ARM64)
cargo ndk -t arm64-v8a build --release

# Multiple architectures
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 build --release
```

**Output locations:**
```
target/aarch64-linux-android/release/libYOUR_CRATE.so   # ARM64
target/armv7-linux-androideabi/release/libYOUR_CRATE.so # ARM32
target/x86_64-linux-android/release/libYOUR_CRATE.so    # x86_64
```

**Copy to jniLibs:**
```bash
cp target/aarch64-linux-android/release/libmy_lib.so \
   android/app/src/main/jniLibs/arm64-v8a/

cp target/armv7-linux-androideabi/release/libmy_lib.so \
   android/app/src/main/jniLibs/armeabi-v7a/

cp target/x86_64-linux-android/release/libmy_lib.so \
   android/app/src/main/jniLibs/x86_64/
```

**Recommended Cargo.toml release profile:**
```toml
[profile.release]
lto = true       # Link-time optimization
opt-level = 3    # Maximum optimization
strip = true     # Strip symbols (smaller binary)
```

## Key Notes

- **load-dynamic** is essential - without it, ort tries to link statically which fails on Android
- **Download from Maven Central** - the AAR file contains the correct .so for Android
- **NNAPI** may not accelerate all operations - unsupported ops fall back to CPU
- **Session is `Send + Sync`** but `run()` needs `&mut self`, so wrap in `Mutex` for sharing
- Model files can be loaded from assets or bundled as bytes
- Use `commit_from_memory()` to load models from byte arrays (useful for bundled assets)
- Check model input/output names with `session.inputs()` and `session.outputs()`

## Debugging Tips

1. **Check model I/O:**
```rust
for input in session.inputs() {
    log::info!("Input: {}", input.name());
}
for output in session.outputs() {
    log::info!("Output: {}", output.name());
}
```

2. **View logs:**
```bash
adb logcat | grep "MyRustLib"
```

3. **Common errors:**
- "dlopen failed" / "Library not found" → Check libonnxruntime.so is in jniLibs/arm64-v8a/
- "ort is not compatible with ONNX Runtime binary" → Version mismatch, use 1.23.x with ort 2.0.0-rc.9+
- "Invalid model" → Ensure ONNX model version is compatible
- NNAPI failures → Some ops aren't supported, falls back to CPU automatically

## Additional: libc++_shared.so

Some devices/setups require the C++ standard library. Extract from NDK:

```bash
# Find in your NDK installation
find $ANDROID_NDK_HOME -name "libc++_shared.so" | grep arm64

# Usually at:
# $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so

# Copy to jniLibs alongside libonnxruntime.so
cp $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
   android/app/src/main/jniLibs/arm64-v8a/
```

## ort 2.0 API Reference

The ort 2.0 API has significant changes from 1.x. Key differences:

### Session Creation

```rust
use ort::session::{Session, builder::GraphOptimizationLevel};

// ort 2.0 - builder pattern
let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_memory(model_bytes)?;

// Load from file instead of memory
let session = Session::builder()?
    .commit_from_file("model.onnx")?;
```

### Tensor Creation

```rust
use ort::value::Tensor;

// From Vec with shape
let tensor = Tensor::from_array(([1usize, 32, 96], data_vec))?;

// Different ranks
let tensor_1d = Tensor::from_array(([128], vec))?;
let tensor_2d = Tensor::from_array(([1, 128], vec))?;
let tensor_3d = Tensor::from_array(([1, 2, 64], vec))?;
let tensor_4d = Tensor::from_array(([1, 1, 32, 96], vec))?;
```

### Running Inference

```rust
// Using ort::inputs! macro (named inputs)
let outputs = session.run(ort::inputs![
    "input_name" => tensor1,
    "another_input" => tensor2,
])?;

// Access outputs by name
let output_value = &outputs["output_name"];

// Extract tensor data
let (shape, data) = output_value.try_extract_tensor::<f32>()?;
// shape: &[i64] - dimensions
// data: ArrayView - use .to_vec() to get Vec<f32>

let output_vec: Vec<f32> = data.to_vec();
```

### Execution Providers

```rust
use ort::execution_providers::{
    NNAPIExecutionProvider,      // Android GPU/NPU
    CoreMLExecutionProvider,     // iOS/macOS
    CUDAExecutionProvider,       // NVIDIA GPU
    TensorRTExecutionProvider,   // NVIDIA TensorRT
};

// NNAPI (Android)
let nnapi = NNAPIExecutionProvider::default();
let session = Session::builder()?
    .with_execution_providers([nnapi.build()])?
    .commit_from_memory(model_bytes)?;

// NNAPI with options
let nnapi = NNAPIExecutionProvider::default()
    .with_cpu_disabled()           // Disable CPU fallback
    .with_fp16();                  // Use FP16 precision

// Check if EP is available
if NNAPIExecutionProvider::is_available()? {
    // Use NNAPI
}
```

### Inspecting Model Inputs/Outputs

```rust
// Get input info
for (i, input) in session.inputs().iter().enumerate() {
    log::info!("Input {}: name='{}', type={:?}",
        i, input.name(), input.input_type());
}

// Get output info
for (i, output) in session.outputs().iter().enumerate() {
    log::info!("Output {}: name='{}'", i, output.name());
}
```

### Key API Types

```rust
// Main types
use ort::session::Session;
use ort::value::{Tensor, Value};
use ort::execution_providers::NNAPIExecutionProvider;

// Session is Send + Sync, but run() needs &mut self
// Wrap in Mutex for multi-threaded access:
use parking_lot::Mutex;
let session: Mutex<Session> = Mutex::new(session);
```

### Error Handling

```rust
use ort::Error as OrtError;

match session.run(ort::inputs!["x" => tensor]) {
    Ok(outputs) => { /* success */ }
    Err(OrtError::...) => { /* handle specific error */ }
}

// Or use anyhow
use anyhow::{Context, Result};
let outputs = session.run(ort::inputs!["x" => tensor])
    .context("Inference failed")?;
```

### Changes from ort 1.x to 2.0

| Feature | ort 1.x | ort 2.0 |
|---------|---------|---------|
| Session creation | `SessionBuilder::new()?.with_model_from_memory()` | `Session::builder()?.commit_from_memory()` |
| Tensor creation | `Value::from_array()` | `Tensor::from_array()` |
| Run inference | `session.run(vec![input])` | `session.run(ort::inputs!["name" => tensor])` |
| Output access | `outputs[0].try_extract()` | `outputs["name"].try_extract_tensor()` |
| Init runtime | `ort::init().commit()` | `ort::init()` or `ort::init_from()` |

### Complete Inference Example

```rust
use ort::session::Session;
use ort::value::Tensor;
use anyhow::Result;

pub fn run_model(session: &mut Session, input: Vec<f32>) -> Result<Vec<f32>> {
    // Create input tensor [batch=1, channels=2, time=1, freq=96]
    let input_tensor = Tensor::from_array(([1usize, 2, 1, 96], input))?;

    // Run inference with named input
    let outputs = session.run(ort::inputs![
        "input" => input_tensor,
    ])?;

    // Extract output tensor
    let (shape, data) = outputs["output"].try_extract_tensor::<f32>()?;

    log::debug!("Output shape: {:?}", shape);

    Ok(data.to_vec())
}
```

## Efficient Audio Streaming

### Audio Format Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Microphone  │────▶│  Convert    │────▶│   Model     │────▶│   Output    │
│   (i16)     │     │  i16 → f32  │     │   (f32)     │     │   (f32)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Input: Raw Audio from Microphone

Audio typically comes as **i16** (16-bit signed integer) from Android/Oboe:

```rust
// Oboe callback receives i16 samples
fn audio_callback(data: &[i16]) {
    // Convert i16 [-32768, 32767] to f32 [-1.0, 1.0]
    let samples_f32: Vec<f32> = data.iter()
        .map(|&s| s as f32 / 32768.0)
        .collect();
}
```

### Frame-Based Processing

Models process audio in fixed-size frames (e.g., 480 samples = 10ms at 48kHz):

```rust
const FRAME_SIZE: usize = 480;  // hop_size from model config
const SAMPLE_RATE: usize = 48000;

struct AudioProcessor {
    input_buffer: Vec<f32>,      // Accumulate samples until frame ready
    output_buffer: Vec<f32>,     // Store processed output
}

impl AudioProcessor {
    pub fn push_samples(&mut self, samples: &[f32]) {
        self.input_buffer.extend_from_slice(samples);

        // Process complete frames
        while self.input_buffer.len() >= FRAME_SIZE {
            let frame: Vec<f32> = self.input_buffer.drain(..FRAME_SIZE).collect();
            let processed = self.process_frame(&frame);
            self.output_buffer.extend_from_slice(&processed);
        }
    }

    fn process_frame(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; FRAME_SIZE];
        // ... run model inference ...
        output
    }
}
```

### Overlap-Add for FFT-Based Models

FFT-based models (like DeepFilterNet) use overlap-add synthesis:

```rust
// Model maintains internal state for overlap-add
// IMPORTANT: Always pair analysis() with synthesis()

// WRONG - breaks overlap-add state:
fn process_wrong(&mut self, input: &[f32]) -> Vec<f32> {
    self.df_state.analysis(input, &mut self.spec);
    if skip_processing {
        return input.to_vec();  // ❌ Skips synthesis!
    }
    self.df_state.synthesis(&mut self.spec, &mut output);
    output
}

// CORRECT - maintains overlap-add continuity:
fn process_correct(&mut self, input: &[f32]) -> Vec<f32> {
    self.df_state.analysis(input, &mut self.spec);
    // Optionally modify spectrum here...
    self.df_state.synthesis(&mut self.spec, &mut output);  // ✓ Always call
    output
}
```

### Output Storage Options

**Option 1: Accumulate in memory (small recordings)**
```rust
struct Recorder {
    raw_samples: Vec<f32>,       // Original audio
    processed_samples: Vec<f32>, // After model processing
}
```

**Option 2: Stream to file (large recordings)**
```rust
use hound::{WavWriter, WavSpec};

struct StreamingRecorder {
    writer: WavWriter<BufWriter<File>>,
}

impl StreamingRecorder {
    fn write_frame(&mut self, samples: &[f32]) {
        for &sample in samples {
            // Convert f32 [-1.0, 1.0] to i16
            let sample_i16 = (sample * 32767.0) as i16;
            self.writer.write_sample(sample_i16).unwrap();
        }
    }
}
```

**Option 3: Ring buffer for real-time playback**
```rust
use std::sync::Arc;
use parking_lot::Mutex;

struct RingBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
}

// Producer (processing thread) writes
// Consumer (playback thread) reads
let shared_buffer = Arc::new(Mutex::new(RingBuffer::new(48000))); // 1 sec
```

### Dual Stream Architecture

For processing two audio sources (e.g., mic + speaker):

```rust
struct DualStreamProcessor {
    stream_a: StreamProcessor,  // Microphone
    stream_b: StreamProcessor,  // Speaker/remote audio

    // Shared ONNX sessions (Arc-wrapped)
    sessions: Arc<SharedSessions>,
}

impl DualStreamProcessor {
    fn process(&mut self, mic: &[f32], speaker: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let out_a = self.stream_a.process(mic);
        let out_b = self.stream_b.process(speaker);
        (out_a, out_b)
    }
}
```

### Memory Efficiency Tips

```rust
// Pre-allocate buffers once, reuse
struct Processor {
    frame_buffer: Vec<f32>,   // Reused each frame
    spec_buffer: Vec<f32>,    // Reused for FFT
}

impl Processor {
    fn new() -> Self {
        Self {
            frame_buffer: vec![0.0; FRAME_SIZE],
            spec_buffer: vec![0.0; FFT_SIZE],
        }
    }

    fn process(&mut self, input: &[f32]) {
        // Copy to pre-allocated buffer instead of allocating
        self.frame_buffer.copy_from_slice(input);
        // ... process using self.frame_buffer ...
    }
}
```

### Complete Audio Pipeline Example

```rust
pub struct AudioPipeline {
    processor: StreamProcessor,
    input_buffer: Vec<f32>,
    raw_recording: Vec<f32>,
    processed_recording: Vec<f32>,
}

impl AudioPipeline {
    /// Called from audio callback with raw i16 samples
    pub fn on_audio_input(&mut self, raw_i16: &[i16]) {
        // Convert to f32
        for &sample in raw_i16 {
            let sample_f32 = sample as f32 / 32768.0;
            self.input_buffer.push(sample_f32);
            self.raw_recording.push(sample_f32);
        }

        // Process complete frames
        while self.input_buffer.len() >= FRAME_SIZE {
            let frame: Vec<f32> = self.input_buffer.drain(..FRAME_SIZE).collect();

            let mut output = vec![0.0f32; FRAME_SIZE];
            self.processor.process_frame(&frame, &mut output);

            self.processed_recording.extend_from_slice(&output);
        }
    }

    /// Save both raw and processed to separate WAV files
    pub fn save(&self, raw_path: &str, processed_path: &str) {
        save_wav(raw_path, &self.raw_recording);
        save_wav(processed_path, &self.processed_recording);
    }
}
```

## Quick Reference: Maven Download URLs

```bash
# ONNX Runtime 1.23.2 (latest 1.23.x)
https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar

# Browse all versions:
https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/
```
