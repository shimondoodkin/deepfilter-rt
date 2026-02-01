//! # DeepFilter-RT
//!
//! Real-time noise suppression using DeepFilterNet neural network with ONNX Runtime inference.
//!
//! This crate provides audio enhancement for real-time applications such as voice calls,
//! streaming, and live audio processing. It implements the two-stage DeepFilterNet algorithm:
//! ERB-based spectral masking followed by deep filtering on low frequencies.
//!
//! ## Quick Start
//!
//! Use [`DeepFilterStream`] for the simplest API - it handles buffering internally:
//!
//! ```ignore
//! use deepfilter_rt::DeepFilterStream;
//!
//! let mut stream = DeepFilterStream::new(Path::new("models/dfn3"))?;
//! stream.warmup()?;
//!
//! // Process any length of audio (must be 48kHz mono f32)
//! let enhanced = stream.process(&input_samples)?;
//! ```
//!
//! ## Model Variants
//!
//! | Variant | Folder | Latency | Mode | Use Case |
//! |---------|--------|---------|------|----------|
//! | DeepFilterNet2 | `dfn2` | 30ms | Stateless | General purpose |
//! | DeepFilterNet2-LL | `dfn2_ll` | 10ms | Stateless | Low-latency |
//! | DeepFilterNet2-H0 | `dfn2_h0` | 30ms | Stateful | Best quality (GRU) |
//! | DeepFilterNet3 | `dfn3` | 30ms | Stateless | High quality |
//! | DeepFilterNet3-LL | `dfn3_ll` | 10ms | Stateless | Real-time |
//! | DeepFilterNet3-H0 | `dfn3_h0` | 30ms | Stateful | Best quality (GRU) |
//!
//! Variant is auto-detected from the model folder name and `config.ini`.
//!
//! ## Audio Requirements
//!
//! - **Sample rate**: 48 kHz (resample before processing if needed)
//! - **Format**: Mono f32 samples in range [-1.0, 1.0]
//!
//! ## API Levels
//!
//! - [`DeepFilterStream`] - High-level streaming API. Handles buffering internally.
//!   Pass any length of samples, get enhanced audio back.
//!
//! - [`DeepFilterProcessor`] - Low-level frame API for integration with audio callbacks.
//!   You manage 480-sample frames yourself (10ms at 48kHz).
//!
//! ## Thread Count
//!
//! The `with_threads` constructors control ONNX Runtime's intra-op parallelism:
//!
//! - **Real-time audio**: Use 1-2 threads to minimize latency jitter
//! - **Batch/offline**: Use more threads (4-8) for throughput
//! - **Default**: ONNX Runtime picks based on CPU cores
//!
//! ## Hardware Acceleration
//!
//! Enable via Cargo features:
//!
//! ```toml
//! # Android GPU/NPU (NNAPI)
//! deepfilter-rt = { path = "...", features = ["nnapi"] }
//!
//! # Android with fp16 relaxation (faster, slightly lower quality)
//! deepfilter-rt = { path = "...", features = ["nnapi", "fp16"] }
//! ```
//!
//! With `fp16` enabled, the processing pipeline looks like:
//!
//! ```text
//! Audio (i16) → f32 → [STFT f32] → [NNAPI: f32→fp16→inference→fp16→f32] → [ISTFT f32] → f32
//!                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//!                                   Only this part uses fp16 internally
//! ```
//!
//! The API remains f32 throughout. The fp16 benefit is faster matrix ops on NPU/GPU,
//! not reduced conversion overhead.
//!
//! ```toml
//!
//! # iOS/macOS (CoreML)
//! deepfilter-rt = { path = "...", features = ["coreml"] }
//!
//! # NVIDIA GPU (CUDA)
//! deepfilter-rt = { path = "...", features = ["cuda"] }
//! ```
//!
//! ## Android Setup
//!
//! See `ONNX_RUNTIME_ANDROID_SETUP.md` for complete instructions:
//!
//! 1. Download `libonnxruntime.so` from Maven Central (version 1.23.x for ort 2.0.0-rc.11)
//! 2. Place in `android/app/src/main/jniLibs/{arch}/`
//! 3. Build with: `cargo ndk -t arm64-v8a build --release --features "nnapi,fp16"`
//!
//! ## Logging
//!
//! This crate uses the `log` crate for debug/info messages. On Android, configure
//! `android_logger` once in your app's initialization (shared across all Rust crates):
//!
//! ```ignore
//! // In your app's init code (not in deepfilter-rt)
//! android_logger::init_once(
//!     android_logger::Config::default()
//!         .with_max_level(log::LevelFilter::Debug)
//!         .with_tag("MyApp"),
//! );
//! ```
//!
//! View logs with: `adb logcat | grep MyApp`
//!
//! ## Thread Safety
//!
//! Each processor instance is independent and `Send`. Create separate instances
//! for parallel processing - they do not share state.

use deep_filter::DFState;
use ndarray::Array3;
use num_complex::Complex32;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::Once;
use thiserror::Error;

// Global ORT initialization guard - ensures init is called only once
static ORT_INIT: Once = Once::new();
static mut ORT_INIT_ERROR: Option<String> = None;

fn init_ort() -> Result<()> {
    ORT_INIT.call_once(|| {
        // With load-dynamic feature, we need to specify where to find the library
        #[cfg(target_os = "android")]
        let lib_name = "libonnxruntime.so";

        #[cfg(target_os = "windows")]
        let lib_name = "onnxruntime";

        #[cfg(all(not(target_os = "android"), not(target_os = "windows")))]
        let lib_name = "libonnxruntime";

        match ort::init_from(lib_name) {
            Ok(builder) => {
                let _ = builder.with_name("deepfilter").commit();
            }
            Err(e) => unsafe {
                ORT_INIT_ERROR = Some(format!("Failed to load ONNX Runtime from '{}': {}", lib_name, e));
            }
        }
    });

    // Check if initialization failed
    unsafe {
        if let Some(ref err) = ORT_INIT_ERROR {
            return Err(DfError::Config(err.clone()));
        }
    }
    Ok(())
}

// Common parameters for all DeepFilterNet models
pub const SAMPLE_RATE: usize = 48000;
pub const FFT_SIZE: usize = 960;
pub const HOP_SIZE: usize = 480;
pub const FREQ_SIZE: usize = 481;
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
pub const DF_ORDER: usize = 5;
pub const DEFAULT_NORM_ALPHA: f32 = 0.99;
// Match DeepFilterNet CLI defaults (enhance_wav.rs)
pub const MIN_DB_THRESH: f32 = -15.0;
pub const MAX_DB_ERB_THRESH: f32 = 35.0;
pub const MAX_DB_DF_THRESH: f32 = 35.0;

/// Internal inference mode (derived from ModelVariant).
#[derive(Debug, Clone, Copy, PartialEq)]
enum InferenceMode {
    StatefulH0,
    StatelessWindowLast,
}

impl InferenceMode {
    /// Minimum frames needed before inference can run.
    fn min_required_frames(self, lookahead: usize) -> usize {
        match self {
            InferenceMode::StatefulH0 => 1,
            InferenceMode::StatelessWindowLast => lookahead + 1,
        }
    }
}

#[derive(Error, Debug)]
pub enum DfError {
    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),
    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Config error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, DfError>;

/// Model variant (auto-detected from folder name and config.ini).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelVariant {
    /// Standard DeepFilterNet2 (30ms latency, stateless)
    DeepFilterNet2,
    /// Low-latency DeepFilterNet2 (10ms latency, stateless)
    DeepFilterNet2LL,
    /// Stateful DeepFilterNet2 with GRU (30ms latency, best quality)
    DeepFilterNet2H0,
    /// Standard DeepFilterNet3 (30ms latency, stateless)
    DeepFilterNet3,
    /// Low-latency DeepFilterNet3 (10ms latency, stateless)
    DeepFilterNet3LL,
    /// Stateful DeepFilterNet3 with GRU (30ms latency, best quality)
    DeepFilterNet3H0,
}

impl ModelVariant {
    /// Detect variant from model directory (folder name + config.ini).
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let folder_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        // Check for H0 suffix first (stateful models)
        let is_h0 = folder_name.ends_with("_h0");

        // Read config for model type and lookahead
        let config_path = model_dir.join("config.ini");
        let content = fs::read_to_string(&config_path).unwrap_or_default();
        let params = parse_ini(&content);

        let model = params.get("model").map(|s| s.as_str()).unwrap_or("deepfilternet3");
        let lookahead = params.get("df_lookahead")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(2);

        match (model, lookahead, is_h0) {
            ("deepfilternet2", _, true) => Ok(ModelVariant::DeepFilterNet2H0),
            ("deepfilternet2", 0, false) => Ok(ModelVariant::DeepFilterNet2LL),
            ("deepfilternet2", _, false) => Ok(ModelVariant::DeepFilterNet2),
            ("deepfilternet3", _, true) => Ok(ModelVariant::DeepFilterNet3H0),
            ("deepfilternet3", 0, false) => Ok(ModelVariant::DeepFilterNet3LL),
            ("deepfilternet3", _, false) => Ok(ModelVariant::DeepFilterNet3),
            (_, _, true) => Ok(ModelVariant::DeepFilterNet3H0),
            (_, 0, false) => Ok(ModelVariant::DeepFilterNet3LL),
            _ => Ok(ModelVariant::DeepFilterNet3),
        }
    }

    /// Whether this is a low-latency variant (10ms vs 30ms).
    pub fn is_low_latency(&self) -> bool {
        matches!(self, ModelVariant::DeepFilterNet2LL | ModelVariant::DeepFilterNet3LL)
    }

    /// Whether this is a stateful (H0/GRU) variant.
    pub fn is_stateful(&self) -> bool {
        matches!(self, ModelVariant::DeepFilterNet2H0 | ModelVariant::DeepFilterNet3H0)
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ModelVariant::DeepFilterNet2 => "DeepFilterNet2",
            ModelVariant::DeepFilterNet2LL => "DeepFilterNet2-LL",
            ModelVariant::DeepFilterNet2H0 => "DeepFilterNet2-H0",
            ModelVariant::DeepFilterNet3 => "DeepFilterNet3",
            ModelVariant::DeepFilterNet3LL => "DeepFilterNet3-LL",
            ModelVariant::DeepFilterNet3H0 => "DeepFilterNet3-H0",
        }
    }

    /// Internal inference mode for this variant.
    fn inference_mode(&self) -> InferenceMode {
        if self.is_stateful() {
            InferenceMode::StatefulH0
        } else {
            InferenceMode::StatelessWindowLast
        }
    }
}

/// Simple INI parser
fn parse_ini(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('[') || line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    map
}

/// Real-time DeepFilterNet processor
pub struct DeepFilterProcessor {
    encoder: Session,
    erb_decoder: Session,
    df_decoder: Session,
    df_state: DFState,
    rolling_spec_buf_x: VecDeque<Vec<Complex32>>, // noisy spec
    rolling_spec_buf_y: VecDeque<Vec<Complex32>>, // stage-1 enhanced spec
    erb_feat_buf: VecDeque<Vec<f32>>,
    spec_feat_buf: VecDeque<Vec<Complex32>>,
    enc_feat_buf_erb: VecDeque<Vec<f32>>,
    enc_feat_buf_spec: VecDeque<Vec<Complex32>>,
    frames_processed: usize, // Track frames to know when buffer has valid data
    lookahead: usize,
    enc_h: Vec<f32>,
    enc_hidden_dim: usize,
    enc_window: usize,
    inference_mode: InferenceMode,
    norm_alpha: f32,
    variant: ModelVariant,
    debug_dump_frame: Option<usize>,
}

impl DeepFilterProcessor {
    /// Create processor from model directory.
    ///
    /// Auto-detects model variant from folder name and config.ini.
    ///
    /// The directory should contain:
    /// - enc.onnx, erb_dec.onnx, df_dec.onnx
    /// - config.ini (for variant detection)
    pub fn new(model_dir: &Path) -> Result<Self> {
        let variant = ModelVariant::from_model_dir(model_dir).unwrap_or(ModelVariant::DeepFilterNet3);
        Self::with_variant_and_threads(model_dir, variant, None)
    }

    /// Create processor with explicit thread count.
    ///
    /// Controls ONNX Runtime's intra-op parallelism (threads used within each inference call).
    /// - For real-time audio: use 1-2 to minimize latency variance
    /// - For batch/offline: use 4+ for throughput
    pub fn with_threads(model_dir: &Path, intra_threads: usize) -> Result<Self> {
        let variant = ModelVariant::from_model_dir(model_dir).unwrap_or(ModelVariant::DeepFilterNet3);
        Self::with_variant_and_threads(model_dir, variant, Some(intra_threads))
    }

    /// Create processor with explicit variant and thread count.
    ///
    /// Use this when you want to override auto-detection (e.g., folder name doesn't match model).
    pub fn with_variant_and_threads(
        model_dir: &Path,
        variant: ModelVariant,
        intra_threads: Option<usize>,
    ) -> Result<Self> {
        Self::build(model_dir, variant, intra_threads)
    }

    /// Internal builder with all parameters.
    fn build(
        model_dir: &Path,
        variant: ModelVariant,
        intra_threads: Option<usize>,
    ) -> Result<Self> {
        init_ort()?;
        let inference_mode = variant.inference_mode();

        let (df_lookahead, conv_lookahead, min_nb_erb_freqs, norm_alpha, enc_kernel_t, enc_hidden_dim) = {
            let config_path = model_dir.join("config.ini");
            let content = fs::read_to_string(&config_path).unwrap_or_default();
            let params = parse_ini(&content);
            let df_lookahead = params
                .get("df_lookahead")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2);
            let conv_lookahead = params
                .get("conv_lookahead")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let min_nb_erb_freqs = params
                .get("min_nb_erb_freqs")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1);
            let enc_kernel_t = params
                .get("conv_kernel_inp")
                .and_then(|s| s.split(',').next())
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(3);
            let enc_hidden_dim = params
                .get("emb_hidden_dim")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(256);
            let norm_alpha = if let Some(tau) = params
                .get("norm_tau")
                .and_then(|s| s.parse::<f32>().ok())
            {
                let dt = HOP_SIZE as f32 / SAMPLE_RATE as f32;
                let a = f32::exp(-dt / tau);
                // Match python get_norm_alpha rounding behavior
                let mut precision: u32 = 3;
                let mut rounded = 1.0f32;
                while rounded >= 1.0 {
                    let scale = 10f32.powi(precision as i32);
                    rounded = (a * scale).round() / scale;
                    precision += 1;
                }
                rounded
            } else {
                DEFAULT_NORM_ALPHA
            };
            (df_lookahead, conv_lookahead, min_nb_erb_freqs, norm_alpha, enc_kernel_t, enc_hidden_dim)
        };
        let lookahead = df_lookahead.max(conv_lookahead);

        let build_session = |path: std::path::PathBuf| -> Result<Session> {
            let mut builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?;

            if let Some(threads) = intra_threads {
                builder = builder.with_intra_threads(threads)?;
            }

            // Add execution providers based on features
            // Note: with_execution_providers consumes builder, so we rebuild on failure
            #[cfg(feature = "nnapi")]
            {
                use ort::execution_providers::NNAPIExecutionProvider;
                let nnapi = {
                    let ep = NNAPIExecutionProvider::default();
                    #[cfg(feature = "fp16")]
                    let ep = {
                        log::debug!("NNAPI: fp16 relaxation enabled");
                        ep.with_fp16(true)
                    };
                    ep
                };
                builder = builder.with_execution_providers([nnapi.build()])?;
                log::info!("NNAPI execution provider registered");
            }

            #[cfg(feature = "coreml")]
            {
                use ort::execution_providers::CoreMLExecutionProvider;
                let coreml = {
                    let ep = CoreMLExecutionProvider::default();
                    #[cfg(feature = "fp16")]
                    let ep = {
                        log::debug!("CoreML: low precision accumulation enabled");
                        ep.with_low_precision_accumulation_on_gpu(true)
                    };
                    ep
                };
                builder = builder.with_execution_providers([coreml.build()])?;
                log::info!("CoreML execution provider registered");
            }

            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                let cuda = CUDAExecutionProvider::default();
                builder = builder.with_execution_providers([cuda.build()])?;
                log::info!("CUDA execution provider registered");
            }

            Ok(builder.commit_from_file(path)?)
        };

        let encoder = build_session(model_dir.join("enc.onnx"))?;
        let erb_decoder = build_session(model_dir.join("erb_dec.onnx"))?;
        let df_decoder = build_session(model_dir.join("df_dec.onnx"))?;
        let has_h0 = encoder.inputs().iter().any(|i| i.name() == "h0");
        if inference_mode == InferenceMode::StatefulH0 && !has_h0 {
            return Err(DfError::Config(
                "Model directory ends with _h0 but encoder has no h0 input".to_string(),
            ));
        }
        let enc_window = match inference_mode {
            InferenceMode::StatefulH0 => enc_kernel_t,
            InferenceMode::StatelessWindowLast => enc_kernel_t + 2 * lookahead.max(1),
        };

        let mut df_state = DFState::new(SAMPLE_RATE, FFT_SIZE, HOP_SIZE, NB_ERB, min_nb_erb_freqs);
        df_state.init_norm_states(NB_DF);

        let buf_x_len = DF_ORDER.max(lookahead.max(1));
        let mut rolling_spec_buf_x = VecDeque::with_capacity(buf_x_len);
        for _ in 0..buf_x_len {
            rolling_spec_buf_x.push_back(vec![Complex32::new(0.0, 0.0); FREQ_SIZE]);
        }
        let mut rolling_spec_buf_y = VecDeque::with_capacity(DF_ORDER + lookahead);
        for _ in 0..(DF_ORDER + lookahead) {
            rolling_spec_buf_y.push_back(vec![Complex32::new(0.0, 0.0); FREQ_SIZE]);
        }
        let erb_feat_buf = VecDeque::with_capacity(lookahead + 1);
        let spec_feat_buf = VecDeque::with_capacity(lookahead + 1);
        let enc_feat_buf_erb = VecDeque::with_capacity(enc_window);
        let enc_feat_buf_spec = VecDeque::with_capacity(enc_window);
        let enc_h = vec![0.0f32; enc_hidden_dim];

        let debug_dump_frame = std::env::var("DF_DEBUG_DUMP_FRAME")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());

        Ok(Self {
            encoder,
            erb_decoder,
            df_decoder,
            df_state,
            rolling_spec_buf_x,
            rolling_spec_buf_y,
            erb_feat_buf,
            spec_feat_buf,
            enc_feat_buf_erb,
            enc_feat_buf_spec,
            frames_processed: 0,
            lookahead,
            enc_h,
            enc_hidden_dim,
            enc_window,
            inference_mode,
            norm_alpha,
            variant,
            debug_dump_frame,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn reset(&mut self) {
        self.df_state.reset();
        self.df_state.init_norm_states(NB_DF);
        self.frames_processed = 0;
        // Keep debug_dump_frame across resets
        self.rolling_spec_buf_x.clear();
        let buf_x_len = DF_ORDER.max(self.lookahead.max(1));
        for _ in 0..buf_x_len {
            self.rolling_spec_buf_x
                .push_back(vec![Complex32::new(0.0, 0.0); FREQ_SIZE]);
        }
        self.rolling_spec_buf_y.clear();
        for _ in 0..(DF_ORDER + self.lookahead) {
            self.rolling_spec_buf_y
                .push_back(vec![Complex32::new(0.0, 0.0); FREQ_SIZE]);
        }
        self.erb_feat_buf.clear();
        self.spec_feat_buf.clear();
        self.enc_feat_buf_erb.clear();
        self.enc_feat_buf_spec.clear();
        self.enc_h.fill(0.0);
    }

    /// Perform warm-up inference to avoid cold-start latency
    ///
    /// NNAPI and other execution providers often have significant first-inference
    /// latency (100ms+). Call this during initialization to "wake up" the GPU/NPU.
    /// The processor state is reset after warmup.
    pub fn warmup(&mut self) -> Result<()> {
        let dummy_input = vec![0.0f32; HOP_SIZE];
        let mut dummy_output = vec![0.0f32; HOP_SIZE];

        // Run a few frames to fully warm up the pipeline
        for _ in 0..3 {
            self.process_frame(&dummy_input, &mut dummy_output)?;
        }

        // Reset state so warmup doesn't affect real processing
        self.reset();
        Ok(())
    }

    /// Process single frame (480 samples @ 48kHz = 10ms)
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        debug_assert_eq!(input.len(), HOP_SIZE);
        debug_assert_eq!(output.len(), HOP_SIZE);

        // 1. STFT
        let mut spec = vec![Complex32::new(0.0, 0.0); FREQ_SIZE];
        self.df_state.analysis(input, &mut spec);

        // 2. Store ORIGINAL spectrum in rolling buffers BEFORE any processing (for DF)
        self.rolling_spec_buf_x.pop_front();
        self.rolling_spec_buf_x.push_back(spec.clone());
        self.rolling_spec_buf_y.pop_front();
        self.rolling_spec_buf_y.push_back(spec.clone());

        // 3. ERB features
        let mut erb_feat = vec![0.0f32; NB_ERB];
        self.df_state.feat_erb(&spec, self.norm_alpha, &mut erb_feat);

        // 4. Spec features - use separate input/output buffers
        let spec_feat_input: Vec<Complex32> = spec[..NB_DF].to_vec();
        let mut spec_feat_cplx = vec![Complex32::new(0.0, 0.0); NB_DF];
        self.df_state
            .feat_cplx(&spec_feat_input, self.norm_alpha, &mut spec_feat_cplx);

        // 5. Buffer features for lookahead alignment (pad_feat behavior)
        self.erb_feat_buf.push_back(erb_feat.clone());
        self.spec_feat_buf.push_back(spec_feat_cplx.clone());
        if self.erb_feat_buf.len() <= self.lookahead {
            // Not enough lookahead frames yet
            self.frames_processed = self.frames_processed.saturating_add(1);
            let mut out_spec = vec![Complex32::new(0.0, 0.0); FREQ_SIZE];
            self.df_state.synthesis(&mut out_spec, output);
            return Ok(());
        }
        // Use lookahead frame for encoder input and build temporal context
        let inf_erb = self.erb_feat_buf[self.lookahead].clone();
        let inf_spec = self.spec_feat_buf[self.lookahead].clone();
        // Shift buffer: remove oldest once consumed for lookahead
        if self.erb_feat_buf.len() > self.lookahead + 1 {
            self.erb_feat_buf.pop_front();
        }
        if self.spec_feat_buf.len() > self.lookahead + 1 {
            self.spec_feat_buf.pop_front();
        }
        self.enc_feat_buf_erb.push_back(inf_erb);
        self.enc_feat_buf_spec.push_back(inf_spec);
        let min_required = self.inference_mode.min_required_frames(self.lookahead);
        if self.enc_feat_buf_erb.len() < min_required {
            // Not enough temporal context yet
            self.frames_processed = self.frames_processed.saturating_add(1);
            let mut out_spec = vec![Complex32::new(0.0, 0.0); FREQ_SIZE];
            self.df_state.synthesis(&mut out_spec, output);
            return Ok(());
        }
        if self.enc_feat_buf_erb.len() > self.enc_window {
            self.enc_feat_buf_erb.pop_front();
            self.enc_feat_buf_spec.pop_front();
        }
        let enc_erb_seq: Vec<Vec<f32>> = self.enc_feat_buf_erb.iter().cloned().collect();
        let enc_spec_seq: Vec<Vec<Complex32>> = self.enc_feat_buf_spec.iter().cloned().collect();
        let take_idx = enc_erb_seq.len().saturating_sub(1);
        let (mask, df_coefs, lsnr) =
            self.run_inference_seq(&enc_erb_seq, &enc_spec_seq, take_idx)?;

        if let Some(target) = self.debug_dump_frame {
            if self.frames_processed == target {
                self.dump_debug_frame(target, &erb_feat, &spec_feat_cplx, &mask, &df_coefs)?;
            }
        }

        // 6. Apply ERB mask to the centered frame in the rolling buffer (df_order - 1)
        let apply_mask = lsnr <= MAX_DB_ERB_THRESH;
        let apply_zero_mask = lsnr < MIN_DB_THRESH;
        if let Some(center_spec) = self.rolling_spec_buf_y.get_mut(DF_ORDER - 1) {
            if apply_mask {
                if apply_zero_mask {
                    let zeros = vec![0.0f32; NB_ERB];
                    self.df_state.apply_mask(center_spec, &zeros);
                } else {
                    self.df_state.apply_mask(center_spec, &mask);
                }
            }
        }

        // 7. Output the centered frame (implicit lookahead via rolling buffer)
        self.frames_processed = self.frames_processed.saturating_add(1);
        let mut out_spec = self
            .rolling_spec_buf_y
            .get(DF_ORDER - 1)
            .cloned()
            .unwrap_or_else(|| vec![Complex32::new(0.0, 0.0); FREQ_SIZE]);

        let apply_df = lsnr <= MAX_DB_DF_THRESH && !apply_zero_mask;
        if apply_df && self.rolling_spec_buf_x.len() >= DF_ORDER {
            self.apply_deep_filter(&mut out_spec, &df_coefs);
        }

        // 8. ISTFT
        self.df_state.synthesis(&mut out_spec, output);

        Ok(())
    }

    fn run_inference_seq(
        &mut self,
        erb_feat_seq: &[Vec<f32>],
        spec_feat_seq: &[Vec<Complex32>],
        take_idx: usize,
    ) -> Result<(Vec<f32>, Array3<f32>, f32)> {
        let t = erb_feat_seq.len();
        debug_assert_eq!(t, spec_feat_seq.len());
        // ERB tensor: [1, 1, T, 32]
        let mut erb_data = vec![0.0f32; t * NB_ERB];
        for (ti, frame) in erb_feat_seq.iter().enumerate() {
            let base = ti * NB_ERB;
            erb_data[base..base + NB_ERB].copy_from_slice(frame);
        }
        let erb_tensor = Tensor::from_array(([1usize, 1, t, NB_ERB], erb_data))?;

        // Spec tensor: [1, 2, T, 96] - channels = [real, imag]
        let mut spec_data = vec![0.0f32; 2 * t * NB_DF];
        // channel 0: real
        for (ti, frame) in spec_feat_seq.iter().enumerate() {
            let base = ti * NB_DF;
            for (fi, c) in frame.iter().enumerate() {
                spec_data[base + fi] = c.re;
            }
        }
        // channel 1: imag
        let ch1_offset = t * NB_DF;
        for (ti, frame) in spec_feat_seq.iter().enumerate() {
            let base = ch1_offset + ti * NB_DF;
            for (fi, c) in frame.iter().enumerate() {
                spec_data[base + fi] = c.im;
            }
        }
        let spec_tensor = Tensor::from_array(([1usize, 2, t, NB_DF], spec_data))?;

        // Encoder (optionally stateful)
        let enc_outputs = if self.inference_mode == InferenceMode::StatefulH0 {
            let h0_tensor = Tensor::from_array(([1usize, 1, self.enc_hidden_dim], self.enc_h.clone()))?;
            self.encoder.run(ort::inputs![
                "feat_erb" => erb_tensor,
                "feat_spec" => spec_tensor,
                "h0" => h0_tensor,
            ])?
        } else {
            self.encoder.run(ort::inputs![
                "feat_erb" => erb_tensor,
                "feat_spec" => spec_tensor,
            ])?
        };
        // Extract encoder outputs as (shape, data) tuples
        let (e0_shape, e0_data) = enc_outputs["e0"].try_extract_tensor::<f32>()?;
        let (e1_shape, e1_data) = enc_outputs["e1"].try_extract_tensor::<f32>()?;
        let (e2_shape, e2_data) = enc_outputs["e2"].try_extract_tensor::<f32>()?;
        let (e3_shape, e3_data) = enc_outputs["e3"].try_extract_tensor::<f32>()?;
        let (emb_shape, emb_data) = enc_outputs["emb"].try_extract_tensor::<f32>()?;
        let (c0_shape, c0_data) = enc_outputs["c0"].try_extract_tensor::<f32>()?;
        let (_lsnr_shape, lsnr_data) = enc_outputs["lsnr"].try_extract_tensor::<f32>()?;

        let e0_vec = e0_data.to_vec();
        let e1_vec = e1_data.to_vec();
        let e2_vec = e2_data.to_vec();
        let e3_vec = e3_data.to_vec();
        let emb_vec: Vec<f32> = emb_data.to_vec();
        let c0_vec: Vec<f32> = c0_data.to_vec();
        let lsnr_vec: Vec<f32> = lsnr_data.to_vec();
        let lsnr = lsnr_vec.get(take_idx).copied().unwrap_or(0.0);
        if self.inference_mode == InferenceMode::StatefulH0 {
            let (_h1_shape, h1_data) = enc_outputs["h1"].try_extract_tensor::<f32>()?;
            let h1_vec = h1_data.to_vec();
            if h1_vec.len() >= self.enc_hidden_dim {
                self.enc_h.copy_from_slice(&h1_vec[..self.enc_hidden_dim]);
            }
        }

        // Helper to convert shape from [i64] to Vec<usize>
        let to_shape = |s: &ort::tensor::Shape| -> Vec<usize> {
            s.iter().map(|&d| d as usize).collect()
        };

        // Convert to tensors for ERB decoder
        let e0_tensor = Tensor::from_array((to_shape(e0_shape), e0_data.to_vec()))?;
        let e1_tensor = Tensor::from_array((to_shape(e1_shape), e1_data.to_vec()))?;
        let e2_tensor = Tensor::from_array((to_shape(e2_shape), e2_data.to_vec()))?;
        let e3_tensor = Tensor::from_array((to_shape(e3_shape), e3_data.to_vec()))?;
        let e0_shape_vec = to_shape(e0_shape);
        let e1_shape_vec = to_shape(e1_shape);
        let e2_shape_vec = to_shape(e2_shape);
        let e3_shape_vec = to_shape(e3_shape);
        let emb_shape_vec = to_shape(emb_shape);
        let c0_shape_vec = to_shape(c0_shape);
        let emb_tensor = Tensor::from_array((emb_shape_vec.clone(), emb_vec.clone()))?;
        let c0_tensor = Tensor::from_array((c0_shape_vec.clone(), c0_vec.clone()))?;

        // Drop outputs to release mutable borrow before debug dumping
        drop(enc_outputs);

        if let Some(target) = self.debug_dump_frame {
            if self.frames_processed == target {
                self.dump_debug_tensor("enc_e0", target, e0_shape_vec, &e0_vec)?;
                self.dump_debug_tensor("enc_e1", target, e1_shape_vec, &e1_vec)?;
                self.dump_debug_tensor("enc_e2", target, e2_shape_vec, &e2_vec)?;
                self.dump_debug_tensor("enc_e3", target, e3_shape_vec, &e3_vec)?;
                self.dump_debug_tensor("enc_emb", target, emb_shape_vec.clone(), &emb_vec)?;
                self.dump_debug_tensor("enc_c0", target, c0_shape_vec, &c0_vec)?;
                self.dump_debug_tensor("enc_lsnr", target, vec![1], &lsnr_vec)?;
            }
        }

        // ERB decoder
        let erb_dec_outputs = self.erb_decoder.run(ort::inputs![
            "emb" => emb_tensor,
            "e3" => e3_tensor,
            "e2" => e2_tensor,
            "e1" => e1_tensor,
            "e0" => e0_tensor,
        ])?;
        let (mask_shape, mask_data) = erb_dec_outputs["m"].try_extract_tensor::<f32>()?;
        let mask_flat: Vec<f32> = mask_data.to_vec();
        // mask shape: [1,1,T,NB_ERB]
        let mask_t = mask_shape[2] as usize;
        let mask_offset = take_idx.min(mask_t.saturating_sub(1)) * NB_ERB;
        let mask: Vec<f32> = mask_flat[mask_offset..mask_offset + NB_ERB].to_vec();

        // DF decoder - need fresh tensors
        let emb_tensor2 = Tensor::from_array((emb_shape_vec, emb_vec))?;
        let df_dec_outputs = self.df_decoder.run(ort::inputs![
            "emb" => emb_tensor2,
            "c0" => c0_tensor,
        ])?;

        let (coefs_shape, coefs_data) = df_dec_outputs["coefs"].try_extract_tensor::<f32>()?;
        let coefs_flat: Vec<f32> = coefs_data.to_vec();
        let coefs_t = coefs_shape[1] as usize;
        let coefs_frame_size = NB_DF * DF_ORDER * 2;
        let coefs_offset = take_idx.min(coefs_t.saturating_sub(1)) * coefs_frame_size;
        let coefs_slice = &coefs_flat[coefs_offset..coefs_offset + coefs_frame_size];
        // Model outputs shape [1, 1, 96, 10] = [batch, time, nb_df, df_order*2]
        // Reshape to [nb_df, df_order, 2] for our indexing
        let df_coefs = Array3::from_shape_vec(
            (NB_DF, DF_ORDER, 2),
            Self::reshape_df_coefs(coefs_slice),
        )?;

        Ok((mask, df_coefs, lsnr))
    }

    fn reshape_df_coefs(coefs: &[f32]) -> Vec<f32> {
        // Input layout: [1, 1, 96, 10] flattened = [freq][order*2]
        // where for each freq: [re0, im0, re1, im1, re2, im2, re3, im3, re4, im4]
        // Output layout: [nb_df, df_order, 2] = [freq][order][re/im]
        coefs.to_vec()
    }

    fn dump_debug_frame(
        &self,
        frame_idx: usize,
        erb_feat: &[f32],
        spec_feat: &[Complex32],
        mask: &[f32],
        df_coefs: &Array3<f32>,
    ) -> Result<()> {
        let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("debug_dump");
        std::fs::create_dir_all(&out_dir)?;

        let write_f32 = |name: &str, data: &[f32]| -> Result<()> {
            let path = out_dir.join(format!("{}_frame_{}.txt", name, frame_idx));
            let mut s = String::new();
            for (i, v) in data.iter().enumerate() {
                s.push_str(&format!("{},{}\n", i, v));
            }
            std::fs::write(path, s)?;
            Ok(())
        };

        let write_cplx = |name: &str, data: &[Complex32]| -> Result<()> {
            let path = out_dir.join(format!("{}_frame_{}.txt", name, frame_idx));
            let mut s = String::new();
            for (i, v) in data.iter().enumerate() {
                s.push_str(&format!("{},{},{}\n", i, v.re, v.im));
            }
            std::fs::write(path, s)?;
            Ok(())
        };

        write_f32("erb_feat", erb_feat)?;
        write_cplx("spec_feat_cplx", spec_feat)?;
        write_f32("mask", mask)?;

        // df_coefs shape: [NB_DF, DF_ORDER, 2]
        let mut df_lines = String::new();
        for f in 0..NB_DF {
            for o in 0..DF_ORDER {
                let re = df_coefs[[f, o, 0]];
                let im = df_coefs[[f, o, 1]];
                df_lines.push_str(&format!("{},{},{},{}\n", f, o, re, im));
            }
        }
        let df_path = out_dir.join(format!("df_coefs_frame_{}.txt", frame_idx));
        std::fs::write(df_path, df_lines)?;

        Ok(())
    }

    fn dump_debug_tensor(
        &self,
        name: &str,
        frame_idx: usize,
        shape: Vec<usize>,
        data: &[f32],
    ) -> Result<()> {
        let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("debug_dump");
        std::fs::create_dir_all(&out_dir)?;
        let path = out_dir.join(format!("{}_frame_{}.txt", name, frame_idx));
        let mut s = String::new();
        s.push_str(&format!("# shape: {:?}\n", shape));
        for (i, v) in data.iter().enumerate() {
            s.push_str(&format!("{},{}\n", i, v));
        }
        std::fs::write(path, s)?;
        Ok(())
    }

    fn apply_deep_filter(&mut self, spec: &mut [Complex32], df_coefs: &Array3<f32>) {
        // Buffer contains the last DF_ORDER frames of ORIGINAL spectrum (before mask)
        // Iterate from oldest to newest, multiply by coefs order 0..DF_ORDER-1
        debug_assert!(self.rolling_spec_buf_x.len() >= DF_ORDER);

        for freq in 0..NB_DF {
            let mut filtered = Complex32::new(0.0, 0.0);

            for (order, frame) in self.rolling_spec_buf_x.iter().take(DF_ORDER).enumerate() {
                let orig_spec = frame[freq];
                let coef = Complex32::new(df_coefs[[freq, order, 0]], df_coefs[[freq, order, 1]]);
                filtered += orig_spec * coef;
            }

            // Replace only the DF frequency bins with filtered output
            spec[freq] = filtered;
        }
    }
}

/// High-level streaming API with automatic buffering.
///
/// This is the recommended API for most use cases. Pass any length of audio,
/// get enhanced audio back. Handles frame buffering internally.
///
/// ```ignore
/// let mut stream = DeepFilterStream::new(Path::new("models/dfn3"))?;
/// stream.warmup()?;
///
/// // Process any chunk size - buffering is handled internally
/// let enhanced = stream.process(&audio_chunk)?;
/// let remaining = stream.flush()?; // Get any buffered samples at end
/// ```
pub struct DeepFilterStream {
    processor: DeepFilterProcessor,
    input_buffer: Vec<f32>,
}

impl DeepFilterStream {
    /// Create stream from model directory.
    ///
    /// Auto-detects model variant from folder name and config.ini.
    pub fn new(model_dir: &Path) -> Result<Self> {
        Ok(Self {
            processor: DeepFilterProcessor::new(model_dir)?,
            input_buffer: Vec::with_capacity(HOP_SIZE * 2),
        })
    }

    /// Create stream with explicit thread count.
    ///
    /// Controls ONNX Runtime's intra-op parallelism:
    /// - For real-time audio: use 1-2 to minimize latency variance
    /// - For batch/offline: use 4+ for throughput
    pub fn with_threads(model_dir: &Path, intra_threads: usize) -> Result<Self> {
        Ok(Self {
            processor: DeepFilterProcessor::with_threads(model_dir, intra_threads)?,
            input_buffer: Vec::with_capacity(HOP_SIZE * 2),
        })
    }

    /// Create stream with explicit variant and thread count.
    ///
    /// Use when folder name doesn't match model or you need to override detection.
    pub fn with_variant_and_threads(
        model_dir: &Path,
        variant: ModelVariant,
        intra_threads: usize,
    ) -> Result<Self> {
        Ok(Self {
            processor: DeepFilterProcessor::with_variant_and_threads(
                model_dir,
                variant,
                Some(intra_threads),
            )?,
            input_buffer: Vec::with_capacity(HOP_SIZE * 2),
        })
    }

    /// Warm up the inference engine.
    ///
    /// Call once after construction to avoid cold-start latency on first real audio.
    /// Especially important for GPU/NPU backends (NNAPI, CoreML, CUDA).
    pub fn warmup(&mut self) -> Result<()> {
        self.processor.warmup()
    }

    /// Process audio samples.
    ///
    /// Input: 48kHz mono f32 samples, any length.
    /// Output: Enhanced samples. May be shorter than input due to internal buffering.
    /// Call [`flush`](Self::flush) at end of stream to get remaining samples.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.input_buffer.extend_from_slice(input);
        let mut output = Vec::new();

        while self.input_buffer.len() >= HOP_SIZE {
            let frame_in: Vec<f32> = self.input_buffer.drain(..HOP_SIZE).collect();
            let mut frame_out = vec![0.0f32; HOP_SIZE];
            self.processor.process_frame(&frame_in, &mut frame_out)?;
            output.extend_from_slice(&frame_out);
        }

        Ok(output)
    }

    /// Flush remaining buffered samples.
    ///
    /// Call at end of stream to get any samples still in the internal buffer.
    /// Pads with zeros if needed to complete the final frame.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        if self.input_buffer.is_empty() {
            return Ok(Vec::new());
        }
        let valid_len = self.input_buffer.len();
        self.input_buffer.resize(HOP_SIZE, 0.0);
        let frame_in: Vec<f32> = self.input_buffer.drain(..).collect();
        let mut frame_out = vec![0.0f32; HOP_SIZE];
        self.processor.process_frame(&frame_in, &mut frame_out)?;
        Ok(frame_out[..valid_len].to_vec())
    }

    /// Reset processor state and clear buffers.
    ///
    /// Call between separate audio streams to avoid artifacts from previous audio.
    pub fn reset(&mut self) {
        self.processor.reset();
        self.input_buffer.clear();
    }

    /// Get the detected model variant.
    pub fn variant(&self) -> ModelVariant { self.processor.variant() }

    /// Required sample rate (48000 Hz).
    pub fn sample_rate(&self) -> usize { SAMPLE_RATE }

    /// Algorithmic latency in milliseconds.
    ///
    /// - 10ms for low-latency (LL) variants
    /// - 30ms for standard variants (includes 2-frame lookahead)
    pub fn latency_ms(&self) -> f32 {
        let base = (FFT_SIZE - HOP_SIZE) as f32 / SAMPLE_RATE as f32 * 1000.0;
        if self.processor.variant.is_low_latency() {
            base // 10ms
        } else {
            base + 20.0 // +2 frames lookahead = 30ms total
        }
    }

    /// Access the underlying processor for advanced use.
    pub fn processor_mut(&mut self) -> &mut DeepFilterProcessor {
        &mut self.processor
    }
}

// Verify that processors can be sent between threads
// This is a compile-time check - if it compiles, the types are Send
fn _assert_send<T: Send>() {}
fn _assert_processor_is_send() {
    _assert_send::<DeepFilterProcessor>();
    _assert_send::<DeepFilterStream>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(FFT_SIZE / 2 + 1, FREQ_SIZE);
    }
}
