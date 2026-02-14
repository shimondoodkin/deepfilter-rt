# DeepFilterNet Architecture — Complete Technical Reference

This document describes how the DeepFilterNet3 model works end-to-end, from raw
audio to enhanced audio. It covers the Python (PyTorch) reference implementation,
the Rust signal processing layer, and how our ONNX Runtime streaming inference
replicates the pipeline.

## Table of Contents

1. [Overview](#1-overview)
2. [Signal Processing (Rust DFState)](#2-signal-processing-rust-dfstate)
3. [Feature Extraction](#3-feature-extraction)
4. [Encoder](#4-encoder)
5. [ERB Decoder](#5-erb-decoder)
6. [DF Decoder](#6-df-decoder)
7. [Spectral Reconstruction](#7-spectral-reconstruction)
8. [Padding and Lookahead](#8-padding-and-lookahead)
9. [SqueezedGRU_S — The Core Recurrent Unit](#9-squeezedgru_s--the-core-recurrent-unit)
10. [Streaming Inference](#10-streaming-inference)
11. [Configuration Parameters](#11-configuration-parameters)
12. [Shape Reference](#12-shape-reference)
13. [Complete Data Flow Diagram](#13-complete-data-flow-diagram)

---

## 1. Overview

DeepFilterNet3 is a two-stage noise suppression model:

1. **ERB masking** — A learned spectral mask in the ERB (Equivalent Rectangular
   Bandwidth) domain is applied to all 481 frequency bins. This handles broadband
   noise across the full spectrum.

2. **Deep filtering** — Complex-valued FIR filter coefficients are predicted for
   the lowest 96 frequency bins (0–4.8 kHz, the speech-critical range). This
   applies a multi-frame complex convolution that preserves phase information
   better than masking alone.

The model processes audio at 48 kHz in 10ms frames (480 samples). A 960-sample
FFT with 50% overlap produces 481 complex frequency bins per frame.

### Key dimensions

| Symbol | Value | Description |
|--------|-------|-------------|
| sr | 48000 | Sample rate (Hz) |
| fft_size | 960 | FFT window size (samples) |
| hop_size | 480 | Frame hop / STFT stride (samples) |
| freq_size | 481 | FFT bins = fft_size/2 + 1 |
| nb_erb (E) | 32 | ERB frequency bands |
| nb_df | 96 | Deep filter frequency bins |
| df_order (O) | 5 | Deep filter tap count |
| conv_ch (C) | 16 | Base convolutional channel count |
| emb_hidden_dim | 256 | GRU hidden dimension |

---

## 2. Signal Processing (Rust DFState)

Source: `DeepFilterNet/libDF/src/lib.rs`

The Rust `DFState` handles all signal processing outside the neural network:
STFT, ISTFT, feature extraction, ERB mask application, and deep filtering.

### 2.1 STFT (frame_analysis)

Converts a 480-sample audio frame into a 481-bin complex spectrum.

```
Input:  audio[480]          — raw PCM samples
Output: spectrum[481]       — complex frequency bins

Process:
  1. Concatenate overlap memory (480 samples) with new frame (480 samples)
     → 960-sample buffer
  2. Apply Vorbis window: w[n] = sin(π/2 · sin²(π·(n+0.5)/N))
  3. Real-to-complex FFT → 481 complex bins
  4. Normalize by wnorm = 1 / (fft_size² / (2 · hop_size))
  5. Update overlap memory: shift left by hop_size, copy new frame to end
```

The Vorbis window provides perfect reconstruction with 50% overlap and has smooth
onset/offset properties ideal for spectral processing.

### 2.2 ISTFT (frame_synthesis)

Converts a 481-bin complex spectrum back to a 480-sample audio frame.

```
Input:  spectrum[481]       — enhanced complex frequency bins
Output: audio[480]          — enhanced PCM samples

Process:
  1. Complex-to-real IFFT → 960 samples
  2. Apply Vorbis window
  3. Overlap-add with synthesis memory:
     output[i] = ifft_buf[i] + synthesis_mem[i]
  4. Update synthesis memory with tail of current frame
```

### 2.3 ERB Band Structure

ERB bands map the 481 frequency bins into 32 perceptually-spaced bands.
Lower bands (speech fundamentals) have fewer bins; higher bands are wider.

```
erb_fb(sr=48000, fft_size=960, nb_bands=32, min_nb_freqs=2) → Vec<usize>

Algorithm:
  1. Compute ERB scale endpoints:
     erb_low  = 9.265 · ln(1 + 0 / (24.7 · 9.265))
     erb_high = 9.265 · ln(1 + 24000 / (24.7 · 9.265))
  2. Divide [erb_low, erb_high] into 32 uniform steps
  3. Map back to frequency: erb2freq(e) = 24.7 · 9.265 · (exp(e/9.265) − 1)
  4. Round to nearest FFT bin boundaries
  5. Enforce minimum 2 bins per band
  6. Last band absorbs remaining bins up to Nyquist
```

Output: `erb[32]` where `erb[i]` = number of frequency bins in band i.
Sum of all `erb[i]` = 481.

---

## 3. Feature Extraction

Two feature streams are computed from each spectrum frame and fed to the encoder.

### 3.1 ERB Features (feat_erb)

```
Input:  spectrum[481] complex
Output: feat_erb[32] float (normalized dB)

Process:
  1. Band energy: for each ERB band, compute mean |spectrum[f]|²
     → erb_power[32]
  2. Log compression: 10 · log10(erb_power + 1e-10)
     → erb_db[32]
  3. Exponential mean normalization:
     state[i] = erb_db[i] · (1−α) + state[i] · α
     output[i] = (erb_db[i] − state[i]) / 40.0

     where α = exp(−hop_size / (sr · norm_tau))
     Default norm_tau=1.0 → α ≈ 0.99
```

The normalization removes the long-term mean, making features invariant to
recording level. The `/40.0` scaling keeps values in a reasonable range for
the neural network.

### 3.2 Complex Spectrogram Features (feat_spec)

```
Input:  spectrum[481] complex
Output: feat_spec[96] complex (unit-normalized)

Process:
  1. Take first nb_df=96 frequency bins
  2. Per-frequency unit normalization:
     state[f] = |spectrum[f]| · (1−α) + state[f] · α
     output[f] = spectrum[f] / sqrt(state[f])
```

This normalizes each frequency bin by its running magnitude, preserving phase
while standardizing amplitude. Only the first 96 bins are used (the DF range).

### 3.3 Feature shapes entering the model

```
feat_erb:  [B, 1, T, 32]      — 1 channel, T frames, 32 ERB bands
feat_spec: [B, 2, T, 96]      — 2 channels (real, imag), T frames, 96 freq bins
```

The complex spectrum features are split into real/imaginary channels for the
encoder's convolutional layers.

---

## 4. Encoder

Source: `DeepFilterNet/df/deepfilternet3.py`, class `Encoder` (lines 100-185)

The encoder processes both feature streams through parallel convolutional stacks,
combines them, and runs a GRU for temporal modeling.

### 4.1 ERB Convolutional Stack

```
feat_erb [B, 1, T, 32]
  │
  ├─ erb_conv0: Conv2d(1→16, kernel=(3,3), separable)     → e0 [B, 16, T, 32]
  ├─ erb_conv1: Conv2d(16→16, kernel=(1,3), fstride=2)    → e1 [B, 16, T, 16]
  ├─ erb_conv2: Conv2d(16→16, kernel=(1,3), fstride=2)    → e2 [B, 16, T, 8]
  └─ erb_conv3: Conv2d(16→16, kernel=(1,3), fstride=1)    → e3 [B, 16, T, 8]
```

- Each conv layer is a `Conv2dNormAct` with batch norm and ReLU
- Separable = depthwise + pointwise (reduces parameters)
- `erb_conv0` uses kernel (3,3) — the only layer with temporal kernel > 1
- `erb_conv1/2` downsample frequency by 2× via `fstride=2`
- Skip connections e0, e1, e2, e3 are passed to the ERB decoder

### 4.2 Complex Spectrogram Convolutional Stack

```
feat_spec [B, 2, T, 96]
  │
  ├─ df_conv0: Conv2d(2→16, kernel=(3,3), separable)      → c0_full [B, 16, T, 96]
  └─ df_conv1: Conv2d(16→16, kernel=(1,3), fstride=2)     → c1 [B, 16, T, 48]
```

- c0 is passed to the DF decoder as a skip connection
- c1 is flattened for the embedding combination

### 4.3 Embedding Combination

The two streams are merged into a single embedding vector per frame:

```
emb_erb = e3.permute(0,2,3,1).flatten(2)   → [B, T, 16·8] = [B, T, 128]
emb_df  = c1.permute(0,2,3,1).flatten(2)   → [B, T, 16·48] = [B, T, 768]

cemb = df_fc_emb(emb_df)                    → [B, T, 128]
  (GroupedLinearEinsum with 16 groups: 768 → 128)

emb = combine(emb_erb, cemb)
  If enc_concat=False: emb = emb_erb + cemb  → [B, T, 128]
  If enc_concat=True:  emb = cat(emb_erb, cemb) → [B, T, 256]
```

### 4.4 Embedding GRU (emb_gru)

```
emb_gru: SqueezedGRU_S(
    input_size  = 128,    # emb_in_dim = C × E / 4
    hidden_size = 256,    # emb_hidden_dim
    output_size = 128,    # emb_out_dim = C × E / 4
    num_layers  = 1
)

emb, h1 = emb_gru(emb, h0)    → emb [B, T, 128], h1 [1, B, 256]
```

The GRU provides temporal context — the key recurrent component.
See [Section 9](#9-squeezedgru_s--the-core-recurrent-unit) for SqueezedGRU_S details.

### 4.5 Local SNR Estimation

```
lsnr_fc: Linear(128 → 1) + Sigmoid
lsnr = lsnr_fc(emb) × (lsnr_max − lsnr_min) + lsnr_min

Output: lsnr [B, T, 1]  — estimated local SNR in dB, range [−15, 35]
```

LSNR controls whether noise suppression is applied. When LSNR is very low
(pure noise), the model may zero out the output.

### 4.6 Encoder outputs summary

| Output | Shape | Destination |
|--------|-------|-------------|
| e0 | [B, 16, T, 32] | ERB decoder skip connection |
| e1 | [B, 16, T, 16] | ERB decoder skip connection |
| e2 | [B, 16, T, 8] | ERB decoder skip connection |
| e3 | [B, 16, T, 8] | ERB decoder skip connection |
| c0 | [B, 16, T, 96] | DF decoder skip connection |
| emb | [B, T, 128] | Both decoders |
| lsnr | [B, T, 1] | Output / gating |

---

## 5. ERB Decoder

Source: `DeepFilterNet/df/deepfilternet3.py`, class `ErbDecoder` (lines 188-254)

The ERB decoder generates a spectral mask in the 32-band ERB domain.

### 5.1 Decoder GRU

```
emb_gru: SqueezedGRU_S(
    input_size  = 128,
    hidden_size = 256,
    output_size = 128,
    num_layers  = 1    (= emb_num_layers − 1, default emb_num_layers=2)
)

emb, erb_h1 = emb_gru(emb, erb_h0)   → emb [B, T, 128]
```

This is a **separate GRU** from the encoder's — it has its own weights and
hidden state. In the original model this state resets per batch. In our
streaming implementation, we persist it across frames.

### 5.2 Reshape to spatial

```
emb [B, T, 128] → reshape to [B, T, 8, 16] → permute to [B, 16, T, 8]
  (where 8 = E/4, 16 = C)
  Actually: emb [B, T, C·E/8] → [B, C·2, T, E/8]
```

### 5.3 Upsampling path (with skip connections)

```
Level 3 (E/8 → E/4):
  e3_proc = convt3(conv3p(e3) + emb_spatial)   [B, 16, T, 8]

Level 2 (E/4 → E/2):
  e2_proc = convt2(conv2p(e2) + e3_proc)        [B, 16, T, 16]

Level 1 (E/2 → E):
  e1_proc = convt1(conv1p(e1) + e2_proc)        [B, 16, T, 32]

Level 0 (output mask):
  m = conv0_out(conv0p(e0) + e1_proc)            [B, 1, T, 32]
```

- `conv3p`, `conv2p`, `conv1p`, `conv0p` are 1×1 projection convolutions
- `convt3` has no upsampling (stays at E/4=8)
- `convt2` and `convt1` are `ConvTranspose2d` with fstride=2 (2× upsample)
- `conv0_out` projects 16 channels → 1 channel with **sigmoid** activation

### 5.4 Output

```
m [B, 1, T, 32] ∈ [0, 1]    — ERB spectral mask
```

This mask is applied to the full 481-bin spectrum by interpolating across
ERB bands (each band's gain is applied to all frequency bins in that band).

---

## 6. DF Decoder

Source: `DeepFilterNet/df/deepfilternet3.py`, class `DfDecoder` (lines 278-331)

The DF decoder generates complex FIR filter coefficients for the first 96
frequency bins.

### 6.1 DF GRU

```
df_gru: SqueezedGRU_S(
    input_size  = 128,
    hidden_size = 256,
    output_size = None  (outputs hidden_size=256)
    num_layers  = 2     (default df_num_layers)
)

c, df_h1 = df_gru(emb, df_h0)    → c [B, T, 256]
```

Again, a **separate GRU** with its own weights and hidden state.

### 6.2 Skip connection (optional)

```
If df_skip is not None:
    c = c + df_skip(emb)
```

Where `df_skip` can be Identity or GroupedLinear.

### 6.3 DF pathway (from encoder c0)

```
c0 [B, 16, T, 96]
  → df_convp: Conv2d(16 → 10, kernel=(1,1))    → [B, 10, T, 96]
  → permute to [B, T, 96, 10]
```

This provides a "direct path" from the encoder's spectrogram features
to the DF coefficients, adding spatial detail that the GRU embedding
may not fully capture.

### 6.4 Output coefficients

```
df_out: GroupedLinear(256 → 96·10) + Tanh
c_out = df_out(c)                           → [B, T, 960]
c_out = c_out.view(B, T, 96, 10) + c0_path → [B, T, 96, 10]
```

The 10 = df_order(5) × 2 (real + imaginary parts of complex coefficients).

Final reshape for application:
```
coefs [B, T, 96, 10] → [B, 5, T, 96, 2]   — 5 taps, 96 bins, complex
```

---

## 7. Spectral Reconstruction

The model outputs are applied to the noisy spectrum in two stages.

### 7.1 ERB Mask Application

```
For each ERB band i (with gain m[i]):
  For each frequency bin f in band i:
    spec_masked[f] = spec_noisy[f] × m[i]
```

The sigmoid-bounded mask m ∈ [0,1] attenuates noise while preserving speech.
Applied to all 481 frequency bins.

### 7.2 Deep Filtering (first 96 bins)

Deep filtering applies a learned complex FIR filter across time:

```
For each frequency bin f ∈ [0, 96):
  For each frame t:
    spec_df[t, f] = Σ(o=0 to 4) spec_masked[t−2+o, f] × coef[t, o, f]

Where:
  - coef[t, o, f] is complex (from DF decoder output)
  - spec_masked is the ERB-masked spectrum
  - The filter spans df_order=5 frames centered around t
    (with df_lookahead=2 frames into the future in batch mode)
```

### 7.3 Final combination

```
enhanced_spec[0:96]   = deep_filtered output
enhanced_spec[96:481] = ERB-masked output (no DF for high frequencies)
```

The split at bin 96 (~4.8 kHz) is because:
- Speech fundamentals and harmonics are below ~4 kHz
- Deep filtering is most valuable for preserving speech phase
- Higher frequencies benefit sufficiently from ERB masking alone

### 7.4 Optional post-filter

If enabled (default `pf_beta=0.02`), a perceptual post-filter is applied:

```
pf_gain = (1 + β) / (1 + β · (m / m_sin)²)
spec_enhanced *= pf_gain
```

This slightly reduces residual noise at the cost of minor speech distortion.

---

## 8. Padding and Lookahead

### 8.1 pad_feat — Feature lookahead

```python
pad_feat = nn.ConstantPad2d((0, 0, -conv_lookahead, conv_lookahead), 0.0)
```

Applied to `feat_erb` and `feat_spec` **before** the encoder convolutions.

- Removes the first `conv_lookahead` frames from the start
- Adds `conv_lookahead` zero frames at the end
- Net effect: at time t, the model sees features from time t+conv_lookahead

With `conv_lookahead=2` (default for non-LL models):
- The model effectively looks 2 frames (20ms) into the future
- This is **impossible to replicate in causal streaming**
- Our streaming implementation accepts the quality gap (~3 dB SNR)

### 8.2 pad_spec — DF lookahead

```python
pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -df_lookahead, df_lookahead), 0.0)
```

Applied to the input spectrum for deep filtering, providing future context
for the FIR filter coefficients.

### 8.3 Causal padding inside conv layers

Each encoder conv block contains internal causal padding:

```python
ConstantPad2d((0, 0, kernel_t - 1, 0), 0.0)
# Pads (kernel_t - 1) zeros on the time-left side
# No padding on the time-right side
```

For `kernel_t=3` (erb_conv0, df_conv0): pads 2 zeros on the left.
For `kernel_t=1` (all other convs): no temporal padding needed.

This ensures the convolutions are **causal** — output at time t depends
only on inputs at times ≤ t. The non-causal lookahead comes solely from
`pad_feat`, not from the conv layers themselves.

### 8.4 Algorithmic delay

```
delay = (fft_size − hop_size) + lookahead × hop_size

Non-LL (conv_lookahead=2): 480 + 2×480 = 1440 samples (30ms)
LL     (conv_lookahead=0): 480 + 0     =  480 samples (10ms)
```

The `-D` flag in the process_file example compensates this by trimming the
first `delay` samples from the output, matching Tract CLI behavior.

---

## 9. SqueezedGRU_S — The Core Recurrent Unit

Source: `DeepFilterNet/df/modules.py`, class `SqueezedGRU_S`

All three GRUs in the model (encoder, ERB decoder, DF decoder) use
`SqueezedGRU_S`, which wraps PyTorch's `nn.GRU` with linear projections
and optional skip connections.

### 9.1 Architecture

```
SqueezedGRU_S(input_size=I, hidden_size=H, output_size=O)

Input [B, T, I]
  │
  ├─ linear_in:  GroupedLinearEinsum(I → H) + ReLU
  │              Compresses/expands input to GRU hidden dimension
  │
  ├─ gru:        nn.GRU(H, H, num_layers, batch_first=True)
  │              Standard PyTorch GRU with hidden state h [L, B, H]
  │
  ├─ linear_out: GroupedLinearEinsum(H → O) + ReLU    (if O specified)
  │              Projects GRU output back to desired dimension
  │
  └─ gru_skip:   Optional skip connection (identity or linear)
                 output = linear_out(gru(linear_in(x))) + gru_skip(x)
```

### 9.2 Why "Squeezed"?

The GRU hidden size (256) may differ from the input/output size (128).
The linear layers "squeeze" the dimensionality to match the GRU's
internal representation, reducing parameter count versus a full-sized GRU.

### 9.3 Hidden state dimensions per module

| Module | GRU layers | Hidden size | h shape |
|--------|-----------|-------------|---------|
| Encoder emb_gru | 1 | 256 | [1, B, 256] |
| ERB dec emb_gru | 1 (emb_num_layers−1) | 256 | [1, B, 256] |
| DF dec df_gru | 2 (df_num_layers) | 256 | [2, B, 256] |

### 9.4 ONNX representation

In the exported ONNX model, each `SqueezedGRU_S` becomes:
1. Linear (Einsum) + ReLU nodes for `linear_in`
2. Standard ONNX `GRU` operator (one per layer)
3. Linear (Einsum) + ReLU nodes for `linear_out`
4. Optional Add node for `gru_skip`

The `linear_in` and `linear_out` are **stateless** — only the `GRU` op
has hidden state. This is why our ONNX graph patching targets only the
`GRU` ops: making their `initial_h` an external input and `Y_h` an
external output is sufficient to make the entire SqueezedGRU_S stateful.

---

## 10. Streaming Inference

### 10.1 The problem

The original model processes entire utterances in batch mode:
- All T frames are available at once
- GRUs process the full sequence, building up state naturally
- `pad_feat` provides 2-frame lookahead
- Deep filtering uses both past and future frames

For real-time streaming, we must process **one frame at a time** (10ms).

### 10.2 Our solution: ONNX graph patching

We patch the `combined.onnx` model to make it streaming-capable:

1. **GRU state persistence** — Replace zero-initialized hidden states with
   external model inputs (h0, erb_h0, df_h0) and expose final states as
   outputs (h1, erb_h1, df_h1). The Rust code maintains these between frames.

2. **Time-slicing** — Insert `Slice` nodes that extract T[-1:] (last frame only)
   before entering the GRU and decoder paths. The encoder convolutions still see
   T=enc_kernel_t frames for spatial context, but GRUs process only 1 new frame.

   Without time-slicing, feeding T=3 frames to a GRU with persistent state causes
   **double-processing**: frames already seen in the previous call get re-processed
   with the wrong starting state, corrupting the hidden state trajectory.

3. **Causal processing** — No `pad_feat` lookahead. The model sees only current
   and past features. This costs ~3 dB SNR versus batch mode but enables real-time.

### 10.3 Three inference paths in Rust

```
1. Sessions::Streaming
   - Separate ONNX files: enc_conv + enc_gru + erb_dec + df_dec
   - Exported from PyTorch with explicit GRU state I/O
   - Best quality (corr=0.995 vs Python)
   - Only available for dfn3_h0

2. Sessions::Combined + StatefulH0
   - Single combined_streaming.onnx (patched by patch_onnx_streaming.py)
   - Works with ANY model variant (dfn2, dfn3, LL, non-LL)
   - Near-best quality (corr=0.990 for dfn3)
   - Universal solution

3. Sessions::Combined + StatelessWindowLast
   - Single combined.onnx with no state persistence
   - Feeds window of 40 frames each call for GRU warm-up
   - Very slow (RTF > 1x), poor quality
   - Fallback only
```

### 10.4 Per-frame processing pipeline (Rust)

```
For each 480-sample audio frame:

  1. STFT analysis → spectrum[481] complex
  2. Feature extraction:
     - feat_erb[32]:  ERB band energies (dB, normalized)
     - feat_spec[96]: unit-normalized complex spectrum
  3. Push features into rolling encoder buffer (T=enc_kernel_t frames)
  4. Run ONNX inference:
     - Input:  feat_erb[1,1,T,32], feat_spec[1,2,T,96], h0, erb_h0, df_h0
     - Output: lsnr[1,1,1], m[1,1,1,32], coefs[1,1,96,10], h1, erb_h1, df_h1
  5. Update hidden states: h0←h1, erb_h0←erb_h1, df_h0←df_h1
  6. Apply ERB mask to spectrum (all 481 bins)
  7. Apply deep filter coefficients (first 96 bins, using rolling spec buffer)
  8. ISTFT synthesis → enhanced audio[480]
```

---

## 11. Configuration Parameters

From `config.ini` and `ModelParams`:

### Signal processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| sr | 48000 | Sample rate |
| fft_size | 960 | FFT window size |
| hop_size | 480 | Frame stride |
| nb_erb | 32 | ERB bands |
| nb_df | 96 | DF frequency bins |
| df_order | 5 | DF filter taps |
| norm_tau | 1.0 | Feature normalization time constant (seconds) |
| min_nb_erb_freqs | 2 | Minimum frequency bins per ERB band |

### Model architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| conv_ch | 16 | Convolutional channels |
| conv_kernel | (1,3) | Conv kernel (time, freq) |
| conv_kernel_inp | (3,3) | Input conv kernel |
| conv_lookahead | 0 or 2 | Feature lookahead frames |
| df_lookahead | 0 or 2 | DF lookahead frames |
| emb_hidden_dim | 256 | GRU hidden dimension |
| emb_num_layers | 2 | Encoder + ERB decoder GRU layers total |
| df_hidden_dim | 256 | DF GRU hidden dimension |
| df_num_layers | 2 or 3 | DF GRU layers |
| lsnr_min | -15 | Minimum LSNR estimate (dB) |
| lsnr_max | 35 | Maximum LSNR estimate (dB) |

### Variant-specific

| Variant | conv_lookahead | df_lookahead | hidden_dim | Delay |
|---------|---------------|-------------|------------|-------|
| DFN3 | 2 | 2 | 256 | 30ms |
| DFN3-LL | 0 | 0 | 512 | 10ms |
| DFN2 | 2 | 2 | 256 | 30ms |
| DFN2-LL | 0 | 0 | 256 | 10ms |

DFN3-LL compensates for no lookahead by using a larger hidden size (512 vs 256),
which makes it ~3× slower per frame.

---

## 12. Shape Reference

### ONNX model I/O (patched streaming)

**Inputs:**
```
feat_erb:  [1, 1, T, 32]    float32  — T=enc_kernel_t (typically 3)
feat_spec: [1, 2, T, 96]    float32  — real/imag channels
h0:        [1, 1, 256]      float32  — encoder GRU state
erb_h0:    [2, 1, 256]      float32  — ERB decoder GRU state (2 layers)
df_h0:     [2, 1, 256]      float32  — DF decoder GRU state (2 layers)
```

**Outputs (time-sliced to T=1 by patch):**
```
lsnr:      [1, 1, 1]        float32  — local SNR estimate
m:         [1, 1, 1, 32]    float32  — ERB mask (sigmoid, 0-1)
coefs:     [1, 1, 96, 10]   float32  — DF coefficients (96 bins × 5 taps × 2)
h1:        [1, 1, 256]      float32  — updated encoder GRU state
erb_h1:    [2, 1, 256]      float32  — updated ERB decoder GRU state
df_h1:     [2, 1, 256]      float32  — updated DF decoder GRU state
```

### Internal tensor shapes (during forward pass)

| Tensor | Shape | Notes |
|--------|-------|-------|
| e0 | [1, 16, T, 32] | Encoder conv0 output |
| e1 | [1, 16, T, 16] | Encoder conv1 output (freq/2) |
| e2 | [1, 16, T, 8] | Encoder conv2 output (freq/4) |
| e3 | [1, 16, T, 8] | Encoder conv3 output |
| c0 | [1, 16, T, 96] | DF conv0 output |
| c1 | [1, 16, T, 48] | DF conv1 output (freq/2) |
| emb_erb | [1, T, 128] | Flattened e3 |
| emb_df | [1, T, 768] | Flattened c1 |
| cemb | [1, T, 128] | Projected emb_df |
| emb | [1, T, 128] | Combined embedding |
| emb (post-GRU) | [1, T, 128] | After encoder GRU |

---

## 13. Complete Data Flow Diagram

```
 Audio Frame [480 samples, 48kHz, 10ms]
         │
         ▼
 ┌─────────────────┐
 │  STFT Analysis   │  Vorbis window + FFT
 │  (Rust DFState)  │  fft_size=960, hop=480
 └────────┬────────┘
          │
          ▼
  Complex Spectrum [481 bins]
          │
     ┌────┴────────────────┐
     ▼                     ▼
 ┌─────────┐         ┌──────────┐
 │feat_erb │         │feat_spec │
 │ [1,32]  │         │ [2,96]   │
 │ dB norm │         │ unit norm│
 └────┬────┘         └────┬─────┘
      │                   │
      │    ┌──────────────┘
      ▼    ▼
 ┌──────────────────────────────────────────────────────┐
 │                    ENCODER                            │
 │                                                       │
 │  ERB path:    erb_conv0 → erb_conv1 → erb_conv2 →    │
 │               erb_conv3 → flatten → combine           │
 │                                                       │
 │  Spec path:   df_conv0 → df_conv1 → flatten →        │
 │               df_fc_emb → combine                     │
 │                                                       │
 │  Temporal:    emb_gru(h0) → h1                        │
 │  Output:      lsnr_fc → LSNR estimate                 │
 │                                                       │
 │  Skip outputs: e0, e1, e2, e3, c0                    │
 └──┬──────────┬─────────────┬──────────────────────────┘
    │          │             │
    │     emb [128]     c0 [16,96]
    │          │             │
    │    ┌─────┴────┐   ┌───┴──────┐
    │    ▼          ▼   ▼          │
    │ ┌────────┐  ┌──────────┐    │
    │ │ERB Dec │  │ DF Dec   │    │
    │ │        │  │          │    │
    │ │emb_gru │  │ df_gru   │    │
    │ │(erb_h0)│  │ (df_h0)  │    │
    │ │   +    │  │   +      │    │
    │ │upsample│  │df_out    │    │
    │ │e3→e2→  │  │  +       │    │
    │ │e1→e0→m │  │df_convp  │    │
    │ └───┬────┘  └────┬─────┘    │
    │     │            │          │
    │  mask [32]   coefs [96,10]  │
    │     │            │          │
    └─────┼────────────┼──────────┘
          │            │
          ▼            ▼
 ┌──────────────────────────────────┐
 │     SPECTRAL RECONSTRUCTION      │
 │                                   │
 │  1. ERB mask × spectrum (all 481 │
 │     bins, interpolated per band) │
 │                                   │
 │  2. Deep filter (bins 0-95):     │
 │     spec[t,f] = Σ coef[o,f] ×   │
 │                 spec[t-2+o,f]    │
 │     (5-tap complex FIR)          │
 │                                   │
 │  3. Merge: DF bins [0:96] +      │
 │            mask bins [96:481]     │
 └──────────────┬───────────────────┘
                │
                ▼
 ┌─────────────────┐
 │ ISTFT Synthesis  │  IFFT + Vorbis window + OLA
 │  (Rust DFState)  │
 └────────┬────────┘
          │
          ▼
 Enhanced Audio Frame [480 samples]
```

---

## References

- **DeepFilterNet paper:** [Schröter et al., 2022](https://arxiv.org/abs/2305.08227)
- **Python source:** `DeepFilterNet/df/deepfilternet3.py`
- **Rust signal processing:** `DeepFilterNet/libDF/src/lib.rs`
- **ONNX streaming patch:** `scripts/patch_onnx_streaming.py`
- **Mode selection:** `MODES.md`
