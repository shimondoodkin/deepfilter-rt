# DeepFilterNet Realtime Modes (Model Selection)

## Three inference modes

### 1. Streaming (split encoder) — best quality

Uses separate ONNX files: `enc_conv.onnx` + `enc_gru.onnx` + `erb_dec.onnx` + `df_dec.onnx`.
Exported from PyTorch with all GRU states (encoder + ERB decoder + DF decoder) as explicit I/O.
Convolutions process T=3 frames for context, GRU processes 1 frame with persistent state.

- **Select by:** having `enc_conv.onnx` + `enc_gru.onnx` in model directory
- **Example:** `models/dfn3_h0`
- **Quality:** corr=0.995, SNR=20.4 dB vs Python reference
- **Speed:** RTF ~0.13x (mean ~1.3ms per frame)

### 2. Patched streaming (combined_streaming.onnx) — universal, near-best quality

Uses a single `combined_streaming.onnx` produced by ONNX graph surgery on any `combined.onnx`.
The patch script (`scripts/patch_onnx_streaming.py`) does three things:
1. Replaces zero-initialized GRU hidden states with external model inputs (`h0`, `erb_h0`, `df_h0`)
2. Exposes final GRU hidden states as model outputs (`h1`, `erb_h1`, `df_h1`)
3. Inserts time-slice nodes (`Slice` to T[-1:]) before GRUs and decoder skip connections,
   so convolutions still see T=enc_kernel_t frames for context but GRUs only process 1 new frame

Works with any DeepFilterNet variant (DFN2, DFN3, LL, non-LL) without needing PyTorch re-export.

- **Select by:** having `combined_streaming.onnx` in model directory (auto-detected)
- **Create:** `python scripts/patch_onnx_streaming.py models/dfn3 models/dfn2_ll ...`
- **Quality:** corr=0.990, SNR=17.0 dB (dfn3), corr=0.966 (dfn2)
- **Speed:** RTF ~0.10-0.14x (mean ~1.0-1.4ms per frame)

### 3. Stateless window (fallback) — slowest

Uses a single `combined.onnx` with no GRU state persistence. Feeds a window of 40 frames
each call to give the GRU warm-up context. Very slow (RTF > 1x on some hardware).

- **Select by:** having only `combined.onnx` (no streaming files, no `h0` inputs)
- **Quality:** varies, generally worse than streaming modes
- **Speed:** RTF ~0.5-4x depending on model size (often not real-time)

## Model detection priority

The Rust code checks for model files in this order:

1. `enc_conv.onnx` + `enc_gru.onnx` + `erb_dec.onnx` + `df_dec.onnx` → **Streaming**
2. `combined_streaming.onnx` → **Patched streaming** (overrides to StatefulH0 mode)
3. `combined.onnx` with `h0` input → **Combined stateful** (H0 variant)
4. `combined.onnx` without `h0` → **Stateless window** (fallback)

## How to patch any model for streaming

```bash
# Patch one or more model directories
python scripts/patch_onnx_streaming.py models/dfn3 models/dfn2 models/dfn3_ll models/dfn2_ll

# Creates combined_streaming.onnx in each directory
# The Rust code auto-detects and uses it
```

## How the ONNX patch works

DeepFilterNet has 5 GRU layers (3 modules):
- **Encoder:** 1 GRU layer (emb_gru) — temporal embedding
- **ERB decoder:** 2 GRU layers (emb_gru) — ERB mask generation
- **DF decoder:** 2 GRU layers (df_gru) — deep filtering coefficients

The original `combined.onnx` initializes all GRU hidden states to zeros each call
(via `ConstantOfShape` or `Slice` from a zeros tensor). The patch:

1. **Replaces** zero-init nodes with external model inputs (`h0`, `erb_h0`, `df_h0`)
2. **Exposes** final hidden states as outputs (`h1`, `erb_h1`, `df_h1`)
3. **Slices time dimension** to T[-1:] before entering the GRU/decoder path

Step 3 is critical: without it, feeding T=3 frames to a GRU with persistent state
causes **double-processing** — the GRU re-processes frames it already saw in the
previous call, corrupting its hidden state. By slicing to T=1 before the GRU,
convolutions still get T=3 frames of temporal context but the GRU only processes
the newest frame. This matches how Tract's PulsedModel works internally.

## Benchmark results (34s audio, Windows, ONNX Runtime CPU)

| Model | Mode | RTF | Mean(ms) | Corr vs Py | SNR(dB) |
|-------|------|-----|----------|-----------|---------|
| dfn3_h0 | streaming (split) | 0.132x | 1.3 | 0.9955 | 20.4 |
| dfn3 | patched streaming | 0.105x | 1.0 | 0.9904 | 17.0 |
| dfn3_ll | patched streaming | 0.315x | 3.1 | 0.9540 | 10.4 |
| dfn2_h0 | combined stateful | 0.145x | 1.4 | 0.7813 | 3.0 |
| dfn2 | patched streaming | 0.130x | 1.3 | 0.9660 | -0.4 |
| dfn2_ll | patched streaming | 0.140x | 1.4 | 0.9539 | 2.9 |

All models process within the 10ms frame budget. DFN3 variants have better quality
than DFN2. The patched dfn2 actually outperforms dfn2_h0 because the patch exposes
ALL three GRU states (encoder + erb_dec + df_dec), while dfn2_h0 only has the encoder state.

## LL vs non-LL

Low-latency (LL) models have `conv_lookahead=0, df_lookahead=0` in config.ini:
- **LL:** 480 sample delay (10ms), no lookahead
- **Non-LL:** 1440 sample delay (30ms), 2-frame lookahead

Use the `-D` flag in `process_file` to compensate algorithmic delay (trim start of output).

## Recommended usage

1. **Best quality:** `models/dfn3_h0` (split encoder, all GRU states from PyTorch)
2. **Universal streaming:** Patch with `patch_onnx_streaming.py`, use any model
3. **Lowest latency:** LL variants (10ms vs 30ms delay), at some quality cost
