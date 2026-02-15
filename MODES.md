# DeepFilterNet Realtime Modes (Model Selection)

## Three inference modes

### 1. Streaming (split encoder) — best quality

Uses separate ONNX files: `enc_conv_streaming.onnx` + `enc_gru_streaming.onnx` + `erb_dec_streaming.onnx` + `df_dec_streaming.onnx`.
All GRU states (encoder + ERB decoder + DF decoder) are exposed as explicit I/O.
Convolutions process T=3 frames for context, GRU processes 1 frame with persistent state.

These split files can be produced in two equivalent ways:
1. **PyTorch export** (`scripts/export_onnx_stateful.py`) — requires PyTorch + DFN3 weights
2. **ONNX surgery** (`scripts/split_encoder_and_patch_decoders.py`) — works on any existing `enc.onnx` + decoders, no PyTorch needed

Both produce identical output (correlation=0.999991, SNR=47.7 dB vs Tract reference).

- **Select by:** having `enc_conv_streaming.onnx` + `enc_gru_streaming.onnx` in model directory
- **Example:** `models/dfn3_h0`, `models/dfn3`, `models/dfn3_ll`
- **Quality:** corr=0.999991, SNR=47.7 dB vs Tract reference (DFN3)
- **Speed:** RTF ~0.13x (mean ~1.3ms per frame)

### 2. Combined streaming (combined_streaming.onnx) — same quality, single file

Uses a single `combined_streaming.onnx` with GRU states exposed as I/O. Can be produced by:
1. **`merge_split_models.py`** — merges 4 split models back into one (recommended)
2. **`patch_onnx_streaming.py`** — patches combined.onnx directly (also works)

Both approaches achieve the same quality as split streaming (47.6 dB SNR for DFN3 vs Tract).
The combined model is faster than split (RTF ~0.11x vs ~0.33x) since ORT optimizes the single
graph more effectively, but uses minimal conv temporal context (enc_kernel_t frames).

- **Select by:** having `combined_streaming.onnx` in model directory (auto-detected)
- **Create:** `python scripts/merge_split_models.py models/dfn3` (from split models)
- **Quality:** corr=0.999991, SNR=47.6 dB (dfn3 vs Tract)
- **Speed:** RTF ~0.10-0.14x (mean ~1.0-1.4ms per frame)

### 3. Stateless window (fallback) — slowest

Uses a single `combined.onnx` with no GRU state persistence. Feeds a window of 40 frames
each call to give the GRU warm-up context. Very slow (RTF > 1x on some hardware).

- **Select by:** having only `combined.onnx` (no streaming files, no `h0` inputs)
- **Quality:** varies, generally worse than streaming modes
- **Speed:** RTF ~0.5-4x depending on model size (often not real-time)

## Model detection priority

The Rust code checks for model files in this order:

1. `enc_conv_streaming.onnx` + `enc_gru_streaming.onnx` + `erb_dec_streaming.onnx` + `df_dec_streaming.onnx` → **Streaming**
2. `combined_streaming.onnx` → **Patched streaming** (overrides to StatefulH0 mode)
3. `combined.onnx` with `h0` input → **Combined stateful** (H0 variant)
4. `combined.onnx` without `h0` → **Stateless window** (fallback)

## How to create split streaming models (recommended)

The `split_encoder_and_patch_decoders.py` script converts any standard DeepFilterNet export
into the split-encoder format. It works by:

1. **Splitting `enc.onnx`** into `enc_conv_streaming.onnx` (convolutions only) + `enc_gru_streaming.onnx` (GRU + post-GRU)
2. **Patching decoders** to `*_streaming.onnx` files with GRU hidden states as inputs/outputs (originals preserved)
3. **Time-slicing** encoder conv outputs to T[-1:] so downstream modules process 1 frame

```bash
# Split one or more model directories
python scripts/split_encoder_and_patch_decoders.py models/dfn3 models/dfn3_ll models/dfn2 models/dfn2_ll

# Each directory gets: enc_conv_streaming.onnx, enc_gru_streaming.onnx, erb_dec_streaming.onnx, df_dec_streaming.onnx
# The Rust code auto-detects and uses them (highest priority)
```

This approach is preferred over `patch_onnx_streaming.py` because:
- It achieves near-perfect match to Tract reference output (47.7 dB SNR vs 17.0 dB)
- The encoder split cleanly separates convolutions from GRU, avoiding the approximations
  inherent in patching a merged combined.onnx
- Works with all model variants (DFN2, DFN3, LL, non-LL)

## How to merge split models into combined_streaming.onnx

If you need a single ONNX file (e.g. for mobile deployment) but want the quality of
split models, use `merge_split_models.py`:

```bash
# Merge split models back into a single combined_streaming.onnx
python scripts/merge_split_models.py models/dfn3

# The merged model preserves all GRU state I/O and quality
```

## How to patch any model for streaming (legacy)

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

## Benchmark results (34s audio, Windows, ONNX Runtime CPU, vs Tract reference)

| Model | Mode | Corr vs Tract | SNR (dB) | RTF |
|-------|------|---------------|----------|-----|
| dfn3 | split streaming | 0.999991 | 47.6 | 0.34x |
| dfn3 | combined streaming | 0.999991 | 47.6 | 0.11x |
| dfn3_ll | split streaming | 0.999605 | 31.0 | 0.82x |
| dfn3_h0 | split streaming | 0.999991 | 47.6 | 0.34x |

Split streaming uses more temporal context (T=40) giving slightly higher quality for LL models.
Combined streaming is faster (single ORT session) but uses T=enc_kernel_t.
Both achieve near-perfect match to Tract reference for non-LL models.

## LL vs non-LL

Low-latency (LL) models have `conv_lookahead=0, df_lookahead=0` in config.ini:
- **LL:** 480 sample delay (10ms), no lookahead
- **Non-LL:** 1440 sample delay (30ms), 2-frame lookahead

Use the `-D` flag in `process_file` to compensate algorithmic delay (trim start of output).

## Recommended usage

1. **Best quality + speed:** `combined_streaming.onnx` from `merge_split_models.py` (single file, fast)
2. **Best quality for LL:** Split streaming (T=40 gives extra conv context for LL models)
3. **From scratch:** Run `split_encoder_and_patch_decoders.py` then `merge_split_models.py`
4. **Lowest latency:** LL variants (10ms vs 30ms delay), at some quality cost
