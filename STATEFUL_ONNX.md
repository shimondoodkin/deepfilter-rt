# DeepFilterNet ONNX Stateful Export

This document explains how to export **stateful** (GRU) ONNX models for DeepFilterNet
so the Rust pipeline preserves hidden state across frames.

## Why this is needed

The standard ONNX encoder export does **not** include GRU state inputs/outputs.
Rust realtime inference runs the encoder per small chunk, so without GRU state the encoder
outputs diverge from Python. This causes artifacts (choppy or mask-like sound).

The fix is to export a **stateful** encoder with:

- inputs: `feat_erb`, `feat_spec`, `h0`
- outputs: `e0, e1, e2, e3, emb, c0, lsnr, h1`

Rust keeps `h1` and feeds it back as `h0` for the next frame.

## Quick start

### 1. Install Python dependencies

```bash
pip install torch deepfilternet onnx
```

### 2. Export stateful ONNX models

The export script is included at `scripts/export_onnx_stateful.py`:

```bash
# Export to a model directory (e.g. dfn3_h0)
python scripts/export_onnx_stateful.py models/dfn3_h0

# Optionally provide a 48kHz WAV for tracing (uses synthetic audio if omitted)
python scripts/export_onnx_stateful.py models/dfn3_h0 audio_48k.wav
```

This produces in the target directory:
- `enc.onnx` (stateful encoder with h0/h1)
- `erb_dec.onnx`
- `df_dec.onnx`
- `config.ini` (copied from DeepFilterNet cache if found)

### 3. Merge into combined model

```bash
pip install -r scripts/requirements.txt
python scripts/merge_onnx.py models/dfn3_h0
```

### 4. Test in Rust

```bash
cargo run --example process_file -- input.wav output.wav models/dfn3_h0
```

You should hear clean output (not choppy).

## How it works in Rust

The Rust pipeline automatically detects if `combined.onnx` has an `h0` input:

- If `h0` exists: runs **stateful** inference, preserves GRU state across frames
- If not: runs stateless inference (windowed encoder context)

Key behavior in `lib.rs`:
- `enc_h` buffer stores GRU hidden state
- On inference: passes `h0` tensor, reads `h1` output, stores it for next frame
- On reset: clears `enc_h` to zeros

## Model variants

| Folder | Stateful | Notes |
|--------|----------|-------|
| `dfn2` | No | Stateless DeepFilterNet2 |
| `dfn2_ll` | No | Low-latency DeepFilterNet2 |
| `dfn2_h0` | **Yes** | Stateful DeepFilterNet2 (GRU) |
| `dfn3` | No | Stateless DeepFilterNet3 |
| `dfn3_ll` | No | Low-latency DeepFilterNet3 |
| `dfn3_h0` | **Yes** | Stateful DeepFilterNet3 (GRU) |

## Common questions

**Q: How do I know if the model is stateful?**
A: If the combined model has an input named `h0`, it is stateful. Rust checks this automatically.

**Q: I exported but still get choppy audio.**
A: Re-export from the exact checkpoint you want to match. If you changed model families
(e.g. dfn2 -> dfn3), you must export again. Make sure you are using the correct model folder.

**Q: Where do the checkpoints come from?**
A: DeepFilterNet caches them automatically. Example path for DFN3 on Windows:
`%LOCALAPPDATA%\DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3\checkpoints\model_120.ckpt.best`

**Q: Can I export LL models as stateful?**
A: The official DeepFilterNet package only ships checkpoints for standard (non-LL) models.
LL models don't use GRU, so stateful export doesn't apply to them.

**Q: Do I need to edit Rust code for each model?**
A: No. Once exported as stateful ONNX and merged, Rust detects `h0` and works automatically.

## Notes

- Always export from the same DeepFilterNet checkpoint you want to match.
- The export script uses CPU and runs in under a minute.
- After exporting, always run `merge_onnx.py` to create `combined.onnx`.
