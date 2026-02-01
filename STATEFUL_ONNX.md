# DeepFilterNet ONNX Stateful Export (Rust realtime)

This document explains how to build **stateful** ONNX models for DeepFilterNet so the
Rust realtime pipeline matches Python (GRU hidden state preserved across frames).

If you are new to DeepFilterNet or ONNX, follow the **Step?by?step** section.

## Why this is needed

The standard ONNX encoder export for DeepFilterNet3 does **not** include GRU state inputs/outputs.
Rust realtime inference runs the encoder per small chunk, so without GRU state the encoder
outputs (and therefore masks/DF coefficients) diverge from Python. This causes artifacts
(choppy or ?mask?like? sound).

The fix is to export a **stateful** encoder that has:

- inputs: `feat_erb`, `feat_spec`, `h0`
- outputs: `e0, e1, e2, e3, emb, c0, lsnr, h1`

Rust keeps `h1` and feeds it back as `h0` for the next frame.

## Quick start (step?by?step)

1) **Check you are in the right repo**

```
cd C:\Users\user\Documents\projects\aiphone\deepfilter_rt
```

2) **Pick a target model directory**

Example targets:
- `C:\Users\user\Documents\projects\aiphone\deepfilter_rt\models\dfn3_h0`
- `C:\Users\user\Documents\projects\aiphone\deepfilter_rt\models\dfn2_h0`

3) **Edit the export script**

Open:
`C:\Users\user\Documents\projects\aiphone\beam_stt\export_onnx_stateful.py`

Find:
`export_dir = r"..."`

Set it to your target model directory from step 2.

4) **Run the export**

```
C:\Users\user\Documents\projects\aiphone\beam_stt\.venv310\Scripts\python.exe C:\Users\user\Documents\projects\aiphone\beam_stt\export_onnx_stateful.py
```

This overwrites in the target model folder:
- `enc.onnx`
- `erb_dec.onnx`
- `df_dec.onnx`
- `config.ini` (copied from DeepFilterNet cache if found)

5) **Test in Rust**

```
cargo run --example process_file -- input.wav output.wav models\dfn3_h0
```

You should hear clean output (not choppy). If artifacts return, re?export stateful ONNX.

## What this changes in Rust

The Rust pipeline automatically detects if `enc.onnx` has `h0`:

- If `h0` exists, it runs **stateful** inference and keeps GRU state across frames.
- If not, it runs stateless inference (which **will not** match Python).

Key changes are in `deepfilter_rt/src/lib.rs`:

- Added `enc_h` buffer for GRU hidden state
- Added `enc_stateful` flag (checks for `h0` input on encoder session)
- On reset, clears `enc_h`
- On inference:
  - pass `h0` tensor into encoder
  - read `h1` and store into `enc_h`

## Current state in this repo

The following model folders were duplicated with `_h0` suffix:

- `models\dfn2_h0` (stateful export applied)
- `models\dfn3_h0` (stateful export applied)
- `models\dfn2_ll_h0` (copied, but **not** stateful export; see note)
- `models\dfn3_ll_h0` (copied, but **not** stateful export; see note)

**Important note about LL models**

The official Python package only ships checkpoints for:

- `DeepFilterNet2`
- `DeepFilterNet3`

It does **not** include `DeepFilterNet2_ll` or `DeepFilterNet3_ll` checkpoints,
so we cannot export stateful ONNX for LL models using the Python package alone.
If you obtain LL checkpoints, rerun the export and overwrite the `_ll_h0` folders.

## Common questions

**Q: How do I know if the model is stateful?**
A: If the encoder has an input named `h0`, it is stateful. Rust checks this automatically.

**Q: I exported but still get choppy audio.**
A: Re?export from the exact checkpoint you want to match. If you changed model families
(e.g. dfn2 -> dfn3), you must export again. Make sure you are using the new model folder
in Rust.

**Q: Where do the checkpoints come from?**
A: DeepFilterNet caches them here (example for DFN3):
`C:\Users\user\AppData\Local\DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3\checkpoints\model_120.ckpt.best`

**Q: Do I need to edit Rust for each model?**
A: No. Once a model is exported as stateful ONNX, Rust will detect `h0` and work.

## Notes / tips

- Always export from the same DeepFilterNet checkpoint you want to match.
- For DeepFilterNet3, the cached checkpoint is typically:
  `C:\Users\user\AppData\Local\DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3\checkpoints\model_120.ckpt.best`
- If you switch model families or checkpoints, you must re?export the ONNX.
- The export script uses CPU and should run in under a minute.

