# DeepFilterNet Realtime Modes (Model Selection)

This file documents the **working runtime options** and how they are selected based on the
model directory you pass to the Rust API.

## Modes that work

1) **Stateful H0 mode**
   - Uses ONNX encoder with GRU state inputs/outputs (`h0` / `h1`).
   - Best quality; closest to Python.
   - **Select by model folder suffix:** `_h0`
   - Example: `models\dfn3_h0`

2) **Stateless window mode (growing window, take last)**
   - No GRU state. Uses a growing temporal window and takes the last frame.
   - Works for regular ONNX models without `h0`.
   - **Select by model folder without `_h0` suffix**
   - Example: `models\dfn3`, `models\dfn3_ll`

## How selection works (Rust)

- If the model folder name ends with `_h0`, Rust uses **Stateful H0 mode**.
- Otherwise Rust uses **Stateless window mode**.
- If a folder ends with `_h0` but the encoder has no `h0` input, Rust fails fast.

## LL vs non?LL

Low?latency (LL) models are selected by their config (`df_lookahead = 0`).
They have lower algorithmic delay, but the runtime mode still follows the rules above:

- `dfn3_ll_h0` -> stateful (best quality, low latency)
- `dfn3_ll` -> stateless window (works, but not as good as h0)

## Recommended usage

- For best quality: use `*_h0` models (stateful ONNX exports)
- For fallback: use regular models (stateless window)

