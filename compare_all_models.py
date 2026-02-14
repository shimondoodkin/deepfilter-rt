"""Compare all deepfilter-rt model outputs against Python reference.

Usage: python compare_all_models.py [--ref REF_WAV]

Loads outputs from comparison_output/ and compares against Python pad=True reference.
"""
import sys
import numpy as np
from pathlib import Path
from scipy.signal import correlate

ROOT = Path(__file__).parent

def load_wav(path):
    import struct, wave
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    return samples, sr

def find_delay(ref, test, max_lag=5000):
    ref_f = ref.flatten()[:200000]
    test_f = test.flatten()[:200000]
    ref_f = ref_f - np.mean(ref_f)
    test_f = test_f - np.mean(test_f)
    corr = correlate(ref_f, test_f, mode='full')
    lags = np.arange(-len(test_f) + 1, len(ref_f))
    mask = np.abs(lags) <= max_lag
    corr_restricted = corr[mask]
    lags_restricted = lags[mask]
    best_idx = np.argmax(np.abs(corr_restricted))
    best_lag = lags_restricted[best_idx]
    best_corr = corr_restricted[best_idx] / (np.sqrt(np.sum(ref_f**2) * np.sum(test_f**2)) + 1e-30)
    return best_lag, best_corr

def compute_metrics(ref, test):
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    diff = ref - test
    mse = np.mean(diff ** 2)
    rms_ref = np.sqrt(np.mean(ref ** 2))
    rms_diff = np.sqrt(mse)
    corr = np.corrcoef(ref, test)[0, 1] if rms_ref > 0 else 0.0
    snr = 20 * np.log10(rms_ref / rms_diff) if rms_diff > 0 else float('inf')
    return {"corr": corr, "snr_db": snr, "mse": mse, "max_abs": np.max(np.abs(diff)),
            "rms_ref": rms_ref, "rms_test": np.sqrt(np.mean(test ** 2))}

def load_latency_csv(path):
    """Load per-frame latency CSV and return statistics."""
    try:
        data = np.loadtxt(str(path), delimiter=',', skiprows=1)
        if len(data) == 0:
            return None
        us = data[:, 1]
        return {
            "mean_ms": np.mean(us) / 1000.0,
            "median_ms": np.median(us) / 1000.0,
            "p95_ms": np.percentile(us, 95) / 1000.0,
            "p99_ms": np.percentile(us, 99) / 1000.0,
            "max_ms": np.max(us) / 1000.0,
            "frames": len(us),
        }
    except Exception:
        return None

def main():
    print("=" * 80)
    print("DeepFilter-RT: All Model Comparison")
    print("=" * 80)

    # Model definitions: (name, wav_path, description)
    # Prefer *_streaming.wav (patched GRU state), fall back to *.wav
    models = [
        ("dfn3_h0",      "comparison_output/dfn3_h0.wav",           "DFN3 H0 (split enc)"),
        ("dfn3_stream",  "comparison_output/dfn3_streaming.wav",    "DFN3 (patched streaming)"),
        ("dfn3_ll",      "comparison_output/dfn3_ll_streaming.wav", "DFN3-LL (patched streaming)"),
        ("dfn2_h0",      "comparison_output/dfn2_h0.wav",           "DFN2 H0 (enc-only state)"),
        ("dfn2_stream",  "comparison_output/dfn2_streaming.wav",    "DFN2 (patched streaming)"),
        ("dfn2_ll",      "comparison_output/dfn2_ll_streaming.wav", "DFN2-LL (patched streaming)"),
    ]

    # Load reference
    ref_path = ROOT / "DeepFilterNet" / "comparison_output" / "py_pad" / "example1_48k.wav"
    if not ref_path.exists():
        # Try alternative reference paths
        for alt in [ROOT / "comparison_output" / "py_pad.wav",
                    ROOT / "py_pad.wav"]:
            if alt.exists():
                ref_path = alt
                break
        else:
            print(f"\nWARNING: Python reference not found at {ref_path}")
            print("Will compare models against each other only.\n")
            ref_path = None

    ref = None
    if ref_path:
        ref, sr = load_wav(ref_path)
        print(f"\nReference: {ref_path.name} ({len(ref)} samples, {len(ref)/48000:.2f}s)")

    # Also load Tract reference if available
    tract_path = ROOT / "DeepFilterNet" / "comparison_output" / "rust_D" / "example1_48k.wav"
    tract = None
    if tract_path.exists():
        tract, _ = load_wav(tract_path)
        print(f"Tract ref:  rust_D ({len(tract)} samples)")

    # Load input for reference
    input_path = ROOT / "example1_48k.wav"
    inp = None
    if input_path.exists():
        inp, _ = load_wav(input_path)
        print(f"Input:      {input_path.name} ({len(inp)} samples)")

    # Load all model outputs
    print(f"\n{'─' * 80}")
    print("Loading model outputs...")
    print(f"{'─' * 80}")

    loaded = {}
    latencies = {}
    for name, wav_rel, desc in models:
        wav_path = ROOT / wav_rel
        csv_path = ROOT / f"{wav_rel}.csv"
        if wav_path.exists():
            audio, sr = load_wav(wav_path)
            loaded[name] = audio
            lat = load_latency_csv(csv_path)
            if lat:
                latencies[name] = lat
            print(f"  {name:12s} [{desc:30s}]: {len(audio):>8d} samples  ", end="")
            if lat:
                print(f"mean={lat['mean_ms']:.2f}ms  p95={lat['p95_ms']:.2f}ms")
            else:
                print()
        else:
            print(f"  {name:12s} [{desc:30s}]: NOT FOUND")

    if not loaded:
        print("\nNo model outputs found! Run process_file first.")
        return

    # Comparison vs Python reference
    if ref is not None:
        print(f"\n{'=' * 80}")
        print("Comparison vs Python Reference (pad=True)")
        print(f"{'=' * 80}")
        print(f"\n  {'Model':<12s} {'Lag':>6s} {'Lag(ms)':>8s} {'XCorr':>7s} {'Corr':>10s} {'SNR(dB)':>8s} {'MaxErr':>8s} {'Samples':>8s}")
        print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*7} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

        for name, audio in loaded.items():
            lag, xcorr = find_delay(ref, audio)
            # Align based on lag
            if lag > 0:
                m = compute_metrics(ref[lag:], audio)
            elif lag < 0:
                m = compute_metrics(ref, audio[-lag:])
            else:
                m = compute_metrics(ref, audio)
            print(f"  {name:<12s} {lag:>6d} {lag/48.0:>8.1f} {xcorr:>7.4f} {m['corr']:>10.6f} {m['snr_db']:>8.1f} {m['max_abs']:>8.4f} {len(audio):>8d}")

    # Comparison vs Tract reference
    if tract is not None:
        print(f"\n{'=' * 80}")
        print("Comparison vs Tract Reference (batch -D)")
        print(f"{'=' * 80}")
        print(f"\n  {'Model':<12s} {'Lag':>6s} {'Lag(ms)':>8s} {'XCorr':>7s} {'Corr':>10s} {'SNR(dB)':>8s} {'MaxErr':>8s}")
        print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*7} {'─'*10} {'─'*8} {'─'*8}")

        for name, audio in loaded.items():
            lag, xcorr = find_delay(tract, audio)
            if lag > 0:
                m = compute_metrics(tract[lag:], audio)
            elif lag < 0:
                m = compute_metrics(tract, audio[-lag:])
            else:
                m = compute_metrics(tract, audio)
            print(f"  {name:<12s} {lag:>6d} {lag/48.0:>8.1f} {xcorr:>7.4f} {m['corr']:>10.6f} {m['snr_db']:>8.1f} {m['max_abs']:>8.4f}")

    # Latency summary
    if latencies:
        print(f"\n{'=' * 80}")
        print("Per-Frame Latency (inference only)")
        print(f"{'=' * 80}")
        print(f"\n  {'Model':<12s} {'Mean(ms)':>9s} {'Median':>9s} {'P95':>9s} {'P99':>9s} {'Max':>9s} {'Frames':>8s} {'RTF':>7s}")
        print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*7}")

        hop_ms = 10.0  # 480 samples at 48kHz
        for name, lat in latencies.items():
            rtf = lat['mean_ms'] / hop_ms
            print(f"  {name:<12s} {lat['mean_ms']:>9.3f} {lat['median_ms']:>9.3f} {lat['p95_ms']:>9.3f} {lat['p99_ms']:>9.3f} {lat['max_ms']:>9.3f} {lat['frames']:>8d} {rtf:>7.3f}")

    # Cross-model comparison matrix
    names = list(loaded.keys())
    if len(names) > 1:
        print(f"\n{'=' * 80}")
        print("Cross-Model Correlation Matrix")
        print(f"{'=' * 80}")

        # Header
        header = f"  {'':12s}"
        for n in names:
            header += f" {n[:8]:>8s}"
        print(header)
        print(f"  {'─'*12}" + "─" * (9 * len(names)))

        for i, n1 in enumerate(names):
            row = f"  {n1:<12s}"
            for j, n2 in enumerate(names):
                if i == j:
                    row += f" {'1.0000':>8s}"
                elif j > i:
                    m = compute_metrics(loaded[n1], loaded[n2])
                    row += f" {m['corr']:>8.4f}"
                else:
                    row += f" {'':>8s}"
            print(row)

    # SNR vs input (noise reduction effectiveness)
    if inp is not None:
        print(f"\n{'=' * 80}")
        print("Noise Reduction (vs noisy input)")
        print(f"{'=' * 80}")
        print(f"\n  {'Model':<12s} {'RMS_in':>8s} {'RMS_out':>8s} {'Corr_in':>9s}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*9}")

        rms_in = np.sqrt(np.mean(inp[:min(len(inp), 1632000)]**2))
        for name, audio in loaded.items():
            min_len = min(len(inp), len(audio))
            m = compute_metrics(inp[:min_len], audio[:min_len])
            print(f"  {name:<12s} {rms_in:>8.4f} {m['rms_test']:>8.4f} {m['corr']:>9.4f}")

    print(f"\n{'=' * 80}")
    print("Done. Output WAVs in: comparison_output/")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
