"""Compare streaming ONNX output vs Python reference and Tract reference."""
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
    best_corr = corr_restricted[best_idx] / (np.sqrt(np.sum(ref_f**2) * np.sum(test_f**2)))
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
    return {"corr": corr, "snr_db": snr, "mse": mse, "max_abs": np.max(np.abs(diff))}

def main():
    # Load all available outputs
    files = {}
    for name, path in [
        ("L0_D", ROOT / "out_L0_D.wav"),
        ("L1_D", ROOT / "out_L1_D.wav"),
        ("L2_D", ROOT / "out_L2_D.wav"),
        ("py_pad", ROOT / "DeepFilterNet/comparison_output/py_pad/example1_48k.wav"),
        ("py_nopad", ROOT / "DeepFilterNet/comparison_output/py_nopad/example1_48k.wav"),
        ("tract_D", ROOT / "DeepFilterNet/comparison_output/rust_D/example1_48k.wav"),
    ]:
        if path.exists():
            audio, sr = load_wav(path)
            files[name] = audio
            print(f"  {name:20s}: {len(audio)} samples")
        else:
            print(f"  {name:20s}: NOT FOUND ({path})")

    if "py_pad" not in files:
        print("ERROR: Python reference not found. Run compare_python_vs_rust.py first.")
        return

    ref = files["py_pad"]
    print(f"\n{'='*72}")
    print("Cross-correlation vs py_pad (Python pad=True reference)")
    print(f"{'='*72}")
    for name, audio in files.items():
        if name == "py_pad":
            continue
        lag, xcorr = find_delay(ref, audio)
        print(f"\n  {name}:")
        print(f"    lag={lag} samples ({lag/48000*1000:.1f}ms), xcorr={xcorr:.4f}, len={len(audio)}")
        # Also compute metrics at the aligned position
        if lag > 0:
            m = compute_metrics(ref[lag:], audio)
        elif lag < 0:
            m = compute_metrics(ref, audio[-lag:])
        else:
            m = compute_metrics(ref, audio)
        print(f"    aligned: corr={m['corr']:.6f}, SNR={m['snr_db']:.1f}dB, max_err={m['max_abs']:.4f}")

    # Direct comparison of L values vs tract_D
    if "tract_D" in files:
        print(f"\n{'='*72}")
        print("Direct comparison vs tract_D (Tract batch reference)")
        print(f"{'='*72}")
        for name in ["L0_D", "L1_D", "L2_D"]:
            if name in files:
                lag, xcorr = find_delay(files["tract_D"], files[name])
                m = compute_metrics(files["tract_D"], files[name])
                print(f"\n  {name} vs tract_D:")
                print(f"    lag={lag}, xcorr={xcorr:.4f}")
                print(f"    corr={m['corr']:.6f}, SNR={m['snr_db']:.1f}dB, max_err={m['max_abs']:.4f}")

if __name__ == "__main__":
    main()
