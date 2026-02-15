"""Compare two WAV files: compute SNR, correlation, and alignment.

Auto-detects delay between signals using cross-correlation and reports
quality metrics both raw and after alignment.

Usage:
    python scripts/compare_wav.py reference.wav test.wav
    python scripts/compare_wav.py reference.wav test1.wav test2.wav test3.wav
    python scripts/compare_wav.py --no-align reference.wav test.wav

Requirements:
    pip install numpy scipy soundfile
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate


def load_wav(path):
    """Load WAV file as float64 mono. Returns (samples, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mix to mono
    return data, sr


def find_delay(ref, test, max_lag=5000):
    """Find sample delay between two signals using cross-correlation.

    Returns (lag, normalized_correlation).
    Positive lag means test is delayed relative to ref.
    """
    # Use first ~4 seconds for speed
    n = min(200000, len(ref), len(test))
    r = ref[:n] - np.mean(ref[:n])
    t = test[:n] - np.mean(test[:n])

    corr = correlate(r, t, mode="full")
    lags = np.arange(-len(t) + 1, len(r))

    mask = np.abs(lags) <= max_lag
    corr_r = corr[mask]
    lags_r = lags[mask]

    best_idx = np.argmax(np.abs(corr_r))
    best_lag = lags_r[best_idx]

    norm = np.sqrt(np.sum(r ** 2) * np.sum(t ** 2))
    best_corr = corr_r[best_idx] / norm if norm > 0 else 0.0

    return int(best_lag), float(best_corr)


def align(ref, test, lag):
    """Align two signals given a lag value. Returns (ref_aligned, test_aligned)."""
    if lag > 0:
        return ref[lag:], test
    elif lag < 0:
        return ref, test[-lag:]
    return ref, test


def compute_metrics(ref, test):
    """Compute quality metrics between two aligned signals."""
    n = min(len(ref), len(test))
    r = ref[:n]
    t = test[:n]
    diff = r - t

    rms_ref = np.sqrt(np.mean(r ** 2))
    rms_test = np.sqrt(np.mean(t ** 2))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    corr = np.corrcoef(r, t)[0, 1] if rms_ref > 0 and rms_test > 0 else 0.0
    snr = 20 * np.log10(rms_ref / rms_diff) if rms_diff > 0 else float("inf")

    return {
        "samples": n,
        "corr": corr,
        "snr_db": snr,
        "mse": float(np.mean(diff ** 2)),
        "max_abs": float(np.max(np.abs(diff))),
        "rms_ref": rms_ref,
        "rms_test": rms_test,
    }


def print_metrics(label, m):
    print(f"  {label}:")
    print(f"    Correlation:   {m['corr']:.6f}")
    print(f"    SNR (dB):      {m['snr_db']:.1f}")
    print(f"    MSE:           {m['mse']:.2e}")
    print(f"    Max abs error: {m['max_abs']:.6f}")
    print(f"    RMS ref/test:  {m['rms_ref']:.5f} / {m['rms_test']:.5f}")
    print(f"    Samples:       {m['samples']}")


def compare(ref_path, test_path, do_align=True):
    """Compare two WAV files and print metrics."""
    ref, ref_sr = load_wav(ref_path)
    test, test_sr = load_wav(test_path)

    if ref_sr != test_sr:
        print(f"  WARNING: sample rate mismatch ({ref_sr} vs {test_sr})")

    ref_dur = len(ref) / ref_sr
    test_dur = len(test) / test_sr

    print(f"\n  {Path(ref_path).name} ({len(ref)} samples, {ref_dur:.2f}s)")
    print(f"  {Path(test_path).name} ({len(test)} samples, {test_dur:.2f}s)")

    if do_align:
        lag, xcorr = find_delay(ref, test)
        print(f"  Detected lag: {lag} samples ({lag / ref_sr * 1000:.2f}ms), xcorr peak: {xcorr:.4f}")

        if lag != 0:
            # Show raw (unaligned) metrics
            m_raw = compute_metrics(ref, test)
            print_metrics("Raw (no alignment)", m_raw)

            # Show aligned metrics
            ref_a, test_a = align(ref, test, lag)
            m_aligned = compute_metrics(ref_a, test_a)
            print_metrics(f"Aligned (shift {lag})", m_aligned)
        else:
            m = compute_metrics(ref, test)
            print_metrics("Aligned (no shift needed)", m)
    else:
        m = compute_metrics(ref, test)
        print_metrics("Direct comparison", m)


def main():
    parser = argparse.ArgumentParser(
        description="Compare WAV files: compute SNR, correlation, and alignment."
    )
    parser.add_argument("reference", help="Reference WAV file")
    parser.add_argument("test", nargs="+", help="One or more test WAV files to compare against reference")
    parser.add_argument("--no-align", action="store_true", help="Skip auto-alignment (assume already aligned)")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Reference file not found: {ref_path}")
        sys.exit(1)

    print(f"Reference: {ref_path}")

    for test in args.test:
        test_path = Path(test)
        if not test_path.exists():
            print(f"\n  Test file not found: {test_path}")
            continue
        compare(ref_path, test_path, do_align=not args.no_align)

    print()


if __name__ == "__main__":
    main()
