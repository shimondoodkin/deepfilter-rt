//! Example: Pipelined real-time denoising with separate producer/consumer threads
//!
//! Demonstrates how to run DeepFilter inference on a dedicated thread, decoupled
//! from the audio I/O thread via a bounded channel. This is the recommended pattern
//! for real-time voice applications where ONNX inference is too heavy for the
//! audio callback thread.
//!
//! Architecture:
//!   Audio thread  ──(channel)──►  DeepFilter thread
//!                                   (ONNX inference)
//!
//! Usage: cargo run --example pipelined -- input.wav output.wav [model_dir]

use deepfilter_rt::{DeepFilterProcessor, HOP_SIZE, SAMPLE_RATE};
use std::path::Path;
use std::sync::mpsc;

/// A job sent from the producer (audio) thread to the DeepFilter consumer thread.
struct DenoiseJob {
    /// Sequential frame index (for ordering / diagnostics).
    frame_idx: u64,
    /// 480 samples of audio to denoise.
    audio: Vec<f32>,
}

/// Result sent back from DeepFilter thread (optional — you could also write to a
/// shared ring buffer or timeline instead).
struct DenoiseResult {
    frame_idx: u64,
    denoised: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.wav> <output.wav> [model_dir]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let model_dir = if args.len() > 3 {
        Path::new(&args[3]).to_path_buf()
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/dfn3_ll")
    };

    // ── Load input audio ──────────────────────────────────────────────
    let mut reader = hound::WavReader::open(input_path)?;
    let spec = reader.spec();
    println!(
        "Input: {} Hz, {} ch, {:?}",
        spec.sample_rate, spec.channels, spec.sample_format
    );
    if spec.sample_rate != SAMPLE_RATE as u32 {
        eprintln!(
            "Warning: sample rate {} != expected {}. Resample first!",
            spec.sample_rate, SAMPLE_RATE
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    };
    let mono: Vec<f32> = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        samples
    };

    let total_frames = mono.len() / HOP_SIZE;
    println!(
        "Processing {} frames ({:.2}s) with pipelined threads...",
        total_frames,
        mono.len() as f32 / SAMPLE_RATE as f32
    );

    // ── Channels ──────────────────────────────────────────────────────
    // Bounded channel prevents unbounded memory growth if producer is faster.
    // 500 slots ≈ 5 seconds of buffering at 10ms frames.
    let (job_tx, job_rx) = mpsc::sync_channel::<DenoiseJob>(500);
    let (result_tx, result_rx) = mpsc::sync_channel::<DenoiseResult>(500);

    // ── DeepFilter consumer thread ────────────────────────────────────
    let df_thread = std::thread::spawn(move || {
        let mut proc = DeepFilterProcessor::new(&model_dir).expect("load DeepFilter model");
        proc.warmup().expect("DeepFilter warmup");
        eprintln!("DeepFilter ready: {}", proc.variant().name());

        let mut denoised = vec![0.0f32; HOP_SIZE];

        while let Ok(job) = job_rx.recv() {
            denoised.fill(0.0);
            proc.process_frame(&job.audio, &mut denoised)
                .expect("denoise frame");

            let _ = result_tx.send(DenoiseResult {
                frame_idx: job.frame_idx,
                denoised: denoised.clone(),
            });
        }
    });

    // ── Producer: simulate audio I/O thread ───────────────────────────
    let start = std::time::Instant::now();
    for (i, chunk) in mono.chunks(HOP_SIZE).enumerate() {
        if chunk.len() < HOP_SIZE {
            break; // skip incomplete final frame
        }
        // In a real app, this is where you'd run any fast pre-processing
        // (e.g. gain, filtering) before sending to the DeepFilter thread:
        //
        //   dsp.process(&raw_mic, &mut processed);
        //   job_tx.send(DenoiseJob { audio: processed, ... });
        //
        let job = DenoiseJob {
            frame_idx: i as u64,
            audio: chunk.to_vec(),
        };
        job_tx.send(job)?;
    }
    drop(job_tx); // signal consumer to finish

    // ── Collect results ───────────────────────────────────────────────
    let mut out_samples: Vec<f32> = Vec::with_capacity(mono.len());
    while let Ok(result) = result_rx.recv() {
        out_samples.extend_from_slice(&result.denoised);
    }

    df_thread.join().unwrap();

    let elapsed = start.elapsed();
    let rtf = elapsed.as_secs_f32() / (mono.len() as f32 / SAMPLE_RATE as f32);
    println!(
        "Done in {:.2}s (RTF: {:.3}x realtime)",
        elapsed.as_secs_f32(),
        rtf
    );

    // ── Write output ──────────────────────────────────────────────────
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, out_spec)?;
    for &s in &out_samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    writer.finalize()?;

    println!("Saved to {}", output_path);
    Ok(())
}
