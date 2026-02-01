//! Example: Process an audio file with DeepFilterNet (auto mode selection)
//!
//! Usage: cargo run --example process_file -- input.wav output.wav [model_dir/]

use deepfilter_rt::{DeepFilterStream, SAMPLE_RATE};
use std::path::Path;

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
        // Default to models/dfn3 relative to crate root
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/dfn3")
    };

    // Create processor
    println!("Loading models from {:?}...", model_dir);
    let mut stream = DeepFilterStream::new(&model_dir)?;
    let variant = stream.variant();
    println!("Using model variant: {}", variant.name());
    println!("Inference mode: {}", if variant.is_stateful() { "stateful (h0)" } else { "stateless" });

    // Warm up to avoid cold-start latency affecting timing
    stream.warmup()?;

    // Read input audio
    let mut reader = hound::WavReader::open(input_path)?;
    let spec = reader.spec();

    println!("Input: {} Hz, {} channels, {:?}",
             spec.sample_rate, spec.channels, spec.sample_format);

    if spec.sample_rate != SAMPLE_RATE as u32 {
        eprintln!("Warning: Input sample rate {} != expected {}. Resample first!",
                  spec.sample_rate, SAMPLE_RATE);
    }

    // Read samples
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => reader
                    .samples::<i16>()
                    .map(|s| s.unwrap() as f32 / 32768.0)
                    .collect(),
                24 | 32 => reader
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 / 2147483648.0)
                    .collect(),
                _ => {
                    eprintln!("Unsupported bits per sample: {}", spec.bits_per_sample);
                    std::process::exit(1);
                }
            }
        }
    };

    // Convert to mono if stereo
    let mono: Vec<f32> = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else if spec.channels == 1 {
        samples
    } else {
        // Take first channel for multi-channel
        samples.chunks(spec.channels as usize).map(|c| c[0]).collect()
    };

    println!("Processing {} samples ({:.2}s)...", mono.len(),
             mono.len() as f32 / SAMPLE_RATE as f32);

    // Process using streaming API
    let start = std::time::Instant::now();

    let mut output = stream.process(&mono)?;
    // Flush any remaining samples
    output.extend(stream.flush()?);

    let elapsed = start.elapsed();
    let rtf = elapsed.as_secs_f32() / (mono.len() as f32 / SAMPLE_RATE as f32);
    println!("Done in {:.2}s (RTF: {:.3}x realtime)", elapsed.as_secs_f32(), rtf);

    // Write output
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, out_spec)?;
    for sample in &output {
        let s = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(s)?;
    }
    writer.finalize()?;

    println!("Saved to {}", output_path);
    Ok(())
}
