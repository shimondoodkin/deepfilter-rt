//! Build script for deepfilter-rt
//!
//! Downloads ONNX Runtime if not present and exposes paths for dependent crates.
//! When the `cuda` feature is enabled, downloads the GPU variant with CUDA/TensorRT support.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const ORT_VERSION: &str = "1.23.2";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "windows".to_string());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "x86_64".to_string());
    let use_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();

    // Determine ORT folder name and download URL based on target
    let (ort_folder, ort_url) = get_ort_info(&target_os, &target_arch, use_cuda);

    let ort_dir = manifest_dir.join(&ort_folder);
    let ort_lib_dir = ort_dir.join("lib");

    // Download if not present
    if !ort_lib_dir.exists() {
        let variant = if use_cuda { "GPU" } else { "CPU" };
        println!("cargo:warning=ONNX Runtime {} not found, downloading v{}...", variant, ORT_VERSION);
        if let Err(e) = download_and_extract(&ort_url, &manifest_dir, &ort_folder) {
            println!("cargo:warning=Failed to download ONNX Runtime: {}", e);
            println!("cargo:warning=Please download manually from: {}", ort_url);
        }
    }

    // Copy DLLs to target directory for runtime availability
    if target_os == "windows" {
        copy_dlls_to_target(&ort_lib_dir, use_cuda);
    }

    // Export paths for dependent crates
    println!("cargo:ort_lib_dir={}", ort_lib_dir.display());
    println!("cargo:models_dir={}", manifest_dir.join("models").display());

    // Rerun if these change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
}

fn get_ort_info(target_os: &str, target_arch: &str, use_cuda: bool) -> (String, String) {
    let base_url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{}",
        ORT_VERSION
    );

    match (target_os, target_arch, use_cuda) {
        // Windows x64 with CUDA/TensorRT
        ("windows", "x86_64", true) => (
            format!("onnxruntime-win-x64-gpu-{}", ORT_VERSION),
            format!("{}/onnxruntime-win-x64-gpu-{}.zip", base_url, ORT_VERSION),
        ),
        // Windows x64 CPU only
        ("windows", "x86_64", false) => (
            format!("onnxruntime-win-x64-{}", ORT_VERSION),
            format!("{}/onnxruntime-win-x64-{}.zip", base_url, ORT_VERSION),
        ),
        ("windows", "aarch64", _) => (
            format!("onnxruntime-win-arm64-{}", ORT_VERSION),
            format!("{}/onnxruntime-win-arm64-{}.zip", base_url, ORT_VERSION),
        ),
        // Linux x64 with CUDA
        ("linux", "x86_64", true) => (
            format!("onnxruntime-linux-x64-gpu-{}", ORT_VERSION),
            format!("{}/onnxruntime-linux-x64-gpu-{}.tgz", base_url, ORT_VERSION),
        ),
        // Linux x64 CPU only
        ("linux", "x86_64", false) => (
            format!("onnxruntime-linux-x64-{}", ORT_VERSION),
            format!("{}/onnxruntime-linux-x64-{}.tgz", base_url, ORT_VERSION),
        ),
        ("linux", "aarch64", _) => (
            format!("onnxruntime-linux-aarch64-{}", ORT_VERSION),
            format!("{}/onnxruntime-linux-aarch64-{}.tgz", base_url, ORT_VERSION),
        ),
        ("macos", "x86_64", _) => (
            format!("onnxruntime-osx-x64-{}", ORT_VERSION),
            format!("{}/onnxruntime-osx-x64-{}.tgz", base_url, ORT_VERSION),
        ),
        ("macos", "aarch64", _) => (
            format!("onnxruntime-osx-arm64-{}", ORT_VERSION),
            format!("{}/onnxruntime-osx-arm64-{}.tgz", base_url, ORT_VERSION),
        ),
        _ => {
            println!("cargo:warning=Unsupported target: {}-{}", target_os, target_arch);
            (format!("onnxruntime-{}-{}-{}", target_os, target_arch, ORT_VERSION), String::new())
        }
    }
}

/// Copy required DLLs to the target directory so they're available at runtime
fn copy_dlls_to_target(ort_lib_dir: &Path, use_cuda: bool) {
    let out_dir = match env::var("OUT_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => return,
    };

    // Target bin is typically 3 levels up from OUT_DIR: target/<profile>/build/<pkg>/out
    let target_bin = match out_dir.ancestors().nth(3) {
        Some(p) => p.to_path_buf(),
        None => return,
    };

    // Base DLLs always needed
    let mut dlls = vec!["onnxruntime.dll"];

    // GPU-specific DLLs
    if use_cuda {
        dlls.extend([
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_cuda.dll",
            "onnxruntime_providers_tensorrt.dll",
        ]);
    }

    for dll in dlls {
        let src = ort_lib_dir.join(dll);
        let dst = target_bin.join(dll);
        if src.exists() {
            match fs::copy(&src, &dst) {
                Ok(_) => println!("cargo:warning=Copied {} to {}", dll, target_bin.display()),
                Err(e) => println!("cargo:warning=Failed to copy {}: {}", dll, e),
            }
        }
    }
}

fn download_and_extract(url: &str, dest_dir: &Path, folder_name: &str) -> io::Result<()> {
    if url.is_empty() {
        return Err(io::Error::new(io::ErrorKind::Other, "No download URL for this platform"));
    }

    let archive_path = dest_dir.join(if url.ends_with(".zip") {
        "onnxruntime.zip"
    } else {
        "onnxruntime.tgz"
    });

    // Download using curl (available on Windows 10+, Linux, macOS)
    println!("cargo:warning=Downloading from {}", url);
    let status = std::process::Command::new("curl")
        .args(["-L", "-o", archive_path.to_str().unwrap(), url])
        .status()?;

    if !status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "curl download failed"));
    }

    // Extract
    println!("cargo:warning=Extracting to {}", dest_dir.display());
    if url.ends_with(".zip") {
        extract_zip(&archive_path, dest_dir)?;
    } else {
        extract_tgz(&archive_path, dest_dir)?;
    }

    // Cleanup archive
    let _ = fs::remove_file(&archive_path);

    // Verify extraction
    let extracted = dest_dir.join(folder_name);
    if !extracted.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Expected folder {} not found after extraction", folder_name),
        ));
    }

    println!("cargo:warning=ONNX Runtime {} ready", ORT_VERSION);
    Ok(())
}

fn extract_zip(archive: &Path, dest: &Path) -> io::Result<()> {
    // Use PowerShell on Windows
    #[cfg(target_os = "windows")]
    {
        let status = std::process::Command::new("powershell")
            .args([
                "-Command",
                &format!(
                    "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                    archive.display(),
                    dest.display()
                ),
            ])
            .status()?;
        if !status.success() {
            return Err(io::Error::new(io::ErrorKind::Other, "PowerShell extract failed"));
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let status = std::process::Command::new("unzip")
            .args(["-o", archive.to_str().unwrap(), "-d", dest.to_str().unwrap()])
            .status()?;
        if !status.success() {
            return Err(io::Error::new(io::ErrorKind::Other, "unzip failed"));
        }
    }

    Ok(())
}

fn extract_tgz(archive: &Path, dest: &Path) -> io::Result<()> {
    let status = std::process::Command::new("tar")
        .args(["-xzf", archive.to_str().unwrap(), "-C", dest.to_str().unwrap()])
        .status()?;
    if !status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "tar extract failed"));
    }
    Ok(())
}
