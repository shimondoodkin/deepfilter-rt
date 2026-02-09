"""
build_onnxsim.py — Foolproof builder for onnx-simplifier (onnxsim) on Windows.

Checks all prerequisites, tells you what's missing, offers to install them,
then clones, builds, and installs the onnxsim wheel.

Usage:
    python build_onnxsim.py
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

WORK_DIR = Path(__file__).parent / "_onnxsim_build"
REPO_URL = "https://github.com/daquexian/onnx-simplifier.git"
# Use master to get latest compatibility (needed for Python 3.12+)
REPO_REF = "master"


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    """Run a command, print it, and return the CompletedProcess."""
    if isinstance(cmd, str):
        print(f"  > {cmd}")
    else:
        print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def run_check(cmd, **kwargs):
    """Run a command and raise on failure."""
    r = run(cmd, **kwargs)
    if r.returncode != 0:
        sys.exit(f"\n[FAIL] Command failed (exit {r.returncode}): {cmd}")
    return r


def which(name):
    """Find an executable on PATH."""
    return shutil.which(name)


def ask_yes_no(prompt):
    """Ask a yes/no question, default yes."""
    while True:
        ans = input(f"{prompt} [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def winget_install(package_id, name=None):
    """Try to install a package via winget."""
    name = name or package_id
    if not which("winget"):
        print(f"  winget not available. Please install {name} manually.")
        return False
    print(f"\n[INSTALL] Installing {name} via winget...")
    r = run(["winget", "install", "--id", package_id, "-e", "--accept-source-agreements",
             "--accept-package-agreements"])
    if r.returncode != 0:
        print(f"  winget install returned {r.returncode}. You may need to install {name} manually.")
        return False
    return True


# ── Prerequisite Checks ─────────────────────────────────────────────────────

def check_python():
    """Verify Python >= 3.8 and show full version info."""
    v = sys.version_info
    print(f"[CHECK] Python ... {v.major}.{v.minor}.{v.micro} ({platform.python_implementation()})")
    print(f"  Executable: {sys.executable}")
    print(f"  Architecture: {platform.architecture()[0]}")
    if v < (3, 8):
        sys.exit("[FAIL] Python 3.8+ is required.")
    if v >= (3, 13):
        print("  NOTE: Python 3.13+ — no prebuilt wheel on PyPI, building from source is the way to go!")
    elif v >= (3, 12):
        print("  NOTE: Python 3.12 — prebuilt wheels may be missing on PyPI for Windows.")
    print("  OK")


def check_git():
    """Check git is available."""
    print("[CHECK] Git ...", end=" ")
    if which("git"):
        r = subprocess.run(["git", "--version"], capture_output=True, text=True)
        print(r.stdout.strip())
        return True

    print("NOT FOUND")
    print("  Git is required to clone the onnxsim repo (with submodules).")
    if ask_yes_no("  Install Git via winget?"):
        winget_install("Git.Git", "Git")
        # winget installs may need a new shell — check again
        if which("git"):
            return True
        # Try adding default git path
        git_default = r"C:\Program Files\Git\cmd"
        if os.path.isdir(git_default):
            os.environ["PATH"] = git_default + ";" + os.environ["PATH"]
            if which("git"):
                print(f"  Added {git_default} to PATH for this session.")
                return True
        print("  Git installed but not yet on PATH. Please restart this script in a new terminal.")
        sys.exit(1)
    else:
        sys.exit("[FAIL] Git is required. Install it from https://git-scm.com/download/win")


def check_cmake():
    """Check cmake is available, install via pip if missing."""
    print("[CHECK] CMake ...", end=" ")
    if which("cmake"):
        r = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        first_line = r.stdout.strip().splitlines()[0] if r.stdout else "found"
        print(first_line)
        return True

    print("NOT FOUND")
    print("  CMake is required to build the C++ components.")
    if ask_yes_no("  Install cmake via pip?"):
        run_check([sys.executable, "-m", "pip", "install", "cmake"])
        # pip installs cmake into Scripts which should be on PATH
        if which("cmake"):
            return True
        # Try to find it in the Scripts folder
        scripts_dir = Path(sys.executable).parent / "Scripts"
        if (scripts_dir / "cmake.exe").exists():
            os.environ["PATH"] = str(scripts_dir) + ";" + os.environ["PATH"]
            print(f"  Added {scripts_dir} to PATH for this session.")
            return True
        print("  cmake installed via pip but not on PATH. Trying winget fallback...")

    if ask_yes_no("  Install CMake via winget?"):
        winget_install("Kitware.CMake", "CMake")
        if which("cmake"):
            return True
        cmake_default = r"C:\Program Files\CMake\bin"
        if os.path.isdir(cmake_default):
            os.environ["PATH"] = cmake_default + ";" + os.environ["PATH"]
            if which("cmake"):
                print(f"  Added {cmake_default} to PATH for this session.")
                return True
        print("  CMake installed but not on PATH. Please restart in a new terminal.")
        sys.exit(1)

    sys.exit("[FAIL] CMake is required. Install from https://cmake.org/download/")


def find_vsdevcmd():
    """Find Visual Studio VsDevCmd.bat for MSVC."""
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

    # 1) Check common paths directly
    for year in ("2022", "2019"):
        for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
            candidate = Path(pf) / "Microsoft Visual Studio" / year / edition / "Common7" / "Tools" / "VsDevCmd.bat"
            if candidate.exists():
                return str(candidate)

    # 2) Use vswhere
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        r = subprocess.run(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True
        )
        if r.returncode == 0 and r.stdout.strip():
            candidate = Path(r.stdout.strip()) / "Common7" / "Tools" / "VsDevCmd.bat"
            if candidate.exists():
                return str(candidate)

    return None


def check_msvc():
    """Check MSVC (cl.exe) is available; activate VS env if needed."""
    print("[CHECK] MSVC (cl.exe) ...", end=" ")

    # Already activated?
    if which("cl"):
        r = subprocess.run(["cl"], capture_output=True, text=True)
        # cl prints version to stderr
        ver_line = r.stderr.strip().splitlines()[0] if r.stderr else "found"
        print(ver_line)
        return True

    print("NOT FOUND in current environment")

    vsdevcmd = find_vsdevcmd()
    if vsdevcmd:
        print(f"  Found: {vsdevcmd}")
        print("  Activating MSVC environment (x64)...")

        # Run VsDevCmd and capture the resulting environment
        cmd = f'"{vsdevcmd}" -arch=x64 -host_arch=x64 >nul 2>&1 && set'
        r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if r.returncode != 0:
            sys.exit(f"[FAIL] VsDevCmd.bat failed:\n{r.stderr}")

        # Parse and apply environment variables
        for line in r.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key] = val

        if which("cl"):
            print("  OK — MSVC activated")
            return True
        else:
            sys.exit("[FAIL] VsDevCmd ran but cl.exe still not found. Check your VS install.")

    # Not found at all
    print("\n  Visual Studio with C++ tools is required.")
    print("  You need 'Desktop development with C++' workload installed.")
    if which("winget"):
        print("\n  You can install Visual Studio Build Tools via winget:")
        print("    winget install Microsoft.VisualStudio.2022.BuildTools")
        print("  Then launch 'Visual Studio Installer' and add 'Desktop development with C++' workload.")
        if ask_yes_no("  Install Visual Studio 2022 Build Tools via winget now?"):
            winget_install("Microsoft.VisualStudio.2022.BuildTools", "VS 2022 Build Tools")
            print("\n  After install completes, open 'Visual Studio Installer',")
            print("  click 'Modify' and enable 'Desktop development with C++',")
            print("  then re-run this script.")
            sys.exit(0)
    else:
        print("  Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    sys.exit(1)


# ── Clone & Build ────────────────────────────────────────────────────────────

def clone_source():
    """Clone onnx-simplifier with submodules."""
    src = WORK_DIR / "onnx-simplifier"

    # Reuse existing clone if present
    if (src / "setup.py").exists():
        print(f"\n[SKIP] Existing source found at {src}, reusing it.")
        if not (src / "third_party" / "pybind11" / "CMakeLists.txt").exists():
            print("  [WARN] pybind11 submodule may be empty, trying manual init...")
            run_check(["git", "submodule", "update", "--init", "--recursive"], cwd=str(src))
        return src

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[CLONE] Cloning onnx-simplifier ({REPO_REF}) with submodules...")
    run_check([
        "git", "clone",
        "--depth", "1",
        "--branch", REPO_REF,
        "--recurse-submodules", "--shallow-submodules",
        REPO_URL,
        str(src)
    ])

    # Verify critical paths
    if not (src / "setup.py").exists():
        sys.exit(f"[FAIL] setup.py not found in {src}")
    if not (src / "third_party" / "pybind11" / "CMakeLists.txt").exists():
        print("  [WARN] pybind11 submodule may be empty, trying manual init...")
        run_check(["git", "submodule", "update", "--init", "--recursive"], cwd=str(src))

    return src


def install_python_deps():
    """Install Python build dependencies."""
    print("\n[DEPS] Installing Python build dependencies...")
    run_check([sys.executable, "-m", "pip", "install", "--upgrade",
               "pip", "setuptools", "wheel", "numpy", "protobuf", "onnx", "rich"])


def build_wheel(src):
    """Build the onnxsim wheel using setup.py."""
    print("\n[BUILD] Building onnxsim wheel (this may take a few minutes)...")
    env = os.environ.copy()
    # Ensure cmake from pip is findable
    scripts_dir = str(Path(sys.executable).parent / "Scripts")
    if scripts_dir not in env.get("PATH", ""):
        env["PATH"] = scripts_dir + ";" + env["PATH"]

    # Extra CMake flags passed via CMAKE_ARGS (setup.py reads this)
    #  - CMAKE_POLICY_VERSION_MINIMUM=3.5 : fix onnx submodule's outdated cmake_minimum_required
    #  - protobuf_WITH_ZLIB=OFF           : skip zlib dependency (not bundled on Windows)
    cmake_extra = [
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-Dprotobuf_WITH_ZLIB=OFF",
    ]
    env["CMAKE_ARGS"] = " ".join(cmake_extra)

    run_check(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=str(src),
        env=env
    )

    # Find produced wheel
    dist_dir = src / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"[FAIL] No wheel found in {dist_dir}")

    wheel = wheels[0]
    print(f"  Wheel: {wheel}")
    return wheel


def install_wheel(wheel):
    """Install the built wheel."""
    print(f"\n[INSTALL] Installing {wheel.name}...")
    run_check([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)])


def verify_install():
    """Quick import test."""
    print("\n[TEST] Verifying onnxsim import...")
    r = run([sys.executable, "-c",
             "import onnxsim; print('  onnxsim version:', getattr(onnxsim, '__version__', '(unknown)'))"])
    if r.returncode != 0:
        print("[WARN] Import test failed. The wheel was built but may have issues.")
        return False
    return True


def copy_wheel_out(wheel):
    """Copy wheel to the scripts dir for easy access."""
    dest = Path(__file__).parent / wheel.name
    if dest != wheel:
        shutil.copy2(wheel, dest)
        print(f"  Wheel copied to: {dest}")
    return dest


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  onnxsim Windows wheel builder")
    print("=" * 60)
    print()

    if platform.system() != "Windows":
        sys.exit("[FAIL] This script is for Windows only.")

    # Check all prerequisites
    check_python()
    check_git()
    check_cmake()
    check_msvc()

    print("\n" + "-" * 60)
    print("  All prerequisites satisfied!")
    print("-" * 60)

    # Build
    install_python_deps()
    src = clone_source()
    wheel = build_wheel(src)
    final = copy_wheel_out(wheel)

    print()
    print("=" * 60)
    print("  SUCCESS: onnxsim wheel built!")
    print("=" * 60)
    print()
    print(f"  Wheel file: {final}")
    print()
    print("  To install it, run:")
    print(f'    pip install "{final}"')
    print()


if __name__ == "__main__":
    main()
