#!/usr/bin/env python3
"""
Cross-platform installer for BICSNet-PIV.

Features:
- Creates a virtual environment in .venv (configurable) using the current Python.
- Installs base dependencies from pyproject.toml using pip (no external tool required).
- Installs the appropriate PyTorch build:
  * --cuda: NVIDIA CUDA wheels
  * --cpu: CPU-only wheels (also fine for Apple Silicon; MPS will be available via default torch on macOS)
  * auto (default): chooses CUDA if a compatible NVIDIA GPU/CUDA toolkit is detected, otherwise CPU
- Registers ipykernel for Jupyter (optional, enabled by default).

Usage examples:
  python scripts/install.py           # auto-detect CUDA/CPU
  python scripts/install.py --cpu     # force CPU build
  python scripts/install.py --cuda    # force CUDA build
  python scripts/install.py --venv .venv-bicsnet
  python scripts/install.py --no-kernel
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    completed = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def create_virtualenv_with_uv(venv_dir: Path, python_path: str | None, allow_pip: bool) -> Path:
    if venv_dir.exists():
        return venv_dir
    # Prefer uv to create the venv (manages metadata used by uv sync)
    try:
        run([(python_path or sys.executable), "-m", "uv", "--version"])  # ensure uv visible
    except SystemExit:
        # install uv into bootstrap interpreter, then use it
        if allow_pip:
            run([python_path or sys.executable, "-m", "pip", "install", "uv"])  # install uv
        # verify again (if it still fails, let uv venv attempt below handle)
    uv_args = [(python_path or sys.executable), "-m", "uv", "venv"]
    if python_path:
        uv_args += ["--python", python_path]
    uv_args += [str(venv_dir)]
    try:
        run(uv_args)
    except SystemExit:
        # fallback: stdlib venv as last resort
        run([python_path or sys.executable, "-m", "venv", str(venv_dir)])
    return venv_dir


def venv_python(venv_dir: Path) -> str:
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def pip_install(python_bin: str, packages: list[str]) -> None:
    if not packages:
        return
    run([python_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([python_bin, "-m", "pip", "install", *packages])


def uv_install(python_bin: str, packages: list[str]) -> None:
    if not packages:
        return
    # Try to run uv via the venv's Python first; install if missing
    try:
        run([python_bin, "-m", "uv", "--version"])
    except SystemExit:
        # Install uv into the venv and retry
        try:
            run([python_bin, "-m", "pip", "install", "uv"])  # uv publishes wheels
            run([python_bin, "-m", "uv", "--version"])  # verify
        except SystemExit:
            raise SystemExit("uv is required unless --pip is specified. Failed to install uv into the virtual environment.")
    # Target the specific interpreter using uv pip -p
    run([python_bin, "-m", "uv", "pip", "-p", python_bin, "install", *packages])


def read_pyproject() -> dict:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if tomllib is None:
        return {}
    with pyproject_path.open("rb") as f:
        return tomllib.load(f)


def read_base_dependencies() -> list[str]:
    data = read_pyproject()
    deps: list[str] = []
    try:
        raw = data.get("project", {}).get("dependencies", [])  # type: ignore[assignment]
        for dep in raw:
            low = dep.lower()
            if low.startswith("torch") or low.startswith("numpy"):
                continue
            deps.append(dep)
    except Exception:
        # Fallback to hardcoded list if parsing fails
        deps = [
            "matplotlib>=3.10.6",
            "jupyter>=1.1.1",
            "openpiv>=0.25.0",
            "pandas>=2.3.3",
            "pillow>=11.3.0",
            "scikit-image>=0.25.2",
            "scikit-learn>=1.7.2",
            "scipy>=1.16.2",
            "seaborn>=0.13.2",
            "tifffile>=2025.9.30",
            "tqdm>=4.67.1",
            "ipykernel>=6.30.1",
            "huggingface_hub>=0.25.0",
        ]
    return deps


def read_python_requires() -> tuple[int, int] | None:
    data = read_pyproject()
    spec = (data.get("project", {}) or {}).get("requires-python") if data else None
    # Expect format like ">=3.12,<3.13"; use the lower bound
    if isinstance(spec, str) and spec.startswith(">="):
        try:
            version_part = spec.split(",")[0].replace(">=", "").strip()
            major, minor = version_part.split(".")[:2]
            return int(major), int(minor)
        except Exception:
            return None
    return None


def detect_cuda_available() -> bool:
    # Heuristic: try nvidia-smi; if available, assume CUDA-capable
    try:
        completed = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return completed.returncode == 0
    except FileNotFoundError:
        return False


def install_torch(python_bin: str, prefer: str) -> None:
    if prefer == "cuda":
        uv_install(python_bin, [
            "torch==2.2.2", "torchvision==0.17.2",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ])
    elif prefer == "cpu":
        uv_install(python_bin, [
            "torch==2.2.2", "torchvision==0.17.2",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])
    else:
        # default channel (auto handles macOS MPS, CPU on others)
        uv_install(python_bin, ["torch==2.2.2", "torchvision==0.17.2"]) 


def ensure_ipykernel(python_bin: str, name: str, display_name: str) -> None:
    run([python_bin, "-m", "ipykernel", "install", "--user", "--name", name, "--display-name", display_name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install BICSNet-PIV in a virtualenv")
    parser.add_argument("--venv", type=str, default=str(DEFAULT_VENV_DIR), help="Path to create/use the virtualenv")
    parser.add_argument("--python", type=str, default=None, help="Path to the Python interpreter to use for the venv")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="Force CPU-only PyTorch")
    group.add_argument("--cuda", action="store_true", help="Force CUDA PyTorch (NVIDIA)")
    parser.add_argument("--no-kernel", action="store_true", help="Skip creating a Jupyter kernel")
    parser.add_argument("--pip", action="store_true", help="Use pip instead of uv (override)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv_dir = Path(args.venv).resolve()

    # Python version guard based on pyproject requires-python
    requires = read_python_requires()
    if requires is not None:
        req_major, req_minor = requires
        if args.python:
            # Check the provided interpreter version
            code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            out = subprocess.check_output([args.python, "-c", code], text=True).strip()
            maj_str, min_str = out.split(".")[:2]
            if (int(maj_str), int(min_str)) < (req_major, req_minor):
                raise SystemExit(f"Python>={req_major}.{req_minor} required, got {out} at --python")
        else:
            if (sys.version_info.major, sys.version_info.minor) < (req_major, req_minor):
                raise SystemExit(f"Python>={req_major}.{req_minor} required. Provide --python to select a compatible interpreter.")

    print(f"[1/6] Creating virtual environment at: {venv_dir}")
    # If using uv path (default), create venv via uv; else stdlib venv
    if args.pip:
        run([args.python or sys.executable, "-m", "venv", str(venv_dir)])
    else:
        create_virtualenv_with_uv(venv_dir, args.python, allow_pip=True)
    python_bin = venv_python(venv_dir)

    print(f"[2/6] Selecting PyTorch build (CUDA/CPU)")
    preferred = "auto"
    if args.cpu:
        preferred = "cpu"
    elif args.cuda:
        preferred = "cuda"
    else:
        preferred = "cuda" if detect_cuda_available() else "cpu"
    print(f"[3/6] Installing PyTorch: {preferred}")
    if args.pip:
        if preferred == "cuda":
            pip_install(python_bin, [
                "torch==2.2.2", "torchvision==0.17.2",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            ])
        elif preferred == "cpu":
            pip_install(python_bin, [
                "torch==2.2.2", "torchvision==0.17.2",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            ])
        else:
            pip_install(python_bin, ["torch==2.2.2", "torchvision==0.17.2"]) 
    else:
        # Install torch first using uv pip
        if preferred == "cuda":
            uv_install(python_bin, [
                "torch==2.2.2", "torchvision==0.17.2",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            ])
        elif preferred == "cpu":
            uv_install(python_bin, [
                "torch==2.2.2", "torchvision==0.17.2",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            ])
        else:
            uv_install(python_bin, ["torch==2.2.2", "torchvision==0.17.2"]) 

    print(f"[4/6] Installing remaining dependencies (excluding numpy)")
    base_deps = read_base_dependencies()
    if args.pip:
        pip_install(python_bin, base_deps)
    else:
        uv_install(python_bin, base_deps)

    print(f"[5/6] Verifying torch and CUDA availability")
    check_code = (
        "import torch;\n"
        "print('torch', torch.__version__);\n"
        "print('cuda_available', torch.cuda.is_available());\n"
        "print('mps_available', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())\n"
    )
    run([python_bin, "-c", check_code])

    if not args.no_kernel:
        print(f"[6/6] Registering Jupyter kernel")
        kernel_name = "bicsnet-piv"
        display_name = f"Python ({kernel_name})"
        ensure_ipykernel(python_bin, kernel_name, display_name)
    else:
        print(f"[6/6] Skipping Jupyter kernel registration")

    print(f"Final import smoke test")
    run([python_bin, "-c", "import numpy, openpiv; print('OK')"])  # basic import smoke test

    activate_hint = (
        f"Activate with: {venv_dir / 'Scripts' / 'activate.bat' if platform.system()=='Windows' else 'source ' + str(venv_dir / 'bin' / 'activate')}"
    )
    print("\nInstallation complete.")
    print(activate_hint)


if __name__ == "__main__":
    main()


