#!/usr/bin/env python3
"""
Cross-platform uninstaller for BICSNet-PIV.

Removes the project's virtual environment and the Jupyter kernel created by the installer.

Usage:
  python scripts/uninstall.py                  # prompts for confirmation
  python scripts/uninstall.py --yes            # non-interactive
  python scripts/uninstall.py --venv .venv2    # custom venv path
  python scripts/uninstall.py --kernel-name bicsnet-piv  # custom kernel name
  python scripts/uninstall.py --keep-kernel    # only remove venv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"
DEFAULT_KERNEL = "bicsnet-piv"


def prompt_yes(question: str) -> bool:
    try:
        reply = input(f"{question} [y/N]: ").strip().lower()
        return reply in {"y", "yes"}
    except EOFError:
        return False


def remove_venv(venv_dir: Path) -> None:
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        print(f"Removed virtual environment: {venv_dir}")
    else:
        print(f"Virtual environment not found (skipped): {venv_dir}")


def remove_jupyter_kernel(kernel_name: str) -> None:
    jupyter = shutil.which("jupyter")
    if not jupyter:
        print("'jupyter' not found on PATH. Skipping kernel removal.")
        return
    try:
        subprocess.run([jupyter, "kernelspec", "remove", kernel_name, "-y"], check=True)
        print(f"Removed Jupyter kernel: {kernel_name}")
    except subprocess.CalledProcessError:
        print(f"Kernel '{kernel_name}' not found (skipped).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uninstall BICSNet-PIV environment and kernel")
    parser.add_argument("--venv", type=str, default=str(DEFAULT_VENV_DIR), help="Path to the virtualenv to remove")
    parser.add_argument("--kernel-name", type=str, default=DEFAULT_KERNEL, help="Jupyter kernel name to remove")
    parser.add_argument("--keep-kernel", action="store_true", help="Do not remove Jupyter kernel")
    parser.add_argument("--yes", "-y", action="store_true", help="Do not prompt for confirmation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv_dir = Path(args.venv).resolve()

    if not args.yes:
        if not prompt_yes(
            f"This will remove the virtual environment at '{venv_dir}'"
            + (" and the Jupyter kernel '" + args.kernel_name + "'" if not args.keep_kernel else "")
            + ". Continue?"
        ):
            print("Aborted.")
            return

    remove_venv(venv_dir)
    if not args.keep_kernel:
        remove_jupyter_kernel(args.kernel_name)

    print("Uninstall complete.")


if __name__ == "__main__":
    main()


