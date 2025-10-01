#!/usr/bin/env python3
"""
PyTorch Installation Script for BICSNet-PIV

This script automatically detects the system architecture and installs the appropriate
PyTorch version (GPU or CPU) using uv package manager.

Usage:
    python install_pytorch.py [--force-cpu] [--cuda-version 121]
"""

import subprocess
import sys
import platform
import argparse
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None, e.stderr


def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        # Try to run nvidia-smi
        stdout, stderr = run_command("nvidia-smi", check=False)
        if stdout and "NVIDIA" in stdout:
            return True
    except:
        pass
    
    # Try to check for CUDA libraries
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    
    return False


def get_cuda_version():
    """Get the CUDA version if available."""
    try:
        stdout, stderr = run_command("nvcc --version", check=False)
        if stdout:
            # Parse version from nvcc output
            for line in stdout.split('\n'):
                if 'release' in line.lower():
                    version = line.split('release')[1].split(',')[0].strip()
                    # Extract major.minor version
                    major_minor = '.'.join(version.split('.')[:2])
                    return major_minor
    except:
        pass
    
    # Try to get from nvidia-smi
    try:
        stdout, stderr = run_command("nvidia-smi", check=False)
        if stdout:
            for line in stdout.split('\n'):
                if 'CUDA Version' in line:
                    version = line.split('CUDA Version:')[1].strip().split()[0]
                    return version
    except:
        pass
    
    return None


def install_pytorch_gpu(cuda_version="121"):
    """Install PyTorch with CUDA support."""
    print(f"Installing PyTorch with CUDA {cuda_version} support...")
    
    # Use uv to install from PyTorch's CUDA index
    cmd = f'uv add --index-url https://download.pytorch.org/whl/cu{cuda_version} torch torchvision'
    
    stdout, stderr = run_command(cmd)
    if stdout is not None:
        print("✅ PyTorch with CUDA support installed successfully!")
        return True
    else:
        print("❌ Failed to install PyTorch with CUDA support")
        return False


def install_pytorch_cpu():
    """Install PyTorch CPU-only version."""
    print("Installing PyTorch CPU-only version...")
    
    # Use uv to install from PyTorch's CPU index
    cmd = 'uv add --index-url https://download.pytorch.org/whl/cpu torch torchvision'
    
    stdout, stderr = run_command(cmd)
    if stdout is not None:
        print("✅ PyTorch CPU-only version installed successfully!")
        return True
    else:
        print("❌ Failed to install PyTorch CPU-only version")
        return False


def verify_installation():
    """Verify PyTorch installation and show device information."""
    try:
        import torch
        print("\n🔍 PyTorch Installation Verification:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("Running on CPU")
            
        # Test basic functionality
        x = torch.randn(2, 3)
        print(f"✅ Basic tensor operation successful: {x.shape}")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Install PyTorch with automatic GPU/CPU detection")
    parser.add_argument("--force-cpu", action="store_true", 
                       help="Force CPU-only installation even if CUDA is available")
    parser.add_argument("--cuda-version", default="121", 
                       help="CUDA version for GPU installation (default: 121)")
    
    args = parser.parse_args()
    
    print("🚀 BICSNet-PIV PyTorch Installation Script")
    print("=" * 50)
    
    # Check if uv is available
    stdout, stderr = run_command("uv --version", check=False)
    if stdout is None:
        print("❌ uv package manager not found. Please install uv first:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    print(f"✅ Found uv: {stdout}")
    
    # System information
    print(f"\n🖥️  System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Check CUDA availability
    cuda_available = check_cuda_available()
    cuda_version = get_cuda_version()
    
    print(f"\n🔍 CUDA Detection:")
    print(f"CUDA available: {cuda_available}")
    if cuda_version:
        print(f"CUDA version: {cuda_version}")
    
    # Decide installation method
    if args.force_cpu:
        print("\n⚙️  Forcing CPU-only installation...")
        success = install_pytorch_cpu()
    elif cuda_available:
        print(f"\n⚙️  CUDA detected, installing GPU version...")
        success = install_pytorch_gpu(args.cuda_version)
    else:
        print("\n⚙️  No CUDA detected, installing CPU version...")
        success = install_pytorch_cpu()
    
    if success:
        print("\n🔍 Verifying installation...")
        verify_installation()
        print("\n✅ Installation completed successfully!")
        print("\nNext steps:")
        print("1. Activate your virtual environment: source .venv/bin/activate")
        print("2. Run the inference script: python src/pivnet_image_gen.py")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
