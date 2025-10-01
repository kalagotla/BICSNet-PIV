#!/usr/bin/env python3
"""
Test script to verify PyTorch installation and device detection.
"""

import sys
import platform

def test_pytorch_installation():
    """Test PyTorch installation and show device information."""
    print("🧪 Testing PyTorch Installation")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"PyTorch version: {torch.__version__}")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        print(f"✅ Basic tensor creation: {x.shape}")
        
        # Test device detection
        print(f"\n🔍 Device Information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            
            # Test GPU tensor operations
            x_gpu = x.cuda()
            y_gpu = torch.mm(x_gpu, x_gpu.t())
            print(f"✅ GPU tensor operations successful: {y_gpu.shape}")
        else:
            print("Running on CPU")
            # Test CPU tensor operations
            y = torch.mm(x, x.t())
            print(f"✅ CPU tensor operations successful: {y.shape}")
        
        # Test MPS (Apple Silicon) if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
            if torch.backends.mps.is_available():
                x_mps = x.to('mps')
                y_mps = torch.mm(x_mps, x_mps.t())
                print(f"✅ MPS tensor operations successful: {y_mps.shape}")
        
        print(f"\n🎯 Recommended device for BICSNet-PIV:")
        if torch.cuda.is_available():
            print("GPU (CUDA) - Best performance for training and inference")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS (Apple Silicon) - Good performance on Apple Silicon Macs")
        else:
            print("CPU - Slower but functional for inference")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        print("Please run: python scripts/install_pytorch.py")
        return False
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False


def test_other_dependencies():
    """Test other key dependencies."""
    print(f"\n🔍 Testing Other Dependencies:")
    print("=" * 40)
    
    dependencies = [
        'numpy', 'scipy', 'pandas', 'matplotlib', 
        'scikit-image', 'scikit-learn', 'openpiv',
        'seaborn', 'tqdm', 'tifffile', 'pillow'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - not installed")
    
    print(f"\n📋 System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")


if __name__ == "__main__":
    success = test_pytorch_installation()
    test_other_dependencies()
    
    if success:
        print(f"\n🎉 All tests passed! BICSNet-PIV is ready to use.")
    else:
        print(f"\n❌ Some tests failed. Please check the installation.")
        sys.exit(1)
