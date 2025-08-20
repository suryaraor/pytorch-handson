"""
Chapter 1.1: PyTorch Installation Check
=====================================

This script checks if PyTorch is properly installed and provides
detailed information about your setup.

Author: PyTorch Deep Dive Tutorial
Date: 2025
"""

import sys
import platform

def check_python_version():
    """Check if Python version is compatible with PyTorch."""
    print("🐍 Python Version Check")
    print("=" * 50)
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible with PyTorch")
    else:
        print("❌ Python version is too old. PyTorch requires Python 3.8+")
    
    print("\n")

def check_pytorch_installation():
    """Check if PyTorch is installed and get version info."""
    print("🔥 PyTorch Installation Check")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch is installed!")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch File Location: {torch.__file__}")
        
        # Check if compiled with CUDA
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  CUDA is not available. You'll be using CPU for computations.")
        
        return True
        
    except ImportError:
        print("❌ PyTorch is not installed!")
        print("\nTo install PyTorch, visit: https://pytorch.org/get-started/locally/")
        print("For CPU-only version: pip install torch torchvision torchaudio")
        print("For CUDA version: Choose from the PyTorch website based on your CUDA version")
        return False
    
    print("\n")

def check_related_libraries():
    """Check if related PyTorch libraries are installed."""
    print("📦 Related Libraries Check")
    print("=" * 50)
    
    libraries = {
        'torchvision': 'Computer Vision utilities',
        'torchaudio': 'Audio processing utilities',
        'numpy': 'Numerical computing foundation',
        'matplotlib': 'Plotting and visualization',
        'pandas': 'Data manipulation and analysis'
    }
    
    for lib, description in libraries.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {lib} ({version}) - {description}")
        except ImportError:
            print(f"❌ {lib} - {description} (Not installed)")
    
    print("\n")

def basic_pytorch_test():
    """Perform a basic PyTorch operation test."""
    print("🧪 Basic PyTorch Operations Test")
    print("=" * 50)
    
    try:
        import torch
        
        # Create a simple tensor
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        print(f"Created tensor: {x}")
        print(f"Tensor shape: {x.shape}")
        print(f"Tensor dtype: {x.dtype}")
                
        # Basic operations
        y = x * 2
        print(f"Multiplication result: {y}")
        
        z = torch.sum(x)
        print(f"Sum result: {z}")
        
        # Test GPU if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print(f"✅ GPU tensor created: {x_gpu}")
            print(f"GPU tensor device: {x_gpu.device}")
        
        print("✅ All basic operations work correctly!")
        
    except Exception as e:
        print(f"❌ Error during basic operations: {e}")
    
    print("\n")

def system_info():
    """Display system information relevant to PyTorch."""
    print("💻 System Information")
    print("=" * 50)
    
    import os
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Count: {os.cpu_count()}")
    
    # Memory information (if psutil is available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("Memory info not available (psutil not installed)")
    
    print("\n")

def main():
    """Main function to run all checks."""
    print("🚀 PyTorch Deep Dive - Installation Check")
    print("=" * 60)
    print("This script will verify your PyTorch setup and environment.\n")
    
    # Run all checks
    check_python_version()
    
    pytorch_installed = check_pytorch_installation()
    
    if pytorch_installed:
        check_related_libraries()
        basic_pytorch_test()
    
    system_info()
    
    print("🎉 Installation check complete!")
    print("\nIf everything looks good, you're ready to proceed to the next chapter!")
    print("If you see any errors, refer to the README.md for troubleshooting tips.")

if __name__ == "__main__":
    main()
