"""
Chapter 1.4: GPU Setup and Configuration
=======================================

Learn how to configure and optimize GPU usage with PyTorch.

Author: PyTorch Deep Dive Tutorial
Date: 2025
"""

import os
import platform

def check_nvidia_setup():
    """Check NVIDIA driver and CUDA installation."""
    print("🎮 NVIDIA GPU Setup Check")
    print("=" * 50)
    
    print("System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Check nvidia-smi
    print("\n🔍 Checking NVIDIA Driver...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed and working!")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    driver_version = line.split('Driver Version:')[1].split()[0]
                    print(f"Driver Version: {driver_version}")
                    break
            
            # Extract GPU information
            gpu_info = []
            for line in lines:
                if 'NVIDIA' in line and 'MiB' in line:
                    gpu_info.append(line.strip())
            
            if gpu_info:
                print("\nGPU Information:")
                for i, info in enumerate(gpu_info):
                    print(f"  GPU {i}: {info}")
        else:
            print("❌ nvidia-smi failed. NVIDIA driver may not be installed.")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found. NVIDIA driver may not be installed.")
    except Exception as e:
        print(f"❌ Error checking NVIDIA driver: {e}")
    
    print("\n")

def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print("🔥 PyTorch CUDA Support")
    print("=" * 50)
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch!")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            # Device information
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Multi-processors: {props.multi_processor_count}")
                
            # Current device
            current_device = torch.cuda.current_device()
            print(f"\nCurrent Device: {current_device}")
            print(f"Current Device Name: {torch.cuda.get_device_name(current_device)}")
            
        else:
            print("❌ CUDA is not available in PyTorch")
            print("\nPossible reasons:")
            print("  1. NVIDIA GPU not available")
            print("  2. NVIDIA driver not installed")
            print("  3. PyTorch CPU-only version installed")
            print("  4. CUDA version mismatch")
            
            print("\n🔧 Installation Guide:")
            print("Visit https://pytorch.org/get-started/locally/")
            print("Select your platform and CUDA version")
            
    except ImportError:
        print("❌ PyTorch not installed!")
        
    print("\n")

def gpu_memory_management():
    """Demonstrate GPU memory management."""
    print("💾 GPU Memory Management")
    print("=" * 50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Memory management demo skipped.")
            return
        
        # Memory before allocation
        print("Memory status before allocation:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Create some tensors
        print("\n📦 Creating large tensors...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        print("Memory after tensor creation:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Perform operations
        z = torch.matmul(x, y)
        print(f"  Result shape: {z.shape}")
        
        print("Memory after computation:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
        print("\nMemory after cleanup:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        print("\n💡 Memory Management Tips:")
        print("  1. Use del to remove references")
        print("  2. Call torch.cuda.empty_cache() to free unused memory")
        print("  3. Use torch.no_grad() for inference")
        print("  4. Move tensors to CPU when not needed on GPU")
        
    except Exception as e:
        print(f"❌ Error in memory management demo: {e}")
    
    print("\n")

def device_management_demo():
    """Demonstrate device management best practices."""
    print("📱 Device Management Best Practices")
    print("=" * 50)
    
    try:
        import torch
        
        # Device detection
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("✅ Using CPU")
        
        print(f"Selected device: {device}")
        
        # Creating tensors on device
        print("\n📦 Creating tensors on device:")
        
        # Method 1: Direct creation
        x1 = torch.randn(3, 3, device=device)
        print(f"Method 1 - Direct creation: {x1.device}")
        
        # Method 2: Create then move
        x2 = torch.randn(3, 3).to(device)
        print(f"Method 2 - Create then move: {x2.device}")
        
        # Method 3: Using cuda() method (if available)
        x3 = torch.randn(3, 3)
        if torch.cuda.is_available():
            x3 = x3.cuda()
            print(f"Method 3 - Using cuda(): {x3.device}")
        else:
            print("Method 3 - cuda() not available (CPU only)")
        
        # Device-agnostic code example
        print("\n💻 Device-agnostic code pattern:")
        print("""
def create_model_and_data(device):
    model = MyModel().to(device)
    data = torch.randn(32, 10).to(device)
    return model, data

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = create_model_and_data(device)
        """)
        
        # Multi-GPU setup (if available)
        if torch.cuda.device_count() > 1:
            print(f"\n🔗 Multi-GPU Setup ({torch.cuda.device_count()} GPUs available)")
            print("Example for using multiple GPUs:")
            print("  model = nn.DataParallel(model)")
            print("  # This will automatically use all available GPUs")
        
    except Exception as e:
        print(f"❌ Error in device management demo: {e}")
    
    print("\n")

def performance_optimization_tips():
    """Provide GPU performance optimization tips."""
    print("⚡ GPU Performance Optimization")
    print("=" * 50)
    
    tips = [
        {
            'title': 'Use Appropriate Data Types',
            'description': 'Use float16 for inference, float32 for training',
            'example': 'model.half()  # Convert to float16'
        },
        {
            'title': 'Batch Operations',
            'description': 'Process data in batches to maximize GPU utilization',
            'example': 'DataLoader(dataset, batch_size=32, shuffle=True)'
        },
        {
            'title': 'Avoid CPU-GPU Transfers',
            'description': 'Keep data on GPU as much as possible',
            'example': 'Create tensors directly on GPU: torch.randn(10, device="cuda")'
        },
        {
            'title': 'Use torch.no_grad() for Inference',
            'description': 'Disable gradient computation when not needed',
            'example': 'with torch.no_grad(): output = model(input)'
        },
        {
            'title': 'Pin Memory for DataLoader',
            'description': 'Faster CPU to GPU transfer',
            'example': 'DataLoader(dataset, pin_memory=True)'
        },
        {
            'title': 'Use Asynchronous Operations',
            'description': 'Overlap computation and data transfer',
            'example': 'data = data.to(device, non_blocking=True)'
        },
        {
            'title': 'Mixed Precision Training',
            'description': 'Use automatic mixed precision for faster training',
            'example': 'from torch.cuda.amp import autocast, GradScaler'
        },
        {
            'title': 'Optimize Memory Usage',
            'description': 'Use gradient checkpointing for large models',
            'example': 'torch.utils.checkpoint.checkpoint()'
        }
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip['title']}")
        print(f"   📝 {tip['description']}")
        print(f"   💻 {tip['example']}")
        print()
    
    print("🎯 Performance Monitoring:")
    print("  • Use nvidia-smi to monitor GPU usage")
    print("  • Use torch.profiler for detailed profiling")
    print("  • Monitor memory usage with torch.cuda.memory_summary()")
    print("\n")

def troubleshooting_guide():
    """Common GPU issues and solutions."""
    print("🔧 Common GPU Issues & Solutions")
    print("=" * 50)
    
    issues = [
        {
            'problem': 'CUDA out of memory',
            'solutions': [
                'Reduce batch size',
                'Use gradient accumulation',
                'Clear unused variables with del',
                'Call torch.cuda.empty_cache()',
                'Use mixed precision training'
            ]
        },
        {
            'problem': 'RuntimeError: CUDA error: device-side assert triggered',
            'solutions': [
                'Check for invalid indices in loss functions',
                'Verify data labels are in correct range',
                'Use CUDA_LAUNCH_BLOCKING=1 for detailed error'
            ]
        },
        {
            'problem': 'Slow training despite GPU availability',
            'solutions': [
                'Increase batch size to utilize GPU better',
                'Check if data loading is the bottleneck',
                'Use pin_memory=True in DataLoader',
                'Ensure all operations are on GPU'
            ]
        },
        {
            'problem': 'CUDA version mismatch',
            'solutions': [
                'Check CUDA version with nvidia-smi',
                'Install PyTorch version compatible with CUDA',
                'Reinstall PyTorch with correct CUDA version'
            ]
        }
    ]
    
    for issue in issues:
        print(f"❌ Problem: {issue['problem']}")
        print("   Solutions:")
        for solution in issue['solutions']:
            print(f"   • {solution}")
        print()
    
    print("🆘 Getting Help:")
    print("  • PyTorch Forums: https://discuss.pytorch.org/")
    print("  • GitHub Issues: https://github.com/pytorch/pytorch/issues")
    print("  • Stack Overflow: Use 'pytorch' tag")
    print("\n")

def main():
    """Main function to run all GPU setup checks."""
    print("🚀 GPU Setup and Configuration Guide")
    print("=" * 60)
    print("Let's set up and optimize your GPU for PyTorch!\n")
    
    check_nvidia_setup()
    check_pytorch_cuda()
    gpu_memory_management()
    device_management_demo()
    performance_optimization_tips()
    troubleshooting_guide()
    
    print("🎯 Key Takeaways:")
    print("  1. Always check CUDA availability before using GPU")
    print("  2. Write device-agnostic code for portability")
    print("  3. Manage GPU memory carefully to avoid OOM errors")
    print("  4. Use performance optimization techniques")
    print("  5. Know how to troubleshoot common GPU issues")
    
    print("\n🎉 Congratulations! You've completed Chapter 1!")
    print("🔜 Next: Deep dive into tensors in Chapter 2!")

if __name__ == "__main__":
    main()
