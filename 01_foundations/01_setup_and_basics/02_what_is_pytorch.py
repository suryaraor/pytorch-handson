"""
Chapter 1.2: What is PyTorch?
============================

A comprehensive introduction to PyTorch, its philosophy, and
how it differs from other deep learning frameworks.

Author: PyTorch Deep Dive Tutorial
Date: 2025
"""

import torch
import numpy as np
import time

def pytorch_philosophy():
    """Explain PyTorch's core philosophy and design principles."""
    print("🔥 PyTorch Philosophy & Design")
    print("=" * 50)
    
    print("""
PyTorch is built on several key principles:

1. 🐍 PYTHONIC: Feels natural to Python developers
   - Intuitive debugging with standard Python tools
   - Dynamic computation graphs
   - Object-oriented design

2. 🔬 RESEARCH-FIRST: Designed for experimentation
   - Easy prototyping and iteration
   - Flexible architecture
   - Popular in academic research

3. ⚡ PERFORMANCE: Optimized for speed
   - Efficient tensor operations
   - GPU acceleration
   - JIT compilation support

4. 🏭 PRODUCTION-READY: Not just for research
   - TorchServe for model serving
   - TorchScript for deployment
   - Mobile and embedded support
    """)
    
    print("\n")

def dynamic_vs_static_graphs():
    """Demonstrate dynamic computation graphs vs static graphs."""
    print("📊 Dynamic vs Static Computation Graphs")
    print("=" * 50)
    
    print("PyTorch uses DYNAMIC computation graphs:")
    print("- Graph is built on-the-fly during execution")
    print("- Can change structure based on data/conditions")
    print("- Easy debugging and introspection")
    print()
    
    # Example of dynamic computation
    def dynamic_network(x, use_dropout=True):
        """Example of dynamic neural network behavior."""
        h = torch.relu(x)
        
        # Dynamic decision based on input
        if use_dropout and x.mean() > 0.5:
            h = torch.dropout(h, p=0.5, training=True)
            print(f"  Applied dropout (input mean: {x.mean():.3f})")
        else:
            print(f"  No dropout (input mean: {x.mean():.3f})")
        
        return h
    
    print("Example: Dynamic network behavior")
    
    # Test with different inputs
    x1 = torch.randn(5, 10) * 0.3  # Low mean
    x2 = torch.randn(5, 10) * 2.0  # High mean
    
    print("\nInput 1 (low values):")
    dynamic_network(x1)
    
    print("\nInput 2 (high values):")
    dynamic_network(x2)
    
    print("\n✨ This flexibility is PyTorch's superpower!")
    print("\n")

def tensor_vs_numpy():
    """Compare PyTorch tensors with NumPy arrays."""
    print("🔢 PyTorch Tensors vs NumPy Arrays")
    print("=" * 50)
    
    # Create similar data structures
    np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    
    print("NumPy Array:")
    print(f"  Data: {np_array}")
    print(f"  Type: {type(np_array)}")
    print(f"  Shape: {np_array.shape}")
    print(f"  Dtype: {np_array.dtype}")
    
    print("\nPyTorch Tensor:")
    print(f"  Data: {torch_tensor}")
    print(f"  Type: {type(torch_tensor)}")
    print(f"  Shape: {torch_tensor.shape}")
    print(f"  Dtype: {torch_tensor.dtype}")
    print(f"  Device: {torch_tensor.device}")
    print(f"  Requires Grad: {torch_tensor.requires_grad}")
    
    print("\n🔄 Easy Conversion:")
    # NumPy to PyTorch
    tensor_from_numpy = torch.from_numpy(np_array)
    print(f"  From NumPy: {tensor_from_numpy}")
    
    # PyTorch to NumPy
    numpy_from_tensor = torch_tensor.numpy()
    print(f"  To NumPy: {numpy_from_tensor}")
    
    print("\n⚡ Key Differences:")
    print("  1. Tensors can live on GPU")
    print("  2. Tensors support automatic differentiation")
    print("  3. Tensors integrate with PyTorch ecosystem")
    print("\n")

def autograd_preview():
    """Give a preview of PyTorch's automatic differentiation."""
    print("🎯 Automatic Differentiation Preview")
    print("=" * 50)
    
    print("PyTorch can automatically compute gradients!")
    
    # Create a tensor that requires gradients
    x = torch.tensor(2.0, requires_grad=True)
    print(f"Input: x = {x}")
    
    # Define a function: y = x^2 + 3x + 1
    y = x**2 + 3*x + 1
    print(f"Function: y = x² + 3x + 1 = {y}")
    
    # Compute gradients
    y.backward()
    print(f"Gradient dy/dx = 2x + 3 = {x.grad}")
    
    print("\n📚 Mathematical Verification:")
    print("  If y = x² + 3x + 1, then dy/dx = 2x + 3")
    print("  At x = 2: dy/dx = 2(2) + 3 = 7")
    print(f"  PyTorch computed: {x.grad.item()}")
    print("  ✅ Perfect match!")
    
    print("\n🚀 This is the foundation of deep learning!")
    print("   Neural networks use gradients to learn from data.")
    print("\n")

def gpu_demonstration():
    """Demonstrate GPU capabilities if available."""
    print("🎮 GPU Acceleration Demonstration")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Running on CPU only.")
        print("   This is fine for learning, but GPU speeds up training significantly.")
        print("\n")
        return
    
    print(f"✅ CUDA available! Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Performance comparison
    size = 10000
    
    # CPU operations
    start_time = time.time()
    cpu_tensor1 = torch.randn(size, size)
    cpu_tensor2 = torch.randn(size, size)
    torch.matmul(cpu_tensor1, cpu_tensor2)
    cpu_time = time.time() - start_time
    
    # GPU operations
    start_time = time.time()
    gpu_tensor1 = torch.randn(size, size, device='cuda')
    gpu_tensor2 = torch.randn(size, size, device='cuda')
    torch.matmul(gpu_tensor1, gpu_tensor2)
    torch.cuda.synchronize()  # Wait for GPU operations to complete
    gpu_time = time.time() - start_time
    
    print(f"\nMatrix Multiplication ({size}x{size}):")
    print(f"  CPU Time: {cpu_time:.4f} seconds")
    print(f"  GPU Time: {gpu_time:.4f} seconds")
    print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    
    print("\n🚀 GPU tensors are easy to use:")
    print("  1. Create on GPU: torch.tensor(..., device='cuda')")
    print("  2. Move to GPU: tensor.cuda()")
    print("  3. Move to CPU: tensor.cpu()")
    print("\n")

def pytorch_ecosystem_overview():
    """Overview of the PyTorch ecosystem."""
    print("🌐 PyTorch Ecosystem")
    print("=" * 50)
    
    ecosystem = {
        "Core PyTorch": {
            "torch": "Core tensor library and neural network building blocks",
            "torch.nn": "Neural network modules and layers",
            "torch.optim": "Optimization algorithms",
            "torch.utils.data": "Data loading and preprocessing utilities"
        },
        "Domain Libraries": {
            "torchvision": "Computer vision: datasets, models, transforms",
            "torchaudio": "Audio processing and models",
            "torchtext": "Natural language processing utilities",
            "torchgeo": "Geospatial data and satellite imagery"
        },
        "Deployment & Production": {
            "TorchScript": "Model compilation and optimization",
            "TorchServe": "Model serving and deployment",
            "PyTorch Mobile": "Mobile and embedded deployment",
            "PyTorch Lightning": "High-level training framework"
        },
        "Specialized Tools": {
            "Captum": "Model interpretability and explainability",
            "Optuna": "Hyperparameter optimization",
            "Ray": "Distributed training and hyperparameter tuning",
            "Weights & Biases": "Experiment tracking and visualization"
        }
    }
    
    for category, tools in ecosystem.items():
        print(f"\n📦 {category}:")
        for tool, description in tools.items():
            print(f"  • {tool}: {description}")
    
    print("\n")

def main():
    """Main function to run all demonstrations."""
    print("🚀 Understanding PyTorch")
    print("=" * 60)
    print("Let's explore what makes PyTorch special!\n")
    
    pytorch_philosophy()
    dynamic_vs_static_graphs()
    tensor_vs_numpy()
    autograd_preview()
    gpu_demonstration()
    pytorch_ecosystem_overview()
    
    print("🎯 Key Takeaways:")
    print("  1. PyTorch is dynamic, flexible, and Pythonic")
    print("  2. Tensors are like NumPy arrays but with superpowers")
    print("  3. Automatic differentiation makes learning possible")
    print("  4. Rich ecosystem supports every ML/DL task")
    print("  5. Research-friendly but production-ready")
    
    print("\n🔜 Next: Let's dive deeper into tensors in Chapter 2!")

if __name__ == "__main__":
    main()
