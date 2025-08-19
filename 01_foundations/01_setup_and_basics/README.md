# Chapter 1: Environment Setup & PyTorch Basics

Welcome to your PyTorch journey! In this chapter, you'll set up your development environment and learn the absolute basics of PyTorch.

## 🎯 Learning Objectives

By the end of this chapter, you will:
- ✅ Have PyTorch properly installed and configured
- ✅ Understand what PyTorch is and why it's powerful
- ✅ Know how to check your installation and GPU availability
- ✅ Understand the PyTorch ecosystem and key components

## 📚 Chapter Contents

1. **[Installation Check](./01_installation_check.py)** - Verify your PyTorch installation
2. **[What is PyTorch?](./02_what_is_pytorch.py)** - Understanding the framework
3. **[PyTorch Ecosystem](./03_pytorch_ecosystem.py)** - Key libraries and tools
4. **[GPU Setup](./04_gpu_setup.py)** - CUDA configuration and testing

## 🚀 Quick Start

Run the installation check to verify everything is working:

```bash
python 01_installation_check.py
```

## 📖 Theory Overview

### What is PyTorch?

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides:

1. **Dynamic Computation Graphs**: Build and modify neural networks on-the-fly
2. **Automatic Differentiation**: Automatic gradient computation for backpropagation
3. **GPU Acceleration**: Seamless CPU to GPU tensor operations
4. **Pythonic Interface**: Natural Python syntax and debugging capabilities

### Key Components

- **Tensors**: Multi-dimensional arrays (like NumPy, but with GPU support)
- **Autograd**: Automatic differentiation engine
- **nn.Module**: Building blocks for neural networks
- **optim**: Optimization algorithms (SGD, Adam, etc.)
- **DataLoader**: Efficient data loading and batching

### PyTorch vs Other Frameworks

| Feature | PyTorch | TensorFlow | JAX |
|---------|---------|------------|-----|
| Learning Curve | Easy | Moderate | Hard |
| Dynamic Graphs | ✅ | ✅ (2.0+) | ✅ |
| Production Ready | ✅ | ✅ | Moderate |
| Research Friendly | ✅ | Moderate | ✅ |
| Community | Large | Large | Growing |

## 🔍 Common Issues & Solutions

### Installation Problems
- **CUDA Version Mismatch**: Check `nvidia-smi` and install compatible PyTorch
- **Import Errors**: Ensure virtual environment is activated
- **Memory Issues**: Start with CPU-only version for learning

### Performance Tips
- Always move tensors to GPU when available
- Use `torch.no_grad()` for inference
- Enable mixed precision for faster training

## 📝 Exercises

1. **Environment Check**: Run all scripts in this chapter
2. **Version Investigation**: Find your PyTorch, Python, and CUDA versions
3. **Performance Test**: Compare CPU vs GPU tensor operations (if GPU available)

## 🔗 Next Chapter

Once you've completed this chapter, move on to [Chapter 2: Tensors Deep Dive](../02_tensors/README.md)

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
