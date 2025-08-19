# Chapter 2: Tensors Deep Dive

Welcome to the heart of PyTorch - tensors! Tensors are the fundamental building blocks of PyTorch and understanding them is crucial for everything that follows.

## 🎯 Learning Objectives

By the end of this chapter, you will:
- ✅ Master tensor creation, manipulation, and operations
- ✅ Understand tensor shapes, dimensions, and broadcasting
- ✅ Know how to efficiently move data between CPU and GPU
- ✅ Perform mathematical operations on tensors
- ✅ Handle different data types and memory layouts

## 📚 Chapter Contents

1. **[Tensor Basics](./01_tensor_basics.py)** - Creation, types, and properties
2. **[Tensor Operations](./02_tensor_operations.py)** - Mathematical operations and broadcasting
3. **[Indexing and Slicing](./03_indexing_slicing.py)** - Accessing and modifying tensor data
4. **[Tensor Shapes](./04_tensor_shapes.py)** - Reshaping, transposing, and dimension manipulation
5. **[Advanced Tensors](./05_advanced_tensors.py)** - Memory views, sparse tensors, and optimization

## 🚀 Quick Start

Start with tensor basics:

```bash
python 01_tensor_basics.py
```

## 📖 Theory Overview

### What are Tensors?

Tensors are multi-dimensional arrays that can hold numerical data. They are similar to NumPy arrays but with additional capabilities:

- **GPU Acceleration**: Can be moved to GPU for faster computation
- **Automatic Differentiation**: Can track gradients for machine learning
- **Broadcasting**: Automatic handling of operations between different shapes
- **Memory Efficiency**: Views and in-place operations for optimization

### Tensor Hierarchy

```
Scalar (0D) → [5]
Vector (1D) → [1, 2, 3, 4]
Matrix (2D) → [[1, 2], [3, 4]]
Tensor (3D+) → [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

### Key Concepts

1. **Shape**: The size of each dimension (e.g., [2, 3, 4])
2. **Dtype**: Data type (float32, int64, bool, etc.)
3. **Device**: Where the tensor lives (CPU or GPU)
4. **Requires_grad**: Whether to track gradients for this tensor

## 🧮 Mathematical Operations

PyTorch supports all standard mathematical operations:

| Category | Operations |
|----------|------------|
| **Arithmetic** | +, -, *, /, %, ** |
| **Comparison** | ==, !=, <, >, <=, >= |
| **Logical** | &, \|, ~, ^^ |
| **Statistical** | mean, std, sum, max, min |
| **Linear Algebra** | matmul, dot, cross, svd |
| **Trigonometric** | sin, cos, tan, asin, acos |

## 📊 Broadcasting Rules

PyTorch follows NumPy's broadcasting rules:

1. Start from the rightmost dimension
2. Dimensions of size 1 can be "stretched"
3. Missing dimensions are assumed to be 1

```python
a = torch.randn(3, 1, 5)  # Shape: [3, 1, 5]
b = torch.randn(1, 4, 1)  # Shape: [1, 4, 1]
c = a + b                 # Result: [3, 4, 5]
```

## 💾 Memory Management

Understanding memory layout is crucial for performance:

- **Contiguous**: Elements are stored in row-major order
- **Stride**: Number of elements to skip to get to the next element in each dimension
- **View vs Copy**: Views share memory, copies create new memory

## 🔍 Common Pitfalls

1. **Shape Mismatches**: Always check tensor shapes before operations
2. **Device Mismatches**: Ensure all tensors are on the same device
3. **Data Type Issues**: Be aware of automatic type promotion
4. **Memory Leaks**: Use views when possible, delete large tensors

## 📝 Exercises

Each script in this chapter includes hands-on exercises to reinforce learning:

1. **Tensor Creation**: Practice creating tensors with different methods
2. **Shape Manipulation**: Master reshaping and dimension operations
3. **Broadcasting**: Understand how operations work with different shapes
4. **Performance**: Compare different approaches for efficiency

## 🔗 Next Chapter

After mastering tensors, move on to [Chapter 3: Autograd & Gradients](../03_autograd/README.md)

## 📚 Additional Resources

- [PyTorch Tensor Documentation](https://pytorch.org/docs/stable/tensors.html)
- [Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [Tensor Attributes](https://pytorch.org/docs/stable/tensor_attributes.html)
