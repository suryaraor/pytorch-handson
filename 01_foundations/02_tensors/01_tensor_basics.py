"""
Chapter 2.1: Tensor Basics
==========================

Master the fundamentals of PyTorch tensors - creation, properties, and basic operations.

Author: PyTorch Deep Dive Tutorial
Date: 2025
"""

def tensor_creation_methods():
    """Explore different ways to create tensors."""
    print("🔥 Tensor Creation Methods")
    print("=" * 50)
    
    try:
        import torch
        import numpy as np
        
        print("1. Creating tensors from data:")
        
        # From Python lists
        data = [[1, 2], [3, 4]]
        tensor_from_list = torch.tensor(data)
        print(f"   From list: {tensor_from_list}")
        print(f"   Shape: {tensor_from_list.shape}")
        print(f"   Dtype: {tensor_from_list.dtype}")
        
        # From NumPy arrays
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor_from_numpy = torch.from_numpy(np_array)
        print(f"   From NumPy: {tensor_from_numpy}")
        print(f"   Dtype: {tensor_from_numpy.dtype}")
        
        print("\n2. Creating tensors with specific values:")
        
        # Zeros and ones
        zeros = torch.zeros(2, 3)
        ones = torch.ones(2, 3)
        print(f"   Zeros (2x3): \n{zeros}")
        print(f"   Ones (2x3): \n{ones}")
        
        # Identity matrix
        identity = torch.eye(3)
        print(f"   Identity (3x3): \n{identity}")
        
        # Filled with specific value
        filled = torch.full((2, 2), 7.5)
        print(f"   Filled with 7.5: \n{filled}")
        
        print("\n3. Creating tensors with random values:")
        
        # Random from uniform distribution [0, 1)
        rand_uniform = torch.rand(2, 3)
        print(f"   Random uniform [0,1): \n{rand_uniform}")
        
        # Random from standard normal distribution
        rand_normal = torch.randn(2, 3)
        print(f"   Random normal: \n{rand_normal}")
        
        # Random integers
        rand_int = torch.randint(0, 10, (2, 3))
        print(f"   Random integers [0,10): \n{rand_int}")
        
        print("\n4. Creating tensors like other tensors:")
        
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        
        # Same shape, filled with zeros
        zeros_like = torch.zeros_like(x)
        print(f"   Zeros like x: \n{zeros_like}")
        
        # Same shape, filled with ones
        ones_like = torch.ones_like(x)
        print(f"   Ones like x: \n{ones_like}")
        
        # Same shape, random values
        rand_like = torch.randn_like(x)
        print(f"   Random like x: \n{rand_like}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def tensor_properties():
    """Explore tensor properties and attributes."""
    print("📊 Tensor Properties")
    print("=" * 50)
    
    try:
        import torch
        
        # Create a sample tensor
        x = torch.randn(2, 3, 4)
        
        print("Sample tensor:")
        print(f"   Data: \n{x}")
        print(f"   Shape: {x.shape}")
        print(f"   Size: {x.size()}")  # Alternative to shape
        print(f"   Number of dimensions: {x.ndim}")
        print(f"   Total elements: {x.numel()}")
        print(f"   Data type: {x.dtype}")
        print(f"   Device: {x.device}")
        print(f"   Requires gradient: {x.requires_grad}")
        print(f"   Memory layout: {x.layout}")
        print(f"   Is contiguous: {x.is_contiguous()}")
        
        print("\n🔍 Detailed analysis:")
        print(f"   Stride: {x.stride()}")
        print(f"   Storage offset: {x.storage_offset()}")
        print(f"   Storage size: {x.storage().size()}")
        
        # Check specific dimensions
        print("\n📏 Dimension analysis:")
        for i in range(x.ndim):
            print(f"   Dimension {i}: size = {x.size(i)}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def tensor_data_types():
    """Explore different tensor data types."""
    print("🏷️ Tensor Data Types")
    print("=" * 50)
    
    try:
        import torch
        
        print("Common data types:")
        
        # Integer types
        int8_tensor = torch.tensor([1, 2, 3], dtype=torch.int8)
        int32_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        int64_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        
        print(f"   int8: {int8_tensor} (range: -128 to 127)")
        print(f"   int32: {int32_tensor} (range: ~-2B to 2B)")
        print(f"   int64: {int64_tensor} (default for integers)")
        
        # Float types
        float16_tensor = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float16)
        float32_tensor = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)
        float64_tensor = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float64)
        
        print(f"   float16: {float16_tensor} (half precision)")
        print(f"   float32: {float32_tensor} (default for floats)")
        print(f"   float64: {float64_tensor} (double precision)")
        
        # Boolean type
        bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
        print(f"   bool: {bool_tensor}")
        
        # Complex types
        complex64_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
        print(f"   complex64: {complex64_tensor}")
        
        print("\n🔄 Type conversion:")
        
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        print(f"   Original: {x} ({x.dtype})")
        
        # Convert to float
        x_float = x.float()
        print(f"   To float: {x_float} ({x_float.dtype})")
        
        # Convert to double
        x_double = x.double()
        print(f"   To double: {x_double} ({x_double.dtype})")
        
        # Convert to bool
        x_bool = x.bool()
        print(f"   To bool: {x_bool} ({x_bool.dtype})")
        
        print("\n💡 Type promotion:")
        
        # Automatic type promotion
        a = torch.tensor([1, 2], dtype=torch.int32)
        b = torch.tensor([1.5, 2.5], dtype=torch.float32)
        c = a + b
        print(f"   int32 + float32 = {c} ({c.dtype})")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def device_management():
    """Demonstrate tensor device management."""
    print("📱 Device Management")
    print("=" * 50)
    
    try:
        import torch
        
        # Create tensor on CPU
        cpu_tensor = torch.randn(2, 3)
        print(f"CPU tensor: {cpu_tensor.device}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} device(s)")
            
            # Create tensor directly on GPU
            gpu_tensor = torch.randn(2, 3, device='cuda')
            print(f"GPU tensor: {gpu_tensor.device}")
            
            # Move CPU tensor to GPU
            cpu_to_gpu = cpu_tensor.cuda()
            print(f"CPU to GPU: {cpu_to_gpu.device}")
            
            # Move GPU tensor to CPU
            gpu_to_cpu = gpu_tensor.cpu()
            print(f"GPU to CPU: {gpu_to_cpu.device}")
            
            # Using .to() method
            tensor_to_gpu = cpu_tensor.to('cuda')
            print(f"Using .to('cuda'): {tensor_to_gpu.device}")
            
            # Device-agnostic code
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            adaptive_tensor = torch.randn(2, 3, device=device)
            print(f"Adaptive tensor: {adaptive_tensor.device}")
            
        else:
            print("❌ CUDA not available")
            device = torch.device('cpu')
            adaptive_tensor = torch.randn(2, 3, device=device)
            print(f"Using CPU: {adaptive_tensor.device}")
        
        print("\n💡 Best practices:")
        print("   1. Create tensors directly on target device when possible")
        print("   2. Use device-agnostic code for portability")
        print("   3. Move tensors only when necessary (expensive operation)")
        print("   4. Keep related tensors on the same device")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def tensor_gradients_intro():
    """Introduction to gradient tracking."""
    print("🎯 Gradient Tracking Introduction")
    print("=" * 50)
    
    try:
        import torch
        
        print("Tensors can track gradients for automatic differentiation:")
        
        # Tensor without gradients (default)
        x = torch.randn(2, 2)
        print(f"   Default tensor: requires_grad = {x.requires_grad}")
        
        # Tensor with gradients
        x_grad = torch.randn(2, 2, requires_grad=True)
        print(f"   Gradient tensor: requires_grad = {x_grad.requires_grad}")
        
        # Enable gradients on existing tensor
        x.requires_grad_(True)
        print(f"   After enabling: requires_grad = {x.requires_grad}")
        
        print("\n🔄 Gradient flow example:")
        
        # Create inputs with gradients
        a = torch.tensor([2.0], requires_grad=True)
        b = torch.tensor([3.0], requires_grad=True)
        
        # Forward pass
        c = a * b
        d = c + a
        
        print(f"   a = {a.item()}")
        print(f"   b = {b.item()}")
        print(f"   c = a * b = {c.item()}")
        print(f"   d = c + a = {d.item()}")
        
        # Backward pass
        d.backward()
        
        print("\n   Gradients:")
        print(f"   ∂d/∂a = {a.grad.item()}")
        print(f"   ∂d/∂b = {b.grad.item()}")
        
        print("\n💡 Note: Gradient computation will be covered in detail in Chapter 3")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def basic_tensor_operations():
    """Demonstrate basic tensor operations."""
    print("🧮 Basic Tensor Operations")
    print("=" * 50)
    
    try:
        import torch
        
        # Create sample tensors
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        
        print(f"Tensor a:\n{a}")
        print(f"Tensor b:\n{b}")
        
        print("\n➕ Arithmetic operations:")
        print(f"   Addition: a + b =\n{a + b}")
        print(f"   Subtraction: a - b =\n{a - b}")
        print(f"   Multiplication: a * b =\n{a * b}")
        print(f"   Division: a / b =\n{a / b}")
        print(f"   Power: a ** 2 =\n{a ** 2}")
        
        print("\n📊 Statistical operations:")
        print(f"   Sum of a: {torch.sum(a)}")
        print(f"   Mean of a: {torch.mean(a)}")
        print(f"   Max of a: {torch.max(a)}")
        print(f"   Min of a: {torch.min(a)}")
        print(f"   Standard deviation: {torch.std(a, dim=None)}")
        
        print("\n🔗 Matrix operations:")
        print(f"   Matrix multiplication: a @ b =\n{a @ b}")
        print(f"   Transpose of a:\n{a.t()}")
        
        print("\n🔍 Comparison operations:")
        print(f"   a > 2:\n{a > 2}")
        print(f"   a == b:\n{a == b}")
        
        # In-place operations
        print("\n🔄 In-place operations:")
        c = a.clone()  # Make a copy
        print(f"   Before: c =\n{c}")
        c.add_(1)  # In-place addition
        print(f"   After c.add_(1):\n{c}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def exercise_tensor_basics():
    """Hands-on exercises for tensor basics."""
    print("💪 Hands-on Exercises")
    print("=" * 50)
    
    try:
        import torch
        
        print("Exercise 1: Create tensors with different methods")
        print("-" * 45)
        
        # Your code here - try creating tensors using different methods
        print("✅ Create a 3x3 tensor filled with 5:")
        exercise1 = torch.full((3, 3), 5)
        print(exercise1)
        
        print("\n✅ Create a 2x4 tensor with random values from normal distribution:")
        exercise2 = torch.randn(2, 4)
        print(exercise2)
        
        print("\nExercise 2: Tensor properties exploration")
        print("-" * 45)
        
        x = torch.randint(0, 10, (3, 4, 5))
        print(f"✅ Created tensor with shape {x.shape}")
        print(f"   Number of dimensions: {x.ndim}")
        print(f"   Total elements: {x.numel()}")
        print(f"   Data type: {x.dtype}")
        
        print("\nExercise 3: Type conversions")
        print("-" * 45)
        
        int_tensor = torch.tensor([1, 2, 3, 4])
        print(f"✅ Original: {int_tensor} ({int_tensor.dtype})")
        
        float_tensor = int_tensor.float()
        print(f"   Converted to float: {float_tensor} ({float_tensor.dtype})")
        
        print("\nExercise 4: Basic operations")
        print("-" * 45)
        
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        
        print(f"✅ a = {a}")
        print(f"   b = {b}")
        print(f"   a + b = {a + b}")
        print(f"   a * b = {a * b}")
        print(f"   Sum of a: {torch.sum(a)}")
        
        print("\n🎯 Challenge: Create a function that takes a tensor and returns")
        print("   its mean, standard deviation, and maximum value")
        
        def tensor_stats(tensor):
            """Calculate basic statistics of a tensor."""
            return {
                'mean': torch.mean(tensor.float()),
                'std': torch.std(tensor.float(), dim=None),
                'max': torch.max(tensor)
            }
        
        test_tensor = torch.randn(5, 5)
        stats = tensor_stats(test_tensor)
        print("✅ Stats for random 5x5 tensor:")
        print(f"   Mean: {stats['mean']:.4f}")
        print(f"   Std: {stats['std']:.4f}")
        print(f"   Max: {stats['max']:.4f}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def main():
    """Main function to run all tensor basics demonstrations."""
    print("🚀 PyTorch Tensor Basics")
    print("=" * 60)
    print("Let's master the fundamentals of PyTorch tensors!\n")
    
    tensor_creation_methods()
    tensor_properties()
    tensor_data_types()
    device_management()
    tensor_gradients_intro()
    basic_tensor_operations()
    exercise_tensor_basics()
    
    print("🎯 Key Takeaways:")
    print("  1. Tensors are multi-dimensional arrays with special capabilities")
    print("  2. Many ways to create tensors - choose the right method for your use case")
    print("  3. Pay attention to data types, shapes, and devices")
    print("  4. Gradient tracking enables automatic differentiation")
    print("  5. In-place operations save memory but modify original tensors")
    
    print("\n🔜 Next: Advanced tensor operations and broadcasting!")

if __name__ == "__main__":
    main()
