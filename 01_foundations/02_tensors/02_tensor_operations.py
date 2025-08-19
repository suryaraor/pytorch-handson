"""
Chapter 2.2: Tensor Operations
=============================

Explore mathematical operations, broadcasting, and advanced tensor manipulations.

Author: PyTorch Deep Dive Tutorial
Date: 2025
"""

def arithmetic_operations():
    """Demonstrate basic arithmetic operations on tensors."""
    print("➕ Arithmetic Operations")
    print("=" * 50)
    
    try:
        import torch
        
        # Create sample tensors
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        scalar = 2.0
        
        print(f"Tensor a:\n{a}")
        print(f"Tensor b:\n{b}")
        print(f"Scalar: {scalar}")
        
        print("\n1. Element-wise operations:")
        print(f"   Addition (a + b):\n{a + b}")
        print(f"   Subtraction (a - b):\n{a - b}")
        print(f"   Multiplication (a * b):\n{a * b}")
        print(f"   Division (a / b):\n{a / b}")
        print(f"   Power (a ** 2):\n{a ** 2}")
        print(f"   Square root:\n{torch.sqrt(a)}")
        
        print("\n2. Scalar operations:")
        print(f"   a + scalar:\n{a + scalar}")
        print(f"   a * scalar:\n{a * scalar}")
        print(f"   a / scalar:\n{a / scalar}")
        
        print("\n3. Mathematical functions:")
        print(f"   Exponential:\n{torch.exp(a)}")
        print(f"   Logarithm:\n{torch.log(a)}")
        print(f"   Sine:\n{torch.sin(a)}")
        print(f"   Cosine:\n{torch.cos(a)}")
        
        print("\n4. In-place vs out-of-place:")
        c = a.clone()
        print(f"   Original c:\n{c}")
        
        # Out-of-place (creates new tensor)
        d = c + 1
        print(f"   Out-of-place (c + 1):\n{d}")
        print(f"   c unchanged:\n{c}")
        
        # In-place (modifies original tensor)
        c.add_(1)
        print(f"   In-place (c.add_(1)):\n{c}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def broadcasting_rules():
    """Demonstrate PyTorch broadcasting rules."""
    print("📡 Broadcasting Rules")
    print("=" * 50)
    
    try:
        import torch
        
        print("Broadcasting allows operations between tensors of different shapes:")
        
        # Example 1: Vector + Scalar
        print("\n1. Vector + Scalar:")
        vector = torch.tensor([1, 2, 3])
        scalar = 10
        result1 = vector + scalar
        print(f"   {vector.shape} + scalar = {result1} (shape: {result1.shape})")
        
        # Example 2: Matrix + Vector
        print("\n2. Matrix + Vector:")
        matrix = torch.randn(3, 4)
        vector = torch.randn(4)
        result2 = matrix + vector
        print(f"   {matrix.shape} + {vector.shape} = {result2.shape}")
        print(f"   Matrix:\n{matrix}")
        print(f"   Vector: {vector}")
        print(f"   Result:\n{result2}")
        
        # Example 3: Different dimensions
        print("\n3. Broadcasting with different dimensions:")
        a = torch.randn(2, 1, 4)  # Shape: [2, 1, 4]
        b = torch.randn(3, 1)     # Shape: [3, 1]
        
        try:
            result3 = a + b
            print(f"   {a.shape} + {b.shape} = {result3.shape}")
        except RuntimeError as e:
            print(f"   Broadcasting failed: {e}")
        
        # Example 4: Compatible broadcasting
        print("\n4. Compatible broadcasting:")
        a = torch.randn(2, 1, 4)  # Shape: [2, 1, 4]
        b = torch.randn(1, 3, 1)  # Shape: [1, 3, 1]
        result4 = a + b
        print(f"   {a.shape} + {b.shape} = {result4.shape}")
        
        print("\n📋 Broadcasting Rules:")
        print("   1. Start from the rightmost dimension")
        print("   2. Dimensions must be equal, or one must be 1")
        print("   3. Missing dimensions are assumed to be 1")
        print("   4. Result shape is the maximum size in each dimension")
        
        # Broadcasting examples table
        print("\n📊 Broadcasting Examples:")
        examples = [
            ([3, 4], [4], [3, 4]),
            ([2, 3, 4], [4], [2, 3, 4]),
            ([2, 1, 4], [3, 1], [2, 3, 4]),
            ([1, 3, 1], [2, 1, 4], [2, 3, 4]),
        ]
        
        for shape1, shape2, result_shape in examples:
            print(f"   {shape1} + {shape2} → {result_shape}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def matrix_operations():
    """Demonstrate matrix and linear algebra operations."""
    print("🔗 Matrix Operations")
    print("=" * 50)
    
    try:
        import torch
        
        # Create matrices
        A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        vector = torch.tensor([1, 2], dtype=torch.float32)
        
        print(f"Matrix A:\n{A}")
        print(f"Matrix B:\n{B}")
        print(f"Vector: {vector}")
        
        print("\n1. Matrix multiplication:")
        # Matrix-matrix multiplication
        matmul_AB = torch.matmul(A, B)
        print(f"   A @ B (matmul):\n{matmul_AB}")
        
        # Alternative syntax
        matmul_AB_alt = A @ B
        print(f"   A @ B (@ operator):\n{matmul_AB_alt}")
        
        # Matrix-vector multiplication
        matvec = torch.matmul(A, vector)
        print(f"   A @ vector: {matvec}")
        
        print("\n2. Transpose operations:")
        print(f"   A.T (transpose):\n{A.T}")
        print(f"   A.t() (transpose):\n{A.t()}")
        
        # For higher dimensional tensors
        tensor_3d = torch.randn(2, 3, 4)
        print(f"   3D tensor shape: {tensor_3d.shape}")
        print(f"   transpose(0,1): {tensor_3d.transpose(0, 1).shape}")
        print(f"   transpose(1,2): {tensor_3d.transpose(1, 2).shape}")
        
        print("\n3. Advanced linear algebra:")
        
        # Determinant
        det_A = torch.det(A)
        print(f"   Determinant of A: {det_A}")
        
        # Inverse
        inv_A = torch.inverse(A)
        print(f"   Inverse of A:\n{inv_A}")
        
        # Verify A * A^(-1) = I
        identity_check = torch.matmul(A, inv_A)
        print(f"   A @ A^(-1) (should be identity):\n{identity_check}")
        
        # Eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eig(A)
        print(f"   Eigenvalues: {eigenvals}")
        print(f"   Eigenvectors:\n{eigenvecs}")
        
        # SVD (Singular Value Decomposition)
        U, S, V = torch.svd(A)
        print(f"   SVD - U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
        
        print("\n4. Batch operations:")
        # Batch matrix multiplication
        batch_A = torch.randn(3, 2, 2)  # 3 matrices of size 2x2
        batch_B = torch.randn(3, 2, 2)  # 3 matrices of size 2x2
        batch_result = torch.bmm(batch_A, batch_B)
        print(f"   Batch matmul: {batch_A.shape} @ {batch_B.shape} = {batch_result.shape}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def reduction_operations():
    """Demonstrate reduction operations (sum, mean, etc.)."""
    print("📊 Reduction Operations")
    print("=" * 50)
    
    try:
        import torch
        
        # Create a sample tensor
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        print(f"Sample tensor:\n{tensor}")
        print(f"Shape: {tensor.shape}")
        
        print("\n1. Basic reductions (reduce all dimensions):")
        print(f"   Sum: {torch.sum(tensor)}")
        print(f"   Mean: {torch.mean(tensor)}")
        print(f"   Max: {torch.max(tensor)}")
        print(f"   Min: {torch.min(tensor)}")
        print(f"   Standard deviation: {torch.std(tensor, dim=None)}")
        print(f"   Variance: {torch.var(tensor, dim=None)}")
        
        print("\n2. Reductions along specific dimensions:")
        
        # Along dimension 0 (rows)
        print(f"   Sum along dim 0 (columns): {torch.sum(tensor, dim=0)}")
        print(f"   Mean along dim 0: {torch.mean(tensor, dim=0)}")
        print(f"   Max along dim 0: {torch.max(tensor, dim=0)}")
        
        # Along dimension 1 (columns)
        print(f"   Sum along dim 1 (rows): {torch.sum(tensor, dim=1)}")
        print(f"   Mean along dim 1: {torch.mean(tensor, dim=1)}")
        print(f"   Max along dim 1: {torch.max(tensor, dim=1)}")
        
        print("\n3. Keeping dimensions:")
        sum_keepdim = torch.sum(tensor, dim=1, keepdim=True)
        print(f"   Sum dim 1, keepdim=True: {sum_keepdim} (shape: {sum_keepdim.shape})")
        
        sum_no_keepdim = torch.sum(tensor, dim=1, keepdim=False)
        print(f"   Sum dim 1, keepdim=False: {sum_no_keepdim} (shape: {sum_no_keepdim.shape})")
        
        print("\n4. Advanced reductions:")
        
        # Cumulative operations
        print(f"   Cumulative sum along dim 1:\n{torch.cumsum(tensor, dim=1)}")
        print(f"   Cumulative product along dim 0:\n{torch.cumprod(tensor, dim=0)}")
        
        # Norms
        print(f"   L2 norm: {torch.norm(tensor)}")
        print(f"   L1 norm: {torch.norm(tensor, p=1)}")
        print(f"   Frobenius norm: {torch.norm(tensor, p='fro')}")
        
        # Percentiles
        print(f"   Median: {torch.median(tensor)}")
        print(f"   75th percentile: {torch.quantile(tensor, 0.75)}")
        
        print("\n5. Boolean reductions:")
        boolean_tensor = tensor > 5
        print(f"   Boolean tensor (> 5):\n{boolean_tensor}")
        print(f"   Any: {torch.any(boolean_tensor)}")
        print(f"   All: {torch.all(boolean_tensor)}")
        print(f"   Count nonzero: {torch.count_nonzero(boolean_tensor)}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def comparison_operations():
    """Demonstrate comparison and logical operations."""
    print("🔍 Comparison Operations")
    print("=" * 50)
    
    try:
        import torch
        
        a = torch.tensor([1, 2, 3, 4, 5])
        b = torch.tensor([5, 4, 3, 2, 1])
        
        print(f"Tensor a: {a}")
        print(f"Tensor b: {b}")
        
        print("\n1. Element-wise comparisons:")
        print(f"   a == b: {a == b}")
        print(f"   a != b: {a != b}")
        print(f"   a < b:  {a < b}")
        print(f"   a <= b: {a <= b}")
        print(f"   a > b:  {a > b}")
        print(f"   a >= b: {a >= b}")
        
        print("\n2. Comparisons with scalars:")
        print(f"   a > 3: {a > 3}")
        print(f"   a == 3: {a == 3}")
        print(f"   a <= 2: {a <= 2}")
        
        print("\n3. Logical operations:")
        mask1 = a > 2
        mask2 = a < 5
        print(f"   mask1 (a > 2): {mask1}")
        print(f"   mask2 (a < 5): {mask2}")
        print(f"   mask1 & mask2: {mask1 & mask2}")
        print(f"   mask1 | mask2: {mask1 | mask2}")
        print(f"   ~mask1: {~mask1}")
        
        print("\n4. Using masks for indexing:")
        # Boolean indexing
        selected = a[a > 2]
        print(f"   Elements > 2: {selected}")
        
        # Conditional selection
        result = torch.where(a > 3, a, 0)
        print(f"   Replace elements <= 3 with 0: {result}")
        
        # Multiple conditions
        complex_mask = (a > 1) & (a < 5)
        selected_complex = a[complex_mask]
        print(f"   Elements between 1 and 5: {selected_complex}")
        
        print("\n5. Finding specific values:")
        # Find maximum and minimum values with indices
        values, indices = torch.max(a, dim=0)
        print(f"   Max value: {values}, at index: {indices}")
        
        values, indices = torch.min(a, dim=0)
        print(f"   Min value: {values}, at index: {indices}")
        
        # Find specific values
        eq_indices = torch.eq(a, 3).nonzero(as_tuple=True)[0]
        print(f"   Indices where a == 3: {eq_indices}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def advanced_operations():
    """Demonstrate advanced tensor operations."""
    print("🚀 Advanced Operations")
    print("=" * 50)
    
    try:
        import torch
        
        print("1. Concatenation and stacking:")
        
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        
        print(f"Tensor a:\n{a}")
        print(f"Tensor b:\n{b}")
        
        # Concatenation
        cat_dim0 = torch.cat([a, b], dim=0)
        print(f"   Concatenate dim 0:\n{cat_dim0}")
        
        cat_dim1 = torch.cat([a, b], dim=1)
        print(f"   Concatenate dim 1:\n{cat_dim1}")
        
        # Stacking (creates new dimension)
        stack_dim0 = torch.stack([a, b], dim=0)
        print(f"   Stack dim 0 (shape {stack_dim0.shape}):\n{stack_dim0}")
        
        stack_dim1 = torch.stack([a, b], dim=1)
        print(f"   Stack dim 1 (shape {stack_dim1.shape}):\n{stack_dim1}")
        
        print("\n2. Splitting and chunking:")
        
        large_tensor = torch.randn(6, 4)
        print(f"   Large tensor shape: {large_tensor.shape}")
        
        # Split into equal parts
        splits = torch.split(large_tensor, 2, dim=0)
        print(f"   Split into chunks of 2: {len(splits)} parts")
        for i, split in enumerate(splits):
            print(f"     Part {i}: {split.shape}")
        
        # Chunk into specific number of parts
        chunks = torch.chunk(large_tensor, 3, dim=0)
        print(f"   Chunk into 3 parts: {len(chunks)} parts")
        
        print("\n3. Permutation and sorting:")
        
        unsorted = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
        print(f"   Unsorted: {unsorted}")
        
        # Sort
        sorted_values, sorted_indices = torch.sort(unsorted)
        print(f"   Sorted values: {sorted_values}")
        print(f"   Sorted indices: {sorted_indices}")
        
        # Top-k values
        top_k_values, top_k_indices = torch.topk(unsorted, k=3)
        print(f"   Top 3 values: {top_k_values}")
        print(f"   Top 3 indices: {top_k_indices}")
        
        print("\n4. Unique operations:")
        
        tensor_with_duplicates = torch.tensor([1, 2, 2, 3, 3, 3, 4])
        print(f"   With duplicates: {tensor_with_duplicates}")
        
        unique_values = torch.unique(tensor_with_duplicates)
        print(f"   Unique values: {unique_values}")
        
        unique_values, counts = torch.unique(tensor_with_duplicates, return_counts=True)
        print(f"   Unique with counts: values={unique_values}, counts={counts}")
        
        print("\n5. Random operations:")
        
        # Random permutation
        perm = torch.randperm(5)
        print(f"   Random permutation of [0,1,2,3,4]: {perm}")
        
        # Random sampling
        tensor_to_sample = torch.randn(10)
        sampled_indices = torch.randint(0, 10, (3,))
        sampled_values = tensor_to_sample[sampled_indices]
        print(f"   Random sampling from tensor: {sampled_values}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def performance_tips():
    """Performance tips for tensor operations."""
    print("⚡ Performance Tips")
    print("=" * 50)
    
    print("1. 🔄 In-place operations save memory:")
    print("   ✅ tensor.add_(1)     # In-place")
    print("   ❌ tensor = tensor + 1 # Creates new tensor")
    
    print("\n2. 📦 Batch operations are faster:")
    print("   ✅ torch.matmul(batch_a, batch_b)  # Batch matrix multiply")
    print("   ❌ [torch.matmul(a, b) for a, b in zip(batch_a, batch_b)]")
    
    print("\n3. 🎯 Use appropriate data types:")
    print("   ✅ torch.float16 for inference (saves memory)")
    print("   ✅ torch.float32 for training (stability)")
    print("   ✅ torch.int64 for indices")
    
    print("\n4. 📊 Vectorize operations:")
    print("   ✅ torch.sum(tensor, dim=1)  # Vectorized")
    print("   ❌ [row.sum() for row in tensor]  # Loop-based")
    
    print("\n5. 🎮 Use GPU when available:")
    print("   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    print("   tensor = tensor.to(device)")
    
    print("\n6. 💾 Memory management:")
    print("   ✅ Use torch.no_grad() for inference")
    print("   ✅ del large_tensors when done")
    print("   ✅ torch.cuda.empty_cache() to free GPU memory")
    
    print("\n")

def exercises():
    """Hands-on exercises for tensor operations."""
    print("💪 Hands-on Exercises")
    print("=" * 50)
    
    try:
        import torch
        
        print("Exercise 1: Broadcasting practice")
        print("-" * 35)
        
        # Create tensors for broadcasting
        a = torch.randn(3, 1, 4)
        b = torch.randn(1, 2, 1)
        
        try:
            result = a + b
            print(f"✅ Broadcasting {a.shape} + {b.shape} = {result.shape}")
        except RuntimeError:
            print("❌ Broadcasting failed - incompatible shapes")
        
        print("\nExercise 2: Matrix operations")
        print("-" * 35)
        
        # Create two 3x3 matrices
        A = torch.randn(3, 3)
        B = torch.randn(3, 3)
        
        # Compute A @ B, transpose of A, determinant
        matmul_result = A @ B
        transpose_A = A.T
        det_A = torch.det(A)
        
        print(f"✅ Matrix multiplication result shape: {matmul_result.shape}")
        print(f"✅ Transpose shape: {transpose_A.shape}")
        print(f"✅ Determinant: {det_A:.4f}")
        
        print("\nExercise 3: Reduction operations")
        print("-" * 35)
        
        # Create a 4x5 tensor
        tensor = torch.randn(4, 5)
        
        # Compute various statistics
        mean_all = torch.mean(tensor)
        mean_dim0 = torch.mean(tensor, dim=0)
        max_dim1 = torch.max(tensor, dim=1)
        
        print(f"✅ Overall mean: {mean_all:.4f}")
        print(f"✅ Mean along dim 0 shape: {mean_dim0.shape}")
        print(f"✅ Max along dim 1 values shape: {max_dim1.values.shape}")
        
        print("\nExercise 4: Boolean operations")
        print("-" * 35)
        
        # Create tensor and apply boolean operations
        data = torch.tensor([1, 5, 3, 8, 2, 7, 4, 6])
        
        # Find elements greater than 4
        mask = data > 4
        selected = data[mask]
        
        print(f"✅ Original data: {data}")
        print(f"✅ Elements > 4: {selected}")
        print(f"✅ Count of elements > 4: {torch.sum(mask).item()}")
        
        print("\n🎯 Challenge Exercise:")
        print("Create a function that normalizes a tensor to have zero mean and unit variance")
        
        def normalize_tensor(tensor):
            """Normalize tensor to zero mean and unit variance."""
            mean = torch.mean(tensor)
            std = torch.std(tensor, dim=None)
            return (tensor - mean) / std
        
        test_tensor = torch.randn(100)
        normalized = normalize_tensor(test_tensor)
        
        print(f"✅ Original mean: {torch.mean(test_tensor):.4f}, std: {torch.std(test_tensor, dim=None):.4f}")
        print(f"✅ Normalized mean: {torch.mean(normalized):.4f}, std: {torch.std(normalized, dim=None):.4f}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    print("\n")

def main():
    """Main function to run all tensor operations demonstrations."""
    print("🚀 PyTorch Tensor Operations")
    print("=" * 60)
    print("Master the power of tensor operations!\n")
    
    arithmetic_operations()
    broadcasting_rules()
    matrix_operations()
    reduction_operations()
    comparison_operations()
    advanced_operations()
    performance_tips()
    exercises()
    
    print("🎯 Key Takeaways:")
    print("  1. PyTorch supports extensive mathematical operations")
    print("  2. Broadcasting enables operations between different shapes")
    print("  3. Matrix operations are essential for neural networks")
    print("  4. Reduction operations help analyze data")
    print("  5. Use vectorized operations for better performance")
    
    print("\n🔜 Next: Indexing and slicing in Chapter 2.3!")

if __name__ == "__main__":
    main()
