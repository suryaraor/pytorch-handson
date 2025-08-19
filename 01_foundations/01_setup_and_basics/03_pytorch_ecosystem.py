"""
Chapter 1.3: PyTorch Ecosystem
==============================

Explore the rich ecosystem of tools and libraries built around PyTorch.

Author: PyTorch Deep Dive Tutorial
Date: 2025
"""

def check_ecosystem_libraries():
    """Check which PyTorch ecosystem libraries are available."""
    print("🌐 PyTorch Ecosystem Libraries Check")
    print("=" * 50)
    
    ecosystem_libs = {
        # Core Libraries
        'torch': 'Core PyTorch library',
        'torchvision': 'Computer vision tools and datasets',
        'torchaudio': 'Audio processing and datasets',
        
        # High-level Frameworks
        'pytorch_lightning': 'High-level training framework',
        'ignite': 'High-level library for training and evaluating',
        
        # Specialized Libraries
        'transformers': 'Hugging Face Transformers for NLP',
        'timm': 'PyTorch Image Models',
        'detectron2': 'Facebook AI Research object detection',
        
        # Utilities
        'tensorboard': 'TensorBoard for PyTorch',
        'wandb': 'Weights & Biases experiment tracking',
        'optuna': 'Hyperparameter optimization',
        
        # Deployment
        'torchserve': 'Model serving',
        'onnx': 'Open Neural Network Exchange',
    }
    
    available_libs = []
    missing_libs = []
    
    for lib, description in ecosystem_libs.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {lib} ({version})")
            print(f"   {description}")
            available_libs.append(lib)
        except ImportError:
            print(f"❌ {lib}")
            print(f"   {description} (Not installed)")
            missing_libs.append(lib)
        print()
    
    print(f"📊 Summary: {len(available_libs)} available, {len(missing_libs)} missing")
    
    if missing_libs:
        print("\n📦 To install missing libraries:")
        print("pip install " + " ".join(missing_libs))
    
    print("\n")

def torchvision_demo():
    """Demonstrate torchvision capabilities."""
    print("🖼️ TorchVision Demo")
    print("=" * 50)
    
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        from torchvision import datasets
        
        print(f"TorchVision version: {torchvision.__version__}")
        
        # Common transforms
        print("\n🔄 Common Image Transforms:")
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        transforms.Compose(transform_list)  # Example composition
        print("  • Resize to 224x224")
        print("  • Convert to tensor")
        print("  • Normalize with ImageNet statistics")
        
        # Available datasets
        print("\n📚 Popular Datasets (examples):")
        dataset_info = {
            'CIFAR10': 'Small 32x32 color images, 10 classes',
            'CIFAR100': 'Small 32x32 color images, 100 classes',
            'MNIST': 'Handwritten digits, 28x28 grayscale',
            'FashionMNIST': 'Fashion items, 28x28 grayscale',
            'ImageNet': 'Large-scale image dataset, 1000 classes'
        }
        
        for dataset, description in dataset_info.items():
            print(f"  • {dataset}: {description}")
        
        # Pre-trained models
        print("\n🏗️ Pre-trained Models (examples):")
        models_info = {
            'ResNet': 'Deep residual networks',
            'VGG': 'Very deep convolutional networks',
            'DenseNet': 'Densely connected networks',
            'MobileNet': 'Efficient mobile networks',
            'EfficientNet': 'Compound scaled networks'
        }
        
        for model, description in models_info.items():
            print(f"  • {model}: {description}")
        
        print("\n✨ Example usage:")
        print("  import torchvision.models as models")
        print("  model = models.resnet18(pretrained=True)")
        
    except ImportError as e:
        print(f"❌ TorchVision not available: {e}")
        print("Install with: pip install torchvision")
    
    print("\n")

def training_frameworks_overview():
    """Overview of high-level training frameworks."""
    print("⚡ High-Level Training Frameworks")
    print("=" * 50)
    
    frameworks = {
        'PyTorch Lightning': {
            'description': 'Lightweight PyTorch wrapper for professional ML',
            'pros': ['Cleaner code', 'Built-in best practices', 'Easy scaling'],
            'use_case': 'Research and production ML projects'
        },
        'Ignite': {
            'description': 'High-level library for training and evaluating',
            'pros': ['Flexible', 'Event-driven', 'Extensible'],
            'use_case': 'Custom training loops with reusable components'
        },
        'FastAI': {
            'description': 'High-level API built on PyTorch',
            'pros': ['Beginner-friendly', 'Best practices built-in', 'Fast prototyping'],
            'use_case': 'Rapid prototyping and education'
        },
        'Catalyst': {
            'description': 'PyTorch framework for DL research and development',
            'pros': ['Reproducible', 'Configurable', 'Production-ready'],
            'use_case': 'Large-scale ML projects'
        }
    }
    
    for framework, info in frameworks.items():
        print(f"🚀 {framework}")
        print(f"   {info['description']}")
        print(f"   Pros: {', '.join(info['pros'])}")
        print(f"   Use case: {info['use_case']}")
        print()
    
    # Check if Lightning is available
    try:
        import pytorch_lightning as pl
        print(f"✅ PyTorch Lightning {pl.__version__} is available!")
        print("   Try: import pytorch_lightning as pl")
    except ImportError:
        print("❌ PyTorch Lightning not installed")
        print("   Install: pip install pytorch-lightning")
    
    print("\n")

def deployment_tools_overview():
    """Overview of deployment and production tools."""
    print("🏭 Deployment & Production Tools")
    print("=" * 50)
    
    tools = {
        'TorchScript': {
            'purpose': 'Compile PyTorch models for production',
            'benefits': ['No Python dependency', 'Optimized execution', 'Mobile support'],
            'example': 'torch.jit.trace(model, example_input)'
        },
        'TorchServe': {
            'purpose': 'Serve PyTorch models at scale',
            'benefits': ['REST/gRPC APIs', 'Model management', 'A/B testing'],
            'example': 'torch-model-archiver --model-name resnet18'
        },
        'ONNX': {
            'purpose': 'Interoperability between frameworks',
            'benefits': ['Framework agnostic', 'Hardware optimization', 'Broad support'],
            'example': 'torch.onnx.export(model, dummy_input, "model.onnx")'
        },
        'TensorRT': {
            'purpose': 'NVIDIA GPU optimization',
            'benefits': ['High performance', 'Low latency', 'FP16/INT8 support'],
            'example': 'Convert ONNX to TensorRT engine'
        }
    }
    
    for tool, info in tools.items():
        print(f"🔧 {tool}")
        print(f"   Purpose: {info['purpose']}")
        print(f"   Benefits: {', '.join(info['benefits'])}")
        print(f"   Example: {info['example']}")
        print()
    
    print("\n")

def experiment_tracking_tools():
    """Overview of experiment tracking and monitoring tools."""
    print("📊 Experiment Tracking & Monitoring")
    print("=" * 50)
    
    tools = {
        'Weights & Biases (wandb)': {
            'features': ['Experiment tracking', 'Hyperparameter sweeps', 'Model versioning'],
            'setup': 'pip install wandb && wandb login',
            'code': 'import wandb; wandb.init(project="my-project")'
        },
        'TensorBoard': {
            'features': ['Scalar/image logging', 'Graph visualization', 'Profiling'],
            'setup': 'pip install tensorboard',
            'code': 'from torch.utils.tensorboard import SummaryWriter'
        },
        'MLflow': {
            'features': ['Experiment tracking', 'Model registry', 'Deployment'],
            'setup': 'pip install mlflow',
            'code': 'import mlflow; mlflow.start_run()'
        },
        'Neptune': {
            'features': ['Experiment management', 'Model monitoring', 'Collaboration'],
            'setup': 'pip install neptune-client',
            'code': 'import neptune; run = neptune.init()'
        }
    }
    
    for tool, info in tools.items():
        print(f"📈 {tool}")
        print(f"   Features: {', '.join(info['features'])}")
        print(f"   Setup: {info['setup']}")
        print(f"   Usage: {info['code']}")
        print()
    
    # Check if wandb is available
    try:
        import wandb
        print(f"✅ Weights & Biases {wandb.__version__} is available!")
    except ImportError:
        print("❌ Weights & Biases not installed")
        print("   Install: pip install wandb")
    
    print("\n")

def create_ecosystem_cheatsheet():
    """Create a quick reference cheatsheet."""
    print("📝 PyTorch Ecosystem Cheatsheet")
    print("=" * 50)
    
    cheatsheet = """
🔥 CORE PYTORCH
├── torch                 # Core tensor operations
├── torch.nn             # Neural network layers
├── torch.optim          # Optimizers (SGD, Adam, etc.)
└── torch.utils.data     # Data loading utilities

🖼️ COMPUTER VISION
├── torchvision          # CV datasets, models, transforms
├── timm                 # Pre-trained models
├── detectron2           # Object detection
└── kornia               # Differentiable CV library

🗣️ NATURAL LANGUAGE PROCESSING
├── transformers         # Hugging Face models
├── torchtext           # Text processing utilities
└── fairseq             # Sequence-to-sequence toolkit

🎵 AUDIO PROCESSING
├── torchaudio          # Audio datasets and transforms
├── speechbrain         # Speech processing toolkit
└── asteroid            # Audio source separation

⚡ HIGH-LEVEL FRAMEWORKS
├── pytorch_lightning   # Professional ML framework
├── ignite              # Training and evaluation toolkit
├── fastai              # High-level API
└── catalyst            # DL research framework

🏭 DEPLOYMENT & PRODUCTION
├── torchscript         # Model compilation
├── torchserve          # Model serving
├── onnx                # Interoperability
└── tensorrt            # GPU optimization

📊 EXPERIMENT TRACKING
├── wandb               # Weights & Biases
├── tensorboard         # Google's visualization toolkit
├── mlflow              # ML lifecycle management
└── neptune             # Experiment management

🔧 OPTIMIZATION & UTILITIES
├── optuna              # Hyperparameter optimization
├── ray                 # Distributed computing
└── apex                # Mixed precision training
    """
    
    print(cheatsheet)
    print("\n")

def main():
    """Main function to run all ecosystem demos."""
    print("🚀 PyTorch Ecosystem Deep Dive")
    print("=" * 60)
    print("Exploring the rich ecosystem around PyTorch!\n")
    
    check_ecosystem_libraries()
    torchvision_demo()
    training_frameworks_overview()
    deployment_tools_overview()
    experiment_tracking_tools()
    create_ecosystem_cheatsheet()
    
    print("🎯 Key Takeaways:")
    print("  1. PyTorch has a rich ecosystem for every ML task")
    print("  2. TorchVision provides ready-to-use CV tools")
    print("  3. High-level frameworks simplify training")
    print("  4. Production tools enable real-world deployment")
    print("  5. Experiment tracking ensures reproducible research")
    
    print("\n🔜 Next: GPU setup and optimization in Chapter 1.4!")

if __name__ == "__main__":
    main()
