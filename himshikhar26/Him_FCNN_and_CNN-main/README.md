# MNIST Classification - Modular PyTorch Project

A refactored, modular version of the MNIST classification project that extracts code from a Jupyter notebook into well-organized Python modules.

## 📁 Project Structure

```
.
├── main.py                      # Main entry point for training
├── requirements.txt             # Project dependencies
├── README.md                    # This file
├── MNIST/                       # Dataset directory
│   ├── Images/
│   │   ├── train/              # Training images (labeled subdirectories)
│   │   └── test/               # Test images (labeled subdirectories)
│   ├── mnist_train.csv
│   ├── mnist_test.csv
│   └── readme.txt
├── checkpoints/                 # Saved model weights
│   ├── fcnn_mnist.pth
│   └── cnn_mnist.pth
└── src/                         # Source code modules
    ├── __init__.py
    ├── config.py                # Configuration and hyperparameters
    ├── train.py                 # Training functions
    ├── evaluate.py              # Evaluation and metrics
    ├── utils.py                 # Utility functions
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py           # MNIST Dataset class
    └── models/
        ├── __init__.py
        ├── fcnn.py              # Modular FCNN model
        └── cnn.py               # Modular CNN model
```

## 🎯 Key Features

### Modular Architecture
- **Configurable Models**: Both FCNN and CNN accept configuration dictionaries
- **Flexible FCNN**: Customize hidden layer sizes, activation functions, and dropout
- **Flexible CNN**: Configure convolution layers, pooling, and fully connected sizes
- **Separated Concerns**: Each module handles a specific responsibility

### Components

#### 1. **src/config.py**
Central configuration file with modular model architectures:
```python
FCNN_CONFIG = {
    "input_size": 784,
    "hidden_layers": [128, 64],        # Change layer sizes here
    "num_classes": 10,
    "activation": "relu",              # or "tanh", "sigmoid"
    "dropout": 0.0                     # Add dropout if needed
}

CNN_CONFIG = {
    "in_channels": 1,
    "num_classes": 10,
    "conv_layers": [
        {"out_channels": 32, "kernel_size": 3, "padding": 1},
        {"out_channels": 64, "kernel_size": 3, "padding": 1},
    ],
    "pool_kernel_size": 2,
    "fc_hidden": 128,
    "dropout": 0.0
}
```

#### 2. **src/models/fcnn.py**
`ModularFCNN` class with dynamic layer construction:
- Accepts any number of hidden layers
- Supports different activation functions
- Includes optional dropout regularization
- Can count total trainable parameters

#### 3. **src/models/cnn.py**
`ModularCNN` class with flexible convolution architecture:
- Dynamic number of convolutional layers
- Configurable pooling and fully connected sizes
- Supports dropout for regularization
- Automatically calculates flattened size

#### 4. **src/data/dataset.py**
`MNISTDataset` class:
- Loads images from directory structure
- Automatic label mapping
- Preprocessing with configurable transforms
- Supports different file extensions

#### 5. **src/train.py**
Training functions:
- `train_epoch()`: Single epoch training
- `train_model()`: Multi-epoch training with optional scheduler support

#### 6. **src/evaluate.py**
Evaluation utilities:
- `evaluate_model()`: Get accuracy and loss
- `print_classification_metrics()`: Classification report
- `plot_confusion_matrix()`: Visualize predictions
- `plot_training_curves()`: Compare training progress

#### 7. **src/utils.py**
Helper functions:
- Model saving/loading
- Model info printing
- Results comparison

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Training
```bash
python main.py
```

This will:
1. Load the MNIST dataset from `MNIST/Images/` directory
2. Create FCNN and CNN models with configurations from `config.py`
3. Train both models for `NUM_EPOCHS` epochs
4. Evaluate on test set and print metrics
5. Display confusion matrices
6. Plot training loss curves
7. Save models to `checkpoints/` directory
8. Reload and verify saved models

## 🔧 Customization Guide

### Change FCNN Architecture
Edit `src/config.py`:
```python
FCNN_CONFIG = {
    "input_size": 784,
    "hidden_layers": [256, 128, 64, 32],  # 4 hidden layers
    "num_classes": 10,
    "activation": "relu",
    "dropout": 0.1                        # Add 10% dropout
}
```

### Change CNN Architecture
```python
CNN_CONFIG = {
    "in_channels": 1,
    "num_classes": 10,
    "conv_layers": [
        {"out_channels": 32, "kernel_size": 3, "padding": 1},
        {"out_channels": 64, "kernel_size": 3, "padding": 1},
        {"out_channels": 128, "kernel_size": 3, "padding": 1},  # Add layer
    ],
    "pool_kernel_size": 2,
    "fc_hidden": 256,                    # Increase FC layer
    "dropout": 0.2
}
```

### Change Training Parameters
```python
# In src/config.py
BATCH_SIZE = 64          # Increase batch size
LEARNING_RATE = 0.0005   # Reduce learning rate
NUM_EPOCHS = 50          # More epochs
```

### Use Different Activation Functions
```python
FCNN_CONFIG = {
    ...
    "activation": "tanh",  # or "sigmoid"
    ...
}
```

## 📊 Model Information

When you run the training script, it will display:
- Model architecture details
- Total trainable parameters
- Training progress with loss
- Test accuracy and loss
- Classification report (precision, recall, F1)
- Confusion matrix
- Training curves comparison

## 💾 Checkpoints

Models are automatically saved to:
- `checkpoints/fcnn_mnist.pth` - FCNN weights
- `checkpoints/cnn_mnist.pth` - CNN weights

### Load a Saved Model
```python
from src.models import ModularFCNN
from src.config import FCNN_CONFIG, FCNN_CHECKPOINT
from src.utils import load_model

model = ModularFCNN(**FCNN_CONFIG)
model = load_model(model, FCNN_CHECKPOINT)
```

## 🧪 Example: Adding a New Layer to FCNN

```python
# In src/config.py
FCNN_CONFIG = {
    "input_size": 784,
    "hidden_layers": [512, 256, 128, 64],  # Added 512 at start, 128 and 64 at end
    "num_classes": 10,
    "activation": "relu",
    "dropout": 0.1
}

# Run: python main.py
```

The model will automatically create 4 hidden layers with these sizes!

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Main training pipeline |
| `src/config.py` | Hyperparameters & model configs |
| `src/train.py` | Training functions |
| `src/evaluate.py` | Evaluation and visualization |
| `src/utils.py` | Utility functions |
| `src/data/dataset.py` | MNIST dataset loader |
| `src/models/fcnn.py` | Modular FCNN model |
| `src/models/cnn.py` | Modular CNN model |

## 🔄 Workflow

1. **Configure** → Edit `src/config.py` to adjust models/training
2. **Train** → Run `python main.py`
3. **Analyze** → View metrics, plots, and confusion matrices
4. **Save** → Models automatically saved to `checkpoints/`
5. **Reload** → Load any saved model for inference

## 📚 Original Notebook

The code is refactored from: `MNIST_Classification_using_FCNN_&_CNN.ipynb`

All functionality from the notebook is preserved but organized into reusable modules.

## 🎓 Learning Points

This project demonstrates:
- ✅ Modular PyTorch code organization
- ✅ Configuration-driven architecture design
- ✅ Reusable training/evaluation loops
- ✅ Proper project structure for scalability
- ✅ Model saving/loading best practices
- ✅ Comprehensive evaluation metrics
