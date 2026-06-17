"""Configuration file for MNIST Classification project"""
import torch

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# DATA PATHS
# ============================================
TRAIN_DIR = "MNIST/Images/train"
TEST_DIR = "MNIST/Images/test"
NUM_CLASSES = 10

# ============================================
# MODEL CHECKPOINT PATHS
# ============================================
FCNN_CHECKPOINT = "checkpoints/fcnn_mnist.pth"
CNN_CHECKPOINT = "checkpoints/cnn_mnist.pth"

# ============================================
# MODEL ARCHITECTURES (MODULAR)
# ============================================
FCNN_CONFIG = {
    "input_size": 784,
    "hidden_layers": [128, 64],  # Configurable hidden layer sizes
    "num_classes": 10,
    "activation": "relu",
    "dropout": 0.0  # Set to 0 to disable dropout
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
    "dropout": 0.0  # Set to 0 to disable dropout
}
