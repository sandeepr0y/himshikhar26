"""
Main script for training FCNN and CNN models on MNIST dataset

This script demonstrates:
- Loading MNIST dataset
- Creating modular FCNN and CNN models
- Training models with configurable architectures
- Evaluating and comparing models
- Saving and loading trained models
"""

import os
import torch
from torch import optim
from torch.utils.data import DataLoader

# Import from src
from src.config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE,
    TRAIN_DIR, TEST_DIR, NUM_CLASSES,
    FCNN_CHECKPOINT, CNN_CHECKPOINT,
    FCNN_CONFIG, CNN_CONFIG
)
from src.data import MNISTDataset
from src.models import ModularFCNN, ModularCNN
from src.train import train_model
from src.evaluate import (
    evaluate_model, print_classification_metrics,
    plot_confusion_matrix, plot_training_curves
)
from src.utils import (
    save_model, load_model, print_model_info, compare_models
)


def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("🚀 MNIST CLASSIFICATION - MODULAR PYTORCH PROJECT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print("=" * 70 + "\n")
    
    # ============================================
    # 1. LOAD DATASET
    # ============================================
    print("\n📂 Loading MNIST Dataset...")
    train_dataset = MNISTDataset(
        num_labels=NUM_CLASSES,
        path_dir=TRAIN_DIR
    )
    test_dataset = MNISTDataset(
        num_labels=NUM_CLASSES,
        path_dir=TEST_DIR
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Testing samples: {len(test_dataset)}")
    
    # ============================================
    # 2. CREATE MODELS
    # ============================================
    print("\n" + "=" * 70)
    print("🏗️  CREATING MODELS")
    print("=" * 70)
    
    # Create FCNN with config
    fcnn_model = ModularFCNN(
        input_size=FCNN_CONFIG["input_size"],
        hidden_layers=FCNN_CONFIG["hidden_layers"],
        num_classes=FCNN_CONFIG["num_classes"],
        activation=FCNN_CONFIG["activation"],
        dropout=FCNN_CONFIG["dropout"]
    ).to(DEVICE)
    print_model_info(fcnn_model, "FCNN")
    
    # Create CNN with config
    cnn_model = ModularCNN(
        in_channels=CNN_CONFIG["in_channels"],
        num_classes=CNN_CONFIG["num_classes"],
        conv_layers=CNN_CONFIG["conv_layers"],
        pool_kernel_size=CNN_CONFIG["pool_kernel_size"],
        fc_hidden=CNN_CONFIG["fc_hidden"],
        dropout=CNN_CONFIG["dropout"]
    ).to(DEVICE)
    print_model_info(cnn_model, "CNN")
    
    # ============================================
    # 3. TRAIN FCNN
    # ============================================
    print("\n" + "=" * 70)
    print("🚀 TRAINING FCNN MODEL")
    print("=" * 70)
    
    fcnn_optimizer = optim.Adam(fcnn_model.parameters(), lr=LEARNING_RATE)
    fcnn_losses = train_model(
        fcnn_model, train_loader, fcnn_optimizer, NUM_EPOCHS, DEVICE
    )
    
    # Evaluate FCNN
    print("\n📊 Evaluating FCNN...")
    fcnn_acc, fcnn_loss, fcnn_preds, fcnn_labels = evaluate_model(
        fcnn_model, test_loader, DEVICE
    )
    
    # Save FCNN
    save_model(fcnn_model, FCNN_CHECKPOINT)
    
    # Print FCNN metrics
    print_classification_metrics(fcnn_preds, fcnn_labels, "FCNN")
    plot_confusion_matrix(fcnn_preds, fcnn_labels, "FCNN", NUM_CLASSES)
    
    # ============================================
    # 4. TRAIN CNN
    # ============================================
    print("\n" + "=" * 70)
    print("🚀 TRAINING CNN MODEL")
    print("=" * 70)
    
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    cnn_losses = train_model(
        cnn_model, train_loader, cnn_optimizer, NUM_EPOCHS, DEVICE
    )
    
    # Evaluate CNN
    print("\n📊 Evaluating CNN...")
    cnn_acc, cnn_loss, cnn_preds, cnn_labels = evaluate_model(
        cnn_model, test_loader, DEVICE
    )
    
    # Save CNN
    save_model(cnn_model, CNN_CHECKPOINT)
    
    # Print CNN metrics
    print_classification_metrics(cnn_preds, cnn_labels, "CNN")
    plot_confusion_matrix(cnn_preds, cnn_labels, "CNN", NUM_CLASSES)
    
    # ============================================
    # 5. PLOT TRAINING CURVES
    # ============================================
    print("\n📊 Plotting training curves...")
    plot_training_curves(
        {"FCNN": fcnn_losses, "CNN": cnn_losses},
        NUM_EPOCHS
    )
    
    # ============================================
    # 6. VERIFY SAVED MODELS
    # ============================================
    print("\n" + "=" * 70)
    print("🔄 VERIFYING SAVED MODELS")
    print("=" * 70)
    
    # Reload and test FCNN
    print("\n📦 Reloading FCNN model...")
    loaded_fcnn = ModularFCNN(
        input_size=FCNN_CONFIG["input_size"],
        hidden_layers=FCNN_CONFIG["hidden_layers"],
        num_classes=FCNN_CONFIG["num_classes"],
        activation=FCNN_CONFIG["activation"],
        dropout=FCNN_CONFIG["dropout"]
    ).to(DEVICE)
    loaded_fcnn = load_model(loaded_fcnn, FCNN_CHECKPOINT)
    fcnn_reload_acc, _, _, _ = evaluate_model(loaded_fcnn, test_loader, DEVICE)
    
    # Reload and test CNN
    print("\n📦 Reloading CNN model...")
    loaded_cnn = ModularCNN(
        in_channels=CNN_CONFIG["in_channels"],
        num_classes=CNN_CONFIG["num_classes"],
        conv_layers=CNN_CONFIG["conv_layers"],
        pool_kernel_size=CNN_CONFIG["pool_kernel_size"],
        fc_hidden=CNN_CONFIG["fc_hidden"],
        dropout=CNN_CONFIG["dropout"]
    ).to(DEVICE)
    loaded_cnn = load_model(loaded_cnn, CNN_CHECKPOINT)
    cnn_reload_acc, _, _, _ = evaluate_model(loaded_cnn, test_loader, DEVICE)
    
    # ============================================
    # 7. FINAL RESULTS
    # ============================================
    results = {
        "FCNN": {
            "accuracy": fcnn_acc,
            "loss": fcnn_loss,
            "parameters": fcnn_model.count_parameters()
        },
        "CNN": {
            "accuracy": cnn_acc,
            "loss": cnn_loss,
            "parameters": cnn_model.count_parameters()
        }
    }
    
    compare_models(results)
    
    print("✅ Training complete! Models saved to 'checkpoints/' directory.")
    print("📝 To customize model architectures, edit 'src/config.py'")


if __name__ == "__main__":
    main()
