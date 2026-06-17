"""Utility functions for training and evaluation"""
import os
import torch


def create_checkpoint_dir(checkpoint_path):
    """Create directory for checkpoints if it doesn't exist"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)


def save_model(model, checkpoint_path):
    """
    Save model state dict
    
    Args:
        model: PyTorch model to save
        checkpoint_path: Path to save checkpoint
    """
    create_checkpoint_dir(checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Model saved to '{checkpoint_path}'")


def load_model(model, checkpoint_path):
    """
    Load model state dict
    
    Args:
        model: PyTorch model to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"✅ Model loaded from '{checkpoint_path}'")
    return model


def print_model_info(model, model_name="Model"):
    """Print model architecture and parameter count"""
    print(f"\n{'=' * 60}")
    print(f"🏗️  {model_name} ARCHITECTURE")
    print(f"{'=' * 60}")
    print(model)
    print(f"{'=' * 60}\n")


def compare_models(results_dict):
    """
    Compare and print results for multiple models
    
    Args:
        results_dict: Dictionary with model results
            {model_name: {"accuracy": acc, "loss": loss, "params": params}}
    """
    print("\n" + "=" * 70)
    print("🏁 FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    best_model = None
    best_acc = -1
    
    for model_name, metrics in results_dict.items():
        acc = metrics["accuracy"]
        loss = metrics["loss"]
        params = metrics["parameters"]
        
        print(f"\n{model_name}:")
        print(f"  📊 Accuracy: {acc:.2f}%")
        print(f"  📉 Loss: {loss:.4f}")
        print(f"  🔢 Parameters: {params:,}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model_name
    
    print(f"\n🎯 Best Model: {best_model} with {best_acc:.2f}% accuracy")
    print("=" * 70 + "\n")
