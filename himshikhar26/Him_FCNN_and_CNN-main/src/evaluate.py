"""Evaluation and testing functions"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device (cuda or cpu)
        
    Returns:
        Tuple of (accuracy, average_loss, predictions, true_labels)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            # Get predictions
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            
            # Store for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    
    print(f"✅ Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    return accuracy, test_loss, np.array(all_preds), np.array(all_labels)


def print_classification_metrics(predictions, true_labels, model_name="Model"):
    """
    Print detailed classification metrics
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        model_name: Name of model for printing
    """
    print(f"\n📊 {model_name} PERFORMANCE METRICS")
    print("=" * 50)
    print(classification_report(true_labels, predictions, digits=4))


def plot_confusion_matrix(predictions, true_labels, model_name="Model", num_classes=10):
    """
    Plot and display confusion matrix
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        model_name: Name of model for title
        num_classes: Number of classes
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=[str(i) for i in range(num_classes)],
        yticklabels=[str(i) for i in range(num_classes)]
    )
    plt.title(f"🧩 Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
    
    return cm


def plot_training_curves(losses_dict, num_epochs):
    """
    Plot training loss curves for multiple models
    
    Args:
        losses_dict: Dictionary of {model_name: losses_list}
        num_epochs: Number of epochs
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, losses in losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, 'o-', label=model_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('📉 Training Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
