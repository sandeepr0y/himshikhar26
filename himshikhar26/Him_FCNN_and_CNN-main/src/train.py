"""Training functions for MNIST models"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, device, epoch):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device (cuda or cpu)
        epoch: Current epoch number
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"🧠 Epoch [{epoch}] - Train Loss: {avg_loss:.4f}")
    
    return avg_loss


def train_model(
    model,
    train_loader,
    optimizer,
    num_epochs,
    device,
    scheduler=None
):
    """
    Train model for multiple epochs
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device (cuda or cpu)
        scheduler: Optional learning rate scheduler
        
    Returns:
        List of average losses per epoch
    """
    losses = []
    
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, epoch)
        losses.append(loss)
        
        if scheduler is not None:
            scheduler.step()
    
    return losses
