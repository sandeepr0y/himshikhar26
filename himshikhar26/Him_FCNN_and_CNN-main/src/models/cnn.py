"""Modular Convolutional Neural Network (CNN) for MNIST classification"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularCNN(nn.Module):
    """
    Modular Convolutional Neural Network with configurable architecture
    
    Allows customization of:
    - Number of convolutional layers and their parameters
    - Pooling configuration
    - Fully connected layer sizes
    - Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_layers: list = None,
        pool_kernel_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.0
    ):
        """
        Initialize Modular CNN
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            num_classes: Number of output classes
            conv_layers: List of dicts with conv layer configs
                Example: [
                    {"out_channels": 32, "kernel_size": 3, "padding": 1},
                    {"out_channels": 64, "kernel_size": 3, "padding": 1},
                ]
            pool_kernel_size: Kernel size for max pooling
            fc_hidden: Size of hidden fully connected layer
            dropout: Dropout rate (0.0 to 1.0)
        """
        super(ModularCNN, self).__init__()
        
        if conv_layers is None:
            conv_layers = [
                {"out_channels": 32, "kernel_size": 3, "padding": 1},
                {"out_channels": 64, "kernel_size": 3, "padding": 1},
            ]
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers_config = conv_layers
        self.pool_kernel_size = pool_kernel_size
        self.fc_hidden = fc_hidden
        self.dropout_rate = dropout
        
        # Build convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        current_channels = in_channels
        
        for conv_config in conv_layers:
            out_channels = conv_config["out_channels"]
            kernel_size = conv_config["kernel_size"]
            padding = conv_config["padding"]
            
            self.conv_layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size, padding=padding)
            )
            self.pool_layers.append(nn.MaxPool2d(pool_kernel_size, pool_kernel_size))
            self.dropout_layers.append(nn.Dropout2d(dropout))
            
            current_channels = out_channels
        
        # Calculate size after conv and pooling
        # For MNIST: 28x28 input
        # After each pool: size / 2
        # So after 2 pools: 28 / 4 = 7
        self.flattened_size = current_channels * (28 // (pool_kernel_size ** len(conv_layers))) ** 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, fc_hidden)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional layers with pooling
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = F.relu(x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self):
        """String representation of model architecture"""
        config_str = "\n    ".join([
            f"{i}: out_channels={c['out_channels']}, kernel={c['kernel_size']}"
            for i, c in enumerate(self.conv_layers_config)
        ])
        return (
            f"ModularCNN(\n"
            f"  Input channels: {self.in_channels}\n"
            f"  Conv layers:\n    {config_str}\n"
            f"  Pool kernel: {self.pool_kernel_size}\n"
            f"  FC hidden: {self.fc_hidden}\n"
            f"  Output classes: {self.num_classes}\n"
            f"  Dropout: {self.dropout_rate}\n"
            f"  Total parameters: {self.count_parameters()}\n"
            f")"
        )
