"""Modular Fully Connected Neural Network (FCNN) for MNIST classification"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularFCNN(nn.Module):
    """
    Modular Fully Connected Neural Network with configurable architecture
    
    Allows customization of:
    - Input size
    - Number of hidden layers and their sizes
    - Number of output classes
    - Activation function
    - Dropout rate
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_layers: list = None,
        num_classes: int = 10,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize Modular FCNN
        
        Args:
            input_size: Size of input (e.g., 784 for 28x28 images flattened)
            hidden_layers: List of hidden layer sizes, e.g., [128, 64, 32]
            num_classes: Number of output classes
            activation: Activation function name ("relu", "tanh", "sigmoid")
            dropout: Dropout rate (0.0 to 1.0)
        """
        super(ModularFCNN, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64]
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation_name = activation
        self.dropout_rate = dropout
        
        # Select activation function
        if activation.lower() == "relu":
            self.activation = F.relu
        elif activation.lower() == "tanh":
            self.activation = torch.tanh
        elif activation.lower() == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers dynamically
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], num_classes))
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, ...)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Pass through hidden layers with activation
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout_layers[i](x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self):
        """String representation of model architecture"""
        return (
            f"ModularFCNN(\n"
            f"  Input: {self.input_size}\n"
            f"  Hidden layers: {self.hidden_layers}\n"
            f"  Output: {self.num_classes}\n"
            f"  Activation: {self.activation_name}\n"
            f"  Dropout: {self.dropout_rate}\n"
            f"  Total parameters: {self.count_parameters()}\n"
            f")"
        )
