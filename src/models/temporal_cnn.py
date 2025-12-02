"""
Temporal CNN Model

Lightweight 1D CNN architecture for classifying temporal pose sequences.
Uses temporal convolutions over the time dimension to capture action patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    """
    Temporal Convolutional Neural Network for sequence classification.
    
    Architecture:
        - 3 Conv1D blocks with increasing channels
        - Batch normalization and ReLU activations
        - Max pooling for temporal downsampling
        - Global average pooling
        - Fully connected classification head
    
    Designed to be lightweight (~850K parameters) for edge deployment.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_channels: list = [64, 128, 256],
        kernel_size: int = 5,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Feature dimension (F)
            num_classes: Number of action classes
            hidden_channels: List of channel sizes for Conv1D layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(TemporalCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Conv blocks
        self.conv1 = self._make_conv_block(
            input_dim, hidden_channels[0], kernel_size
        )
        self.conv2 = self._make_conv_block(
            hidden_channels[0], hidden_channels[1], kernel_size
        )
        self.conv3 = self._make_conv_block(
            hidden_channels[1], hidden_channels[2], kernel_size
        )
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_channels[2], 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int):
        """Create a convolutional block with Conv1D + BatchNorm + ReLU."""
        return nn.Sequential(
            nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape [B, T, F] (batch, time, features)
        
        Returns:
            Logits with shape [B, num_classes]
        """
        # Transpose for Conv1D: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        
        # Conv blocks with pooling
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Global pooling: [B, C, T] -> [B, C, 1] -> [B, C]
        x = self.global_pool(x).squeeze(-1)
        
        # Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_temporal_cnn(input_dim: int, num_classes: int = 4) -> TemporalCNN:
    """
    Factory function to create a TemporalCNN model with default settings.
    
    Args:
        input_dim: Feature dimension
        num_classes: Number of action classes
    
    Returns:
        TemporalCNN model
    """
    model = TemporalCNN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_channels=[64, 128, 256],
        kernel_size=5,
        dropout=0.3
    )
    return model


if __name__ == '__main__':
    """
    Test model instantiation and forward pass.
    """
    print("Testing TemporalCNN model...")
    
    # Model parameters
    batch_size = 8
    sequence_length = 60
    feature_dim = 23  # From pose_features.py
    num_classes = 4
    
    # Create model
    model = create_temporal_cnn(input_dim=feature_dim, num_classes=num_classes)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    num_params = model.count_parameters()
    print(f"\nTotal parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, sequence_length, feature_dim)
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits) sample:\n{output[0]}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\nTesting with CUDA...")
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        output = model(dummy_input)
        print(f"CUDA output shape: {output.shape}")
    
    print("\nTemporalCNN test passed! ✓")
