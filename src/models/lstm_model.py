"""
LSTM Model

Bidirectional LSTM architecture for classifying temporal pose sequences.
Uses recurrent connections to model long-range temporal dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for sequence classification.
    
    Architecture:
        - 2-layer bidirectional LSTM
        - Dropout for regularization
        - Final hidden state -> FC classification head
    
    Designed to be lightweight (~520K parameters) for edge deployment.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: Feature dimension (F)
            num_classes: Number of action classes
            hidden_size: Hidden state size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape [B, T, F] (batch, time, features)
        
        Returns:
            Logits with shape [B, num_classes]
        """
        # LSTM forward pass
        # output: [B, T, H*2] (if bidirectional)
        # h_n: [num_layers*2, B, H] (final hidden states)
        # c_n: [num_layers*2, B, H] (final cell states)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from last layer
        # For bidirectional: concatenate forward and backward hidden states
        if self.bidirectional:
            # h_n shape: [num_layers*2, B, H]
            # Get last layer's forward and backward states
            h_forward = h_n[-2, :, :]  # [B, H]
            h_backward = h_n[-1, :, :]  # [B, H]
            h_final = torch.cat([h_forward, h_backward], dim=1)  # [B, H*2]
        else:
            h_final = h_n[-1, :, :]  # [B, H]
        
        # Classification head
        x = self.fc1(h_final)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for sequence classification.
    
    Attention allows the model to focus on relevant time steps for classification.
    This is more sophisticated and can improve performance on complex actions.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: Feature dimension (F)
            num_classes: Number of action classes
            hidden_size: Hidden state size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(AttentionLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Linear(lstm_output_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor with shape [B, T, F]
        
        Returns:
            Logits with shape [B, num_classes]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [B, T, H*2]
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize over time
        
        # Weighted sum of LSTM outputs
        # [B, T, H*2] * [B, T, 1] -> [B, T, H*2] -> sum over T -> [B, H*2]
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification head
        x = self.fc1(context)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_lstm_classifier(
    input_dim: int, 
    num_classes: int = 4,
    use_attention: bool = False
) -> nn.Module:
    """
    Factory function to create an LSTM-based classifier.
    
    Args:
        input_dim: Feature dimension
        num_classes: Number of action classes
        use_attention: Whether to use attention mechanism
    
    Returns:
        LSTMClassifier or AttentionLSTM model
    """
    if use_attention:
        model = AttentionLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
    else:
        model = LSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
    return model


if __name__ == '__main__':
    """
    Test model instantiation and forward pass.
    """
    print("Testing LSTM models...")
    
    # Model parameters
    batch_size = 8
    sequence_length = 60
    feature_dim = 23  # From pose_features.py
    num_classes = 4
    
    # Test standard LSTM
    print("\n=== Standard LSTM Classifier ===")
    model = create_lstm_classifier(
        input_dim=feature_dim, 
        num_classes=num_classes,
        use_attention=False
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    num_params = model.count_parameters()
    print(f"\nTotal parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, sequence_length, feature_dim)
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits) sample:\n{output[0]}")
    
    # Test attention LSTM
    print("\n=== LSTM with Attention ===")
    model_attn = create_lstm_classifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        use_attention=True
    )
    
    num_params_attn = model_attn.count_parameters()
    print(f"Total parameters: {num_params_attn:,} (~{num_params_attn/1e6:.2f}M)")
    
    with torch.no_grad():
        output_attn = model_attn(dummy_input)
    
    print(f"Output shape: {output_attn.shape}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\nTesting with CUDA...")
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        output = model(dummy_input)
        print(f"CUDA output shape: {output.shape}")
    
    print("\nLSTM model tests passed! ✓")
