"""
Training Pipeline

Complete training script for fencing action recognition models.
Includes:
- Train/val split
- Training loop with validation
- Learning rate scheduling
- Model checkpointing
- Training curves logging
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import (
    FencingPoseDataset, 
    create_data_splits, 
    create_dataloaders,
    ACTIONS
)
from models.temporal_cnn import create_temporal_cnn
from models.lstm_model import create_lstm_classifier


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_features, batch_labels in pbar:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Validate the model.
    
    Returns:
        Tuple of (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
        for batch_features, batch_labels in pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def plot_training_curves(history: dict, output_path: Path):
    """
    Plot and save training curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train fencing action recognition model"
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir', type=str, default='data/raw',
        help='Directory containing pose sequences and labels.csv'
    )
    parser.add_argument(
        '--output_dir', type=str, default='models/',
        help='Directory to save trained models'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_type', type=str, default='cnn',
        choices=['cnn', 'lstm', 'lstm_attention'],
        help='Type of model to train'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5,
        help='L2 regularization weight decay'
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience (epochs)'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='Number of data loading workers'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = FencingPoseDataset(args.data_dir)
    
    # Get feature dimension from a sample
    sample_features, _ = dataset[0]
    feature_dim = sample_features.shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15,
        seed=args.seed
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    num_classes = len(ACTIONS)
    
    if args.model_type == 'cnn':
        model = create_temporal_cnn(input_dim=feature_dim, num_classes=num_classes)
    elif args.model_type == 'lstm':
        model = create_lstm_classifier(
            input_dim=feature_dim, 
            num_classes=num_classes,
            use_attention=False
        )
    elif args.model_type == 'lstm_attention':
        model = create_lstm_classifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            use_attention=True
        )
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_type': args.model_type,
                'feature_dim': feature_dim,
                'num_classes': num_classes
            }, best_model_path)
            
            print(f"  ✓ New best model saved (val_acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
        
        print("=" * 70)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': args.model_type,
        'feature_dim': feature_dim,
        'num_classes': num_classes
    }, final_model_path)
    
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves
    curves_path = output_dir / 'training_curves.png'
    plot_training_curves(history, curves_path)
    
    # Save training configuration
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    config['feature_dim'] = feature_dim
    config['num_classes'] = num_classes
    config['num_parameters'] = num_params
    
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved in: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
