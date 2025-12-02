"""
Model Evaluation

Comprehensive evaluation script for trained models.
Computes accuracy, per-class metrics, and confusion matrix.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

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


def load_model(model_path: str, device: torch.device):
    """
    Load a trained model from checkpoint.
    
    Returns:
        Loaded model and checkpoint metadata
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint['model_type']
    feature_dim = checkpoint['feature_dim']
    num_classes = checkpoint['num_classes']
    
    # Create model
    if model_type == 'cnn':
        model = create_temporal_cnn(input_dim=feature_dim, num_classes=num_classes)
    elif model_type == 'lstm':
        model = create_lstm_classifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            use_attention=False
        )
    elif model_type == 'lstm_attention':
        model = create_lstm_classifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            use_attention=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model from {model_path}")
    if 'val_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, checkpoint


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate model on a dataset.
    
    Returns:
        Tuple of (all_predictions, all_labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in tqdm(data_loader, desc="Evaluating"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path,
    normalize: bool = False
):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list
):
    """
    Print detailed classification metrics.
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10}")
    
    print("-" * 70)
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print(f"{'Macro Avg':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} "
          f"{f1_macro:<12.4f}")
    print(f"{'Weighted Avg':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} "
          f"{f1_weighted:<12.4f}")
    print("-" * 70)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained fencing action recognition model"
    )
    
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/raw',
        help='Directory containing pose sequences'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        choices=['train', 'val', 'test'],
        help='Which data split to evaluate on'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (must match training seed for correct splits)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, checkpoint = load_model(args.model_path, device)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = FencingPoseDataset(args.data_dir)
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )
    
    # Select split
    if args.split == 'train':
        eval_dataset = train_dataset
    elif args.split == 'val':
        eval_dataset = val_dataset
    else:
        eval_dataset = test_dataset
    
    print(f"\nEvaluating on {args.split} set ({len(eval_dataset)} samples)...")
    
    # Create dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    predictions, labels = evaluate_model(model, eval_loader, device)
    
    # Print metrics
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS ({args.split.upper()} SET)")
    print("=" * 70)
    
    print_classification_metrics(labels, predictions, ACTIONS)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrices
    cm_path = output_dir / f'confusion_matrix_{args.split}.png'
    plot_confusion_matrix(cm, ACTIONS, cm_path, normalize=False)
    
    cm_norm_path = output_dir / f'confusion_matrix_{args.split}_normalized.png'
    plot_confusion_matrix(cm, ACTIONS, cm_norm_path, normalize=True)
    
    # Save results to file
    results = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'per_class_metrics': {},
        'confusion_matrix': cm.tolist()
    }
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=range(len(ACTIONS))
    )
    
    for i, action in enumerate(ACTIONS):
        results['per_class_metrics'][action] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    import json
    results_path = output_dir / f'evaluation_results_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
