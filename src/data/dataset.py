"""
PyTorch Dataset and DataLoader Utilities

Provides dataset classes for loading and preprocessing fencing pose sequences.
Handles train/val/test splits and feature extraction pipeline.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from features.pose_features import extract_sequence_features


# Action labels
ACTIONS = ['idle', 'advance', 'retreat', 'lunge']
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}


class FencingPoseDataset(Dataset):
    """
    Dataset for fencing pose sequences.
    
    Loads raw pose keypoints, extracts features, and returns
    (features, label) pairs for training.
    """
    
    def __init__(
        self,
        data_dir: str,
        labels_file: str = 'labels.csv',
        transform=None,
        feature_cache: Optional[Dict] = None
    ):
        """
        Args:
            data_dir: Directory containing .npy pose files and labels.csv
            labels_file: Name of the CSV file with labels
            transform: Optional transform to apply to features
            feature_cache: Optional dict to cache extracted features
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.feature_cache = feature_cache if feature_cache is not None else {}
        
        # Load labels
        labels_path = self.data_dir / labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        self.labels_df = pd.read_csv(labels_path)
        
        # Validate action labels
        unknown_actions = set(self.labels_df['label'].unique()) - set(ACTIONS)
        if unknown_actions:
            raise ValueError(f"Unknown actions in dataset: {unknown_actions}")
        
        print(f"Loaded dataset with {len(self.labels_df)} sequences")
        print(f"Action distribution:")
        for action in ACTIONS:
            count = (self.labels_df['label'] == action).sum()
            print(f"  {action}: {count}")
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (features, label_idx):
                features: Tensor with shape [T, F] (time, features)
                label_idx: Integer label index
        """
        # Get file and label
        row = self.labels_df.iloc[idx]
        filename = row['file']
        label = row['label']
        label_idx = ACTION_TO_IDX[label]
        
        # Check cache first
        cache_key = filename
        if cache_key in self.feature_cache:
            features = self.feature_cache[cache_key]
        else:
            # Load pose sequence
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Sequence file not found: {filepath}")
            
            pose_sequence = np.load(filepath)  # [T, K, 2]
            
            # Extract features
            features = extract_sequence_features(pose_sequence)  # [T, F]
            
            # Cache for future use
            self.feature_cache[cache_key] = features
        
        # Apply transform if specified
        if self.transform:
            features = self.transform(features)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label_idx])[0]
        
        return features_tensor, label_tensor
    
    def get_action_name(self, label_idx: int) -> str:
        """Convert label index to action name."""
        return IDX_TO_ACTION[label_idx]
    
    def get_label_counts(self) -> Dict[str, int]:
        """Get count of samples per action."""
        return self.labels_df['label'].value_counts().to_dict()


def create_data_splits(
    dataset: FencingPoseDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Full dataset to split
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({test_ratio*100:.0f}%)")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_dataset, val_dataset, test_dataset: Dataset splits
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


def collate_sequences(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    Pads sequences to the same length in a batch.
    
    Args:
        batch: List of (features, label) tuples
    
    Returns:
        Tuple of (padded_features, labels):
            padded_features: [B, T_max, F]
            labels: [B]
    """
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    
    # Find max sequence length in batch
    max_len = max(f.shape[0] for f in features_list)
    feature_dim = features_list[0].shape[1]
    
    # Pad sequences
    padded_features = torch.zeros(len(batch), max_len, feature_dim)
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len, :] = features
    
    labels = torch.stack(labels_list)
    
    return padded_features, labels


def compute_dataset_statistics(dataset: FencingPoseDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of features across the entire dataset.
    Useful for feature normalization.
    
    Args:
        dataset: Dataset to analyze
    
    Returns:
        Tuple of (mean, std) with shape [F]
    """
    all_features = []
    
    print("Computing dataset statistics...")
    for i in range(len(dataset)):
        features, _ = dataset[i]
        all_features.append(features.numpy())
    
    # Stack all features
    all_features = np.concatenate(all_features, axis=0)  # [N*T, F]
    
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero
    
    print(f"Feature statistics computed: mean={mean.shape}, std={std.shape}")
    
    return mean, std


if __name__ == '__main__':
    """
    Test dataset loading and preprocessing.
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory with pose sequences')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    args = parser.parse_args()
    
    print("Testing FencingPoseDataset...")
    print()
    
    # Load dataset
    dataset = FencingPoseDataset(args.data_dir)
    
    # Test single sample
    features, label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Features shape: {features.shape}")
    print(f"  Label: {label} ({dataset.get_action_name(label)})")
    
    # Create splits
    train_ds, val_ds, test_ds = create_data_splits(dataset)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, 
        batch_size=args.batch_size
    )
    
    # Test batch loading
    print(f"\nTesting batch loading (batch_size={args.batch_size})...")
    for batch_features, batch_labels in train_loader:
        print(f"  Batch features shape: {batch_features.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
        print(f"  Labels in batch: {[dataset.get_action_name(l.item()) for l in batch_labels]}")
        break
    
    print("\nDataset test passed! ✓")
