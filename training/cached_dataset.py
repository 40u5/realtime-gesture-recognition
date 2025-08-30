#!/usr/bin/env python3
"""
Cached dataset loader using preprocessed cached landmarks
This replaces live video processing with cached data for 10-100x speedup
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from typing import Tuple, List

class CachedGestureDataset(Dataset):
    """
    Cached dataset that loads preprocessed landmarks from cache files
    """
    
    def __init__(self, cache_dir: str, max_sequence_length: int = 64, augment: bool = False):
        """
        Initialize cached dataset
        
        Args:
            cache_dir: Directory containing cached .pkl files
            max_sequence_length: Maximum sequence length for padding
            augment: Whether to apply data augmentation during training
        """
        self.cache_dir = Path(cache_dir)
        self.max_sequence_length = max_sequence_length
        self.augment = augment
        
        # Import augmentation processor
        if augment:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from gesture_transformer import HandLandmarkProcessor
            self.landmark_processor = HandLandmarkProcessor()
        
        # Load all cached files
        self.data_files = list(self.cache_dir.glob("*.pkl"))
        
        if len(self.data_files) == 0:
            raise ValueError(f"No cached files found in {cache_dir}")
        
        print(f"📂 Found {len(self.data_files)} cached samples")
        
        # Load all data into memory (efficient since already preprocessed)
        self.samples = []
        self.labels = []
        
        for pkl_file in self.data_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                landmarks = data['landmarks']
                label = data['label']
                
                # Pad or truncate sequence
                if len(landmarks) > max_sequence_length:
                    landmarks = landmarks[:max_sequence_length]
                elif len(landmarks) < max_sequence_length:
                    padding = np.zeros((max_sequence_length - len(landmarks), 63))
                    landmarks = np.vstack([landmarks, padding])
                
                self.samples.append(landmarks.astype(np.float32))
                self.labels.append(label)
                
            except Exception as e:
                print(f"⚠️  Error loading {pkl_file}: {e}")
                continue
        
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        
        print(f"✅ Loaded {len(self.samples)} samples")
        print(f"📏 Shape: {self.samples.shape}")
        
        # Calculate class weights
        unique, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        self.class_weights = torch.FloatTensor([
            total_samples / (len(unique) * counts[i]) 
            for i in range(len(unique))
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx].copy()  # Copy to avoid modifying original
        
        # Apply augmentation if enabled
        if self.augment:
            # Apply random augmentation to each frame in sequence
            for i in range(len(sequence)):
                if np.any(sequence[i] != 0):  # Skip zero-padded frames
                    sequence[i] = self.landmark_processor.augment_landmarks(
                        sequence[i],
                        rotation_angle=np.random.uniform(-0.2, 0.2),  # Random rotation
                        scale_factor=np.random.uniform(0.9, 1.1),    # Random scaling
                        noise_std=0.01                               # Small noise
                    )
        
        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([self.labels[idx]])[0]
        length = torch.LongTensor([self.max_sequence_length])[0]  # Return scalar, not tensor
        
        return sequence, label, length
    
    def get_class_weights(self):
        """Get class weights for balanced training"""
        return self.class_weights

def create_cached_data_loaders(
    train_cache_dir: str,
    val_cache_dir: str,
    batch_size: int = 32,
    max_sequence_length: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create cached data loaders using cached landmarks
    
    Args:
        train_cache_dir: Directory with cached training data
        val_cache_dir: Directory with cached validation data
        batch_size: Batch size
        max_sequence_length: Maximum sequence length
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = CachedGestureDataset(train_cache_dir, max_sequence_length, augment=True)  # Augment training data
    val_dataset = CachedGestureDataset(val_cache_dir, max_sequence_length, augment=False)     # No augment for validation
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the cached dataset
    print("🧪 Testing cached dataset...")
    
    try:
        train_loader, val_loader = create_cached_data_loaders(
            train_cache_dir="cache/train",
            val_cache_dir="cache/val",
            batch_size=32,
            max_sequence_length=64
        )
        
        print(f"✅ Train loader: {len(train_loader)} batches")
        print(f"✅ Val loader: {len(val_loader)} batches")
        
        # Test loading one batch
        sequences, labels, lengths = next(iter(train_loader))
        print(f"✅ Batch shape: {sequences.shape}")
        print(f"✅ Labels shape: {labels.shape}")
        print(f"✅ Lengths shape: {lengths.shape}")
        
        print("🎉 Cached dataset working perfectly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
