"""
Gesture Dataset and DataLoader

Dataset class for loading and preprocessing gesture data with MediaPipe landmarks.
Handles video frame sequences and converts them to landmark features for training.
"""

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys

# Add parent directory to path to import hand_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_detector import HandDetector
from training.gesture_transformer import HandLandmarkProcessor


class GestureVideoDataset(Dataset):
    """
    Dataset for gesture recognition from video sequences.
    
    Loads video frames, extracts MediaPipe landmarks, and prepares them for training.
    """
    
    def __init__(
        self,
        data_dir: str,
        csv_file: str,
        max_sequence_length: int = 64,
        target_fps: int = 10,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
        hand_detector: Optional[HandDetector] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to data directory containing video folders
            csv_file: Path to CSV file with labels
            max_sequence_length: Maximum number of frames per sequence
            target_fps: Target FPS for frame sampling
            transform: Optional transforms for data augmentation
            augment: Whether to apply data augmentation
            hand_detector: Pre-initialized HandDetector instance
        """
        self.data_dir = Path(data_dir)
        self.max_sequence_length = max_sequence_length
        self.target_fps = target_fps
        self.transform = transform
        self.augment = augment
        
        # Load CSV data
        self.df = pd.read_csv(csv_file, sep=';', header=None)
        self.df.columns = ['folder_name', 'gesture_name', 'label']
        
        # Initialize hand detector
        if hand_detector is None:
            self.hand_detector = HandDetector(
                max_num_hands=1,  # Assume single hand for simplicity
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        else:
            self.hand_detector = hand_detector
        
        # Initialize landmark processor
        self.landmark_processor = HandLandmarkProcessor()
        
        # Gesture classes
        self.num_classes = len(self.df['label'].unique())
        self.label_to_name = {
            0: 'Left_Swipe',
            1: 'Right_Swipe', 
            2: 'Stop',
            3: 'Thumbs_Down',
            4: 'Thumbs_Up'
        }
        
        # Filter valid samples
        self._filter_valid_samples()
        
        print(f"Loaded dataset with {len(self.df)} samples and {self.num_classes} classes")
    
    def _filter_valid_samples(self):
        """Filter out samples with missing video folders."""
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            folder_path = self.data_dir / row['folder_name']
            if folder_path.exists() and folder_path.is_dir():
                # Check if folder has any image files
                image_files = list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg'))
                if len(image_files) > 0:
                    valid_indices.append(idx)
                else:
                    print(f"Warning: No images found in {folder_path}")
            else:
                print(f"Warning: Folder not found: {folder_path}")
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Filtered to {len(self.df)} valid samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (landmarks_sequence, label, sequence_length)
        """
        row = self.df.iloc[idx]
        folder_name = row['folder_name']
        label = int(row['label'])
        
        # Load video frames and extract landmarks
        landmarks_sequence = self._load_and_process_video(folder_name)
        
        # Convert to tensor
        landmarks_tensor = torch.tensor(landmarks_sequence, dtype=torch.float32)
        
        return landmarks_tensor, label, landmarks_tensor.shape[0]
    
    def _load_and_process_video(self, folder_name: str) -> np.ndarray:
        """
        Load video frames from folder and extract MediaPipe landmarks.
        
        Args:
            folder_name: Name of the video folder
        
        Returns:
            np.ndarray: Landmark sequence of shape [sequence_length, 63]
        """
        folder_path = self.data_dir / folder_name
        
        # Get all image files and sort them
        image_files = sorted(list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg')))
        
        if len(image_files) == 0:
            # Return zero sequence if no images found
            return np.zeros((1, 63), dtype=np.float32)
        
        # Sample frames based on target FPS
        sampled_files = self._sample_frames(image_files)
        
        landmarks_sequence = []
        
        for image_file in sampled_files:
            try:
                # Load image
                frame = cv2.imread(str(image_file))
                if frame is None:
                    continue
                
                # Process with MediaPipe
                detection_results = self.hand_detector.process_frame(frame)
                
                if (detection_results and 
                    detection_results['hands_detected'] and 
                    len(detection_results['hand_landmarks']) > 0):
                    
                    # Use first detected hand
                    hand_landmarks = detection_results['hand_landmarks'][0]
                    
                    # Convert to feature vector
                    landmarks_vector = self.landmark_processor.landmarks_to_vector(hand_landmarks)
                    
                    # Normalize relative to wrist
                    landmarks_vector = self.landmark_processor.normalize_landmarks(landmarks_vector)
                    
                    # Apply augmentation if enabled
                    if self.augment:
                        landmarks_vector = self._augment_landmarks(landmarks_vector)
                    
                    landmarks_sequence.append(landmarks_vector)
                else:
                    # If no hand detected, use zeros or interpolate
                    if len(landmarks_sequence) > 0:
                        # Use last valid landmarks
                        landmarks_sequence.append(landmarks_sequence[-1].copy())
                    else:
                        # Use zero vector
                        landmarks_sequence.append(np.zeros(63, dtype=np.float32))
            
            except Exception as e:
                print(f"Warning: Error processing {image_file}: {e}")
                continue
        
        if len(landmarks_sequence) == 0:
            # Return zero sequence if no valid landmarks found
            return np.zeros((1, 63), dtype=np.float32)
        
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
        
        # Truncate or pad to max_sequence_length
        landmarks_array = self._adjust_sequence_length(landmarks_array)
        
        return landmarks_array
    
    def _sample_frames(self, image_files: List[Path]) -> List[Path]:
        """
        Sample frames from video to achieve target FPS.
        
        Args:
            image_files: List of image file paths
        
        Returns:
            List of sampled image file paths
        """
        total_frames = len(image_files)
        
        if total_frames <= self.max_sequence_length:
            return image_files
        
        # Calculate sampling step
        step = max(1, total_frames // self.max_sequence_length)
        
        # Sample frames
        sampled_indices = range(0, total_frames, step)[:self.max_sequence_length]
        sampled_files = [image_files[i] for i in sampled_indices]
        
        return sampled_files
    
    def _adjust_sequence_length(self, sequence: np.ndarray) -> np.ndarray:
        """
        Adjust sequence length to max_sequence_length by padding or truncating.
        
        Args:
            sequence: Input sequence of shape [seq_len, 63]
        
        Returns:
            np.ndarray: Adjusted sequence of shape [max_sequence_length, 63]
        """
        seq_len = sequence.shape[0]
        
        if seq_len > self.max_sequence_length:
            # Truncate
            return sequence[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            # Pad with zeros
            padding = np.zeros((self.max_sequence_length - seq_len, 63), dtype=np.float32)
            return np.vstack([sequence, padding])
        else:
            return sequence
    
    def _augment_landmarks(self, landmarks_vector: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to landmarks.
        
        Args:
            landmarks_vector: Input landmarks vector
        
        Returns:
            np.ndarray: Augmented landmarks vector
        """
        # Random rotation (±15 degrees)
        rotation_angle = np.random.uniform(-0.26, 0.26)  # ±15 degrees in radians
        
        # Random scaling (0.9 to 1.1)
        scale_factor = np.random.uniform(0.9, 1.1)
        
        # Random noise
        noise_std = 0.01
        
        return self.landmark_processor.augment_landmarks(
            landmarks_vector, rotation_angle, scale_factor, noise_std
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.
        
        Returns:
            torch.Tensor: Class weights
        """
        label_counts = self.df['label'].value_counts().sort_index()
        total_samples = len(self.df)
        
        # Calculate weights inversely proportional to class frequency
        weights = total_samples / (self.num_classes * label_counts.values)
        
        return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching variable-length sequences.
    
    Args:
        batch: List of (landmarks_sequence, label, sequence_length) tuples
    
    Returns:
        Tuple of (batched_sequences, labels, lengths)
    """
    sequences, labels, lengths = zip(*batch)
    
    # Convert to tensors
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    # Stack sequences (they should all have the same length due to padding)
    sequences = torch.stack(sequences)
    
    return sequences, labels, lengths

def test_dataset():
    """Test the dataset loading functionality."""
    # Paths (adjust these for your setup)
    data_dir = "../data/train/train"
    csv_file = "../data/train.csv"
    
    try:
        # Create dataset
        dataset = GestureVideoDataset(
            data_dir=data_dir,
            csv_file=csv_file,
            max_sequence_length=32
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample_landmarks, sample_label, sample_length = dataset[0]
            print(f"Sample shape: {sample_landmarks.shape}")
            print(f"Sample label: {sample_label}")
            print(f"Sample length: {sample_length}")
            
            # Test data loader
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
            batch_sequences, batch_labels, batch_lengths = next(iter(loader))
            
            print(f"Batch sequences shape: {batch_sequences.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            print(f"Batch lengths: {batch_lengths}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()
