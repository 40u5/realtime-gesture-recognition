"""
Gesture Recognition Transformer Model

A transformer-based model for recognizing hand gestures from MediaPipe landmark sequences.
This model processes sequences of hand landmarks and classifies them into gesture categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to handle sequence order.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GestureTransformer(nn.Module):
    """
    Transformer model for gesture recognition from MediaPipe hand landmarks.
    
    The model takes sequences of hand landmarks (21 landmarks * 3 coordinates = 63 features)
    and predicts gesture classes.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 64,
        landmark_dim: int = 63  # 21 landmarks * 3 coordinates (x, y, z)
    ):
        """
        Initialize the GestureTransformer.
        
        Args:
            num_classes: Number of gesture classes (default: 5)
            d_model: Dimension of the model (default: 256)
            nhead: Number of attention heads (default: 8)
            num_encoder_layers: Number of transformer encoder layers (default: 6)
            dim_feedforward: Dimension of feedforward network (default: 1024)
            dropout: Dropout rate (default: 0.1)
            max_seq_length: Maximum sequence length (default: 64)
            landmark_dim: Dimension of input landmarks (default: 63)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.landmark_dim = landmark_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(landmark_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # Use seq_len first format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, landmark_dim]
            src_key_padding_mask: Mask for padding tokens [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes]
        """
        # src: [batch_size, seq_len, landmark_dim]
        batch_size, seq_len, _ = src.shape
        
        # Project input to model dimension
        # [batch_size, seq_len, landmark_dim] -> [batch_size, seq_len, d_model]
        src = self.input_projection(src)
        
        # Transpose to [seq_len, batch_size, d_model] for transformer
        src = src.transpose(0, 1)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer encoder
        # src_key_padding_mask: [batch_size, seq_len] (True for padding positions)
        output = self.transformer_encoder(
            src, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Global average pooling over sequence dimension
        # [seq_len, batch_size, d_model] -> [batch_size, d_model]
        if src_key_padding_mask is not None:
            # Mask out padding positions for averaging
            mask = ~src_key_padding_mask.transpose(0, 1).unsqueeze(-1)  # [seq_len, batch_size, 1]
            output = output * mask
            output = output.sum(dim=0) / mask.sum(dim=0)
        else:
            output = output.mean(dim=0)
        
        # Classification
        logits = self.classifier(output)
        
        return logits
    
    def get_attention_weights(
        self, 
        src: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Get attention weights from all transformer layers for visualization.
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, landmark_dim]
            src_key_padding_mask: Mask for padding tokens [batch_size, seq_len]
        
        Returns:
            list: Attention weights from each layer
        """
        # This is a simplified version - in practice, you'd need to modify
        # the transformer encoder to return attention weights
        with torch.no_grad():
            self.eval()
            _ = self.forward(src, src_key_padding_mask)
        return []


class HandLandmarkProcessor:
    """
    Utility class for processing MediaPipe hand landmarks into model input format.
    """
    
    def __init__(self):
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
    
    def landmarks_to_vector(self, landmarks_dict: dict) -> np.ndarray:
        """
        Convert MediaPipe landmarks dictionary to feature vector.
        
        Args:
            landmarks_dict: Dictionary of landmarks from HandDetector
        
        Returns:
            np.ndarray: Feature vector of shape [63] (21 landmarks * 3 coordinates)
        """
        features = []
        
        for landmark_name in self.landmark_names:
            if landmark_name in landmarks_dict:
                landmark = landmarks_dict[landmark_name]
                # Use normalized coordinates and z-depth
                features.extend([
                    landmark['normalized_x'],
                    landmark['normalized_y'],
                    landmark['z']
                ])
            else:
                # Fill with zeros if landmark is missing
                features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def normalize_landmarks(self, landmarks_vector: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position.
        
        Args:
            landmarks_vector: Raw landmarks vector of shape [63]
        
        Returns:
            np.ndarray: Normalized landmarks vector
        """
        # Reshape to [21, 3] for easier processing
        landmarks = landmarks_vector.reshape(21, 3)
        
        # Get wrist position (first landmark)
        wrist_pos = landmarks[0].copy()
        
        # Normalize all landmarks relative to wrist
        normalized_landmarks = landmarks - wrist_pos
        
        # Flatten back to [63]
        return normalized_landmarks.flatten()
    
    def augment_landmarks(self, landmarks_vector: np.ndarray, 
                         rotation_angle: float = 0.0,
                         scale_factor: float = 1.0,
                         noise_std: float = 0.0) -> np.ndarray:
        """
        Apply data augmentation to landmarks.
        
        Args:
            landmarks_vector: Input landmarks vector of shape [63]
            rotation_angle: Rotation angle in radians
            scale_factor: Scaling factor
            noise_std: Standard deviation of Gaussian noise
        
        Returns:
            np.ndarray: Augmented landmarks vector
        """
        # Reshape to [21, 3]
        landmarks = landmarks_vector.reshape(21, 3)
        
        # Apply rotation (only to x, y coordinates)
        if rotation_angle != 0.0:
            cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            xy_coords = landmarks[:, :2]  # [21, 2]
            rotated_xy = xy_coords @ rotation_matrix.T
            landmarks[:, :2] = rotated_xy
        
        # Apply scaling
        if scale_factor != 1.0:
            landmarks *= scale_factor
        
        # Add noise
        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, landmarks.shape)
            landmarks += noise
        
        return landmarks.flatten()


def create_padding_mask(sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Create padding mask for sequences of different lengths.
    
    Args:
        sequences: Tensor of shape [batch_size, max_seq_len, feature_dim]
        lengths: Tensor of shape [batch_size] containing actual sequence lengths
    
    Returns:
        torch.Tensor: Padding mask of shape [batch_size, max_seq_len]
                     (True for padding positions)
    """
    batch_size, max_seq_len = sequences.shape[:2]
    
    # Create range tensor [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, device=sequences.device).expand(batch_size, max_seq_len)
    
    # Create mask: True where position >= length (i.e., padding positions)
    mask = positions >= lengths.unsqueeze(1)
    
    return mask

