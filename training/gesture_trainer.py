"""
Gesture Recognition Model Training Script

Main training script for the transformer-based gesture recognition model.
Handles training loop, validation, and model saving.
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import classification_report

# TensorBoard removed for simplicity

from gesture_transformer import GestureTransformer, create_padding_mask


class GestureTrainer:
    """
    Trainer class for gesture recognition model.
    """
    
    def __init__(
        self,
        model: GestureTransformer,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_type: str = "plateau",
        class_weights: Optional[torch.Tensor] = None,

        model_dir: str = "models"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GestureTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_type: Type of learning rate scheduler
            class_weights: Class weights for balanced training

            model_dir: Directory for saving models
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Mixed precision training
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # Model directory
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Logging setup (simplified)
        self.writer = None
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Gesture class names
        self.class_names = [
            'Left_Swipe', 'Right_Swipe', 'Stop', 'Thumbs_Down', 'Thumbs_Up'
        ]
        
        print(f"Trainer initialized. Model has {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress tracking
        total_batches = len(self.train_loader)
        
        for batch_idx, (sequences, labels, lengths) in enumerate(self.train_loader):
            # Move to device with non_blocking for faster transfer
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            
            # Create padding mask
            padding_mask = create_padding_mask(sequences, lengths)
            padding_mask = padding_mask.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(sequences, padding_mask)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(sequences, padding_mask)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress reporting with GPU monitoring
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                current_acc = 100.0 * correct / total if total > 0 else 0
                
                # GPU memory monitoring
                gpu_memory = 0.0
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(self.device) / 1e9
                
                progress_pct = (batch_idx + 1) / total_batches * 100
                print(f"    Batch [{batch_idx+1:3d}/{total_batches}] ({progress_pct:5.1f}%) | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {current_acc:5.1f}% | "
                      f"GPU: {gpu_memory:.2f}GB")
            
            # Batch progress logging removed for simplicity
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy, detailed_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels, lengths in self.val_loader:
                # Move to device
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                
                # Create padding mask
                padding_mask = create_padding_mask(sequences, lengths)
                padding_mask = padding_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, padding_mask)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate detailed metrics
        detailed_metrics = {
            'classification_report': classification_report(
                all_labels, all_predictions,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
        }
        
        return avg_loss, accuracy, detailed_metrics
    
    def save_model(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filename: Filename for the checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        filepath = self.model_dir / filename
        torch.save(checkpoint, filepath)
        
        print(f"Model saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            bool: True if successfully loaded
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accs = checkpoint.get('train_accs', [])
            self.val_accs = checkpoint.get('val_accs', [])
            
            print(f"Checkpoint loaded: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
        
    def train(self, num_epochs: int, save_every: int = 10):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save model every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, detailed_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # TensorBoard logging removed for simplicity
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
            
            # Print epoch results with GPU info
            epoch_time = time.time() - start_time
            gpu_memory = torch.cuda.memory_allocated(self.device) / 1e9 if torch.cuda.is_available() else 0.0
            
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"GPU: {gpu_memory:.2f}GB | "
                f"Time: {epoch_time:.2f}s | "
                f"{'BEST' if is_best else ''}"
            )
            
            # Print detailed metrics for best model
            if is_best:
                from sklearn.metrics import classification_report as sklearn_report
                
                # Get predictions and labels from validation
                all_predictions = []
                all_labels = []
                with torch.no_grad():
                    for sequences, labels, lengths in self.val_loader:
                        sequences = sequences.to(self.device)
                        labels = labels.to(self.device)
                        lengths = lengths.to(self.device)
                        
                        padding_mask = create_padding_mask(sequences, lengths).to(self.device)
                        outputs = self.model(sequences, padding_mask)
                        _, predicted = outputs.max(1)
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                print("\nClassification Report:")
                print(sklearn_report(
                    all_labels, all_predictions,
                    target_names=self.class_names,
                    zero_division=0
                ))
            
            # Only save best model (no intermediate checkpoints)
            if is_best:
                self.save_model(f"best_model.pth", is_best=is_best)
        
        # Final model save
        self.save_model("final_model.pth")
        
        # Final validation to get the final model accuracy
        final_val_loss, final_val_acc, final_metrics = self.validate()
        
        # Training completed - display summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"Best Model Accuracy:  {self.best_val_acc:.2f}%")
        print(f"Final Model Accuracy: {final_val_acc:.2f}%")
        print("="*60)
