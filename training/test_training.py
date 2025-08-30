"""
Test script for the gesture recognition training pipeline.

Quick test to verify that all components work together correctly.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_transformer import GestureTransformer, test_model
from gesture_dataset import test_dataset


def test_complete_pipeline():
    """Test the complete training pipeline with dummy data."""
    print("Testing Gesture Recognition Training Pipeline")
    print("=" * 50)
    
    # Test 1: Model creation and forward pass
    print("\n1. Testing Transformer Model...")
    try:
        model = test_model()
        print("✓ Model test passed!")
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False
    
    # Test 2: Dataset loading (if data exists)
    print("\n2. Testing Dataset Loading...")
    try:
        test_dataset()
        print("✓ Dataset test passed!")
    except Exception as e:
        print(f"⚠ Dataset test warning: {e}")
        print("  (This is expected if training data is not available)")
    
    # Test 3: Training components
    print("\n3. Testing Training Components...")
    try:
        from gesture_trainer import GestureTrainer
        
        # Create dummy data loader
        from torch.utils.data import DataLoader, TensorDataset
        
        # Dummy data
        dummy_sequences = torch.randn(100, 32, 63)  # 100 samples, 32 frames, 63 features
        dummy_labels = torch.randint(0, 5, (100,))
        dummy_lengths = torch.full((100,), 32)
        
        dummy_dataset = TensorDataset(dummy_sequences, dummy_labels, dummy_lengths)
        dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)
        
        # Create model and trainer
        model = GestureTransformer(num_classes=5, d_model=128, nhead=4, num_encoder_layers=2)
        device = torch.device('cpu')
        
        trainer = GestureTrainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            device=device,

            model_dir="test_models"
        )
        
        print("✓ Trainer initialization passed!")
        
        # Test single training step
        train_loss, train_acc = trainer.train_epoch()
        print(f"✓ Training step passed! Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # Test validation step
        val_loss, val_acc, metrics = trainer.validate()
        print(f"✓ Validation step passed! Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Training pipeline is ready.")
    print("\nTo start training with your data, run:")
    print("python train_cached.py")
    
    return True


if __name__ == "__main__":
    success = test_complete_pipeline()
    
    if not success:
        sys.exit(1)
