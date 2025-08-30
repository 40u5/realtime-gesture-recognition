#!/usr/bin/env python3
"""
Cached training script using cached landmarks
This should be 10-100x more efficient than the original training
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import training components
from gesture_trainer import GestureTrainer
from gesture_transformer import GestureTransformer
from cached_dataset import create_cached_data_loaders

import torch
import argparse

def main():
    """Cached training main function"""
    parser = argparse.ArgumentParser(description='Cached Gesture Recognition Training')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=16,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--max_seq_length', type=int, default=64,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (can be larger due to cached loading)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # Output arguments

    parser.add_argument('--model_dir', type=str, default='../models',
                       help='Directory for model checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save model every N epochs')
    
    # Cache arguments
    parser.add_argument('--train_cache', type=str, default='cache/train',
                       help='Training cache directory')
    parser.add_argument('--val_cache', type=str, default='cache/val',
                       help='Validation cache directory')
    
    args = parser.parse_args()
    
    # Logging removed
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"CACHED TRAINING MODE ACTIVATED!")
    print(f"Using device: {device}")
    
    try:
        # Create cached data loaders using cached landmarks
        print("Loading cached datasets...")
        train_loader, val_loader = create_cached_data_loaders(
            train_cache_dir=args.train_cache,
            val_cache_dir=args.val_cache,
            batch_size=args.batch_size,
            max_sequence_length=args.max_seq_length,
            num_workers=args.num_workers
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"CACHED MODE: No live video processing!")
        
        # Get class weights for balanced training
        class_weights = train_loader.dataset.get_class_weights()
        print(f"Class weights: {class_weights}")
        
        # Create model (larger since we can afford it with cached training)
        model = GestureTransformer(
            num_classes=5,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_length=args.max_seq_length
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer
        trainer = GestureTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler,
            class_weights=class_weights,

            model_dir=args.model_dir
        )
        
        # Start cached training
        print(f"Starting cached training for {args.num_epochs} epochs...")
        trainer.train(args.num_epochs, args.save_every)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
