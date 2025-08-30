# Gesture Recognition Training Module

This module contains the complete training pipeline for the transformer-based gesture recognition model using MediaPipe hand landmarks.

## Overview

The training system consists of several components:

1. **GestureTransformer** - A transformer-based neural network for sequence classification
2. **GestureVideoDataset** - Dataset loader that processes video sequences and extracts MediaPipe landmarks
3. **GestureTrainer** - Training loop with validation, metrics, and model saving
4. **HandLandmarkProcessor** - Utility for processing MediaPipe landmarks

## Features

- **Transformer Architecture**: Uses multi-head attention to process temporal sequences of hand landmarks
- **MediaPipe Integration**: Automatically extracts hand landmarks from video frames
- **Data Augmentation**: Applies rotation, scaling, and noise to landmarks for better generalization
- **Sequence Handling**: Handles variable-length sequences with padding and masking
- **Class Balancing**: Automatic class weight calculation for balanced training
- **Monitoring**: Detailed metrics and progress tracking
- **Checkpointing**: Automatic model saving and resuming

## Files

- `gesture_transformer.py` - Transformer model implementation
- `gesture_dataset.py` - Dataset and data loading utilities
- `gesture_trainer.py` - GestureTrainer class for training logic
- `test_training.py` - Test script to verify the pipeline

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Test the Pipeline

```bash
python test_training.py
```

### 3. Start Training

```bash
python train_cached.py
```

## Training Arguments

### Data Arguments
- `--train_data_dir`: Path to training data directory (default: ../data/train/train)
- `--val_data_dir`: Path to validation data directory (default: ../data/val/val)
- `--train_csv`: Path to training CSV file (default: ../data/train.csv)
- `--val_csv`: Path to validation CSV file (default: ../data/val.csv)

### Model Arguments
- `--d_model`: Model dimension (default: 256)
- `--nhead`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 6)
- `--dropout`: Dropout rate (default: 0.1)
- `--max_seq_length`: Maximum sequence length (default: 64)

### Training Arguments
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--scheduler`: Learning rate scheduler (plateau/cosine/none, default: plateau)

### System Arguments
- `--device`: Device to use (cuda/cpu/auto, default: auto)
- `--num_workers`: Number of data loading workers (default: 4)
- `--resume`: Path to checkpoint to resume from

### Output Arguments

- `--model_dir`: Directory for model checkpoints (default: models)
- `--save_every`: Save model every N epochs (default: 10)

## Data Format

The training expects the following data structure:

```
data/
├── train/
│   ├── train/
│   │   ├── gesture_video_1/
│   │   │   ├── frame_001.png
│   │   │   ├── frame_002.png
│   │   │   └── ...
│   │   └── gesture_video_2/
│   │       ├── frame_001.png
│   │       └── ...
│   └── train.csv
└── val/
    ├── val/
    │   ├── gesture_video_3/
    │   └── ...
    └── val.csv
```

CSV format: `folder_name;gesture_name;label`

## Gesture Classes

The model recognizes 5 gesture classes:
- 0: Left_Swipe
- 1: Right_Swipe
- 2: Stop
- 3: Thumbs_Down
- 4: Thumbs_Up

## Model Architecture

- **Input**: Sequences of MediaPipe hand landmarks (21 landmarks × 3 coordinates = 63 features)
- **Embedding**: Linear projection to model dimension
- **Positional Encoding**: Sinusoidal positional encoding for sequence order
- **Transformer Encoder**: Multi-layer transformer with self-attention
- **Classification Head**: Multi-layer perceptron for gesture classification

## Output

Training produces:
- `models/best_model.pth` - Best model checkpoint
- `models/final_model.pth` - Final model checkpoint
- `models/epoch_N.pth` - Periodic checkpoints



## Monitoring

Training progress is displayed via console output during training.

## Example Training Command

```bash
python train_cached.py \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --d_model 512 \
    --nhead 16 \
    --num_layers 8 \
    --device auto \
    --model_dir ../models
```

## Advanced Usage

### Resume Training

```bash
python train_cached.py --resume ../models/epoch_50.pth
```

### Custom Model Configuration

```bash
python train_cached.py \
    --d_model 512 \
    --nhead 16 \
    --num_layers 8 \
    --dropout 0.2 \
    --max_seq_length 128
```

### Performance Tips

1. **GPU Training**: Use `--device cuda` for GPU-accelerated training
2. **Batch Size**: Increase batch size if you have more GPU memory
3. **Workers**: Adjust `--num_workers` based on your CPU cores
4. **Sequence Length**: Reduce `--max_seq_length` for more efficient training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **Slow Data Loading**: Reduce num_workers or check storage speed
3. **Poor Convergence**: Try different learning rates or model architectures
4. **MediaPipe Warnings**: These are normal and don't affect training

### Performance Monitoring

- Watch validation accuracy for overfitting
- Monitor learning rate scheduling
- Check class balance in logs
- Monitor console output for detailed metrics

## Integration with Main Project

To use the trained model in the main gesture recognition system:

1. Train and save the model using this pipeline
2. Load the model in your inference script
3. Use the same MediaPipe landmark extraction
4. Apply the same preprocessing (normalization, etc.)

The model outputs logits that can be converted to probabilities using softmax and then to gesture predictions using argmax.
