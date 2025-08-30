#!/usr/bin/env python3
"""
Preprocess all videos to extract and cache MediaPipe landmarks
This will dramatically speed up training by avoiding live video processing
"""

import os
import json
import pickle
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from gesture_dataset import HandLandmarkProcessor
from hand_detector import HandDetector

def preprocess_dataset(data_dir, csv_file, output_dir, max_sequence_length=64):
    """
    Preprocess videos to extract landmarks and save to cache files
    
    Args:
        data_dir: Directory containing video folders
        csv_file: CSV file with video labels
        output_dir: Directory to save preprocessed landmarks
        max_sequence_length: Maximum sequence length
    """
    
    print(f"🔄 Preprocessing dataset: {data_dir}")
    print(f"📋 Labels from: {csv_file}")
    print(f"💾 Output to: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV labels
    df = pd.read_csv(csv_file, sep=';', header=None, names=['folder', 'gesture', 'label'])
    print(f"📊 Found {len(df)} video entries")
    
    # Initialize MediaPipe components
    hand_detector = HandDetector()
    landmark_processor = HandLandmarkProcessor()
    
    # Process each video
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        folder_name = row['folder']
        label = row['label']
        
        # Path to video folder
        video_folder = Path(data_dir) / folder_name
        
        if not video_folder.exists():
            print(f"⚠️  Video folder not found: {video_folder}")
            failed += 1
            continue
        
        # Output file for this video's landmarks
        output_file = output_dir / f"{folder_name}.pkl"
        
        # Skip if already processed
        if output_file.exists():
            print(f"⏭️  Skipping (already cached): {folder_name}")
            continue
        
        try:
            # Get all image files
            image_files = sorted([
                f for f in video_folder.glob("*.png") 
                if f.is_file()
            ])
            
            if len(image_files) == 0:
                print(f"⚠️  No images found in: {video_folder}")
                failed += 1
                continue
            
            # Sample frames if needed
            if len(image_files) > max_sequence_length:
                step = len(image_files) / max_sequence_length
                sampled_indices = [int(i * step) for i in range(max_sequence_length)]
                image_files = [image_files[i] for i in sampled_indices]
            
            # Process each frame
            landmarks_sequence = []
            
            for image_file in image_files:
                # Load image
                frame = cv2.imread(str(image_file))
                if frame is None:
                    continue
                
                # Process with MediaPipe
                detection_results = hand_detector.process_frame(frame)
                
                if (detection_results and 
                    detection_results['hands_detected'] and 
                    len(detection_results['hand_landmarks']) > 0):
                    
                    # Use first detected hand
                    hand_landmarks = detection_results['hand_landmarks'][0]
                    
                    # Convert to feature vector
                    landmarks_vector = landmark_processor.landmarks_to_vector(hand_landmarks)
                    
                    # IMPORTANT: Normalize landmarks relative to wrist position
                    landmarks_vector = landmark_processor.normalize_landmarks(landmarks_vector)
                    
                    landmarks_sequence.append(landmarks_vector)
                else:
                    # If no hand detected, use zero vector
                    landmarks_sequence.append(np.zeros(63))
            
            if len(landmarks_sequence) == 0:
                print(f"⚠️  No valid landmarks extracted from: {video_folder}")
                failed += 1
                continue
            
            # Convert to numpy array
            landmarks_array = np.array(landmarks_sequence)
            
            # Save preprocessed data
            data = {
                'landmarks': landmarks_array,
                'label': label,
                'gesture': row['gesture'],
                'folder': folder_name,
                'sequence_length': len(landmarks_sequence)
            }
            
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            successful += 1
            
        except Exception as e:
            print(f"❌ Error processing {video_folder}: {e}")
            failed += 1
    
    # Cleanup
    hand_detector.cleanup()
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Cache directory: {output_dir}")

def create_cached_dataset():
    """Create preprocessed datasets for cached training"""
    
    print("🚀 CREATING CACHED TRAINING DATASETS")
    print("=" * 50)
    
    # Preprocess training data
    print("\n1️⃣ Preprocessing training data...")
    preprocess_dataset(
        data_dir="../data/train/train",
        csv_file="../data/train.csv", 
        output_dir="cache/train",
        max_sequence_length=64
    )
    
    # Preprocess validation data
    print("\n2️⃣ Preprocessing validation data...")
    preprocess_dataset(
        data_dir="../data/val/val",
        csv_file="../data/val.csv",
        output_dir="cache/val", 
        max_sequence_length=64
    )
    
    print(f"\n🎉 ALL DONE!")
    print(f"Now training will be 10-100x more efficient!")
    print(f"Use: python train_cached.py --use_cache")

if __name__ == "__main__":
    create_cached_dataset()
