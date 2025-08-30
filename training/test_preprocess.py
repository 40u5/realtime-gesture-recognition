#!/usr/bin/env python3
"""
Test preprocessing script to debug issues
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

print("🔍 Debugging preprocessing...")
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {Path(__file__).parent}")

# Test imports
try:
    from gesture_dataset import HandLandmarkProcessor
    print("✅ gesture_dataset imported")
except Exception as e:
    print(f"❌ gesture_dataset import failed: {e}")

try:
    from hand_detector import HandDetector
    print("✅ hand_detector imported")
except Exception as e:
    print(f"❌ hand_detector import failed: {e}")

# Check data files
data_paths = [
    "../data/train/train",
    "../data/train.csv",
    "../data/val/val", 
    "../data/val.csv"
]

for path in data_paths:
    if Path(path).exists():
        print(f"✅ Found: {path}")
    else:
        print(f"❌ Missing: {path}")

# Test cache directory creation
try:
    cache_dir = Path("cache/test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Cache directory created: {cache_dir}")
    
    # List contents
    if cache_dir.parent.exists():
        print(f"Cache parent contents: {list(cache_dir.parent.iterdir())}")
        
except Exception as e:
    print(f"❌ Cache creation failed: {e}")

print("\n📋 Summary complete!")
