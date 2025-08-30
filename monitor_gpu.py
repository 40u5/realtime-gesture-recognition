#!/usr/bin/env python3
"""
Real-time GPU monitoring script to verify which GPU is being used during training
"""

import torch
import time
import psutil
import subprocess
import threading
import sys

def monitor_gpu_usage():
    """Monitor GPU usage in real-time"""
    print("=== Real-time GPU Monitoring ===")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Check PyTorch GPU usage
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device)
                memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
                
                print(f"🔥 NVIDIA RTX 4060: {memory_allocated:.2f}GB / {memory_total:.2f}GB allocated")
                print(f"   Reserved: {memory_reserved:.2f}GB")
                
                # Create a small tensor to test active usage
                if memory_allocated > 0.1:  # If substantial memory is allocated
                    print("   ✅ ACTIVELY USING RTX 4060!")
                else:
                    print("   💤 RTX 4060 idle")
            
            # Check nvidia-smi for detailed info
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.strip():
                            mem_used, mem_total, gpu_util = line.split(', ')
                            print(f"   GPU {i} Utilization: {gpu_util}% | Memory: {mem_used}MB/{mem_total}MB")
            except:
                pass
            
            print("-" * 60)
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

def test_gpu_intensive_operation():
    """Test GPU with intensive operation"""
    print("🧪 Testing GPU intensive operation...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda')
    print(f"📍 Using device: {device}")
    print(f"📍 Device name: {torch.cuda.get_device_name(device)}")
    
    # Large matrix multiplication that should show up in GPU monitoring
    print("🔥 Starting GPU intensive operation...")
    
    for i in range(10):
        # Create large tensors that will definitely use GPU memory
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        c = torch.mm(a, b)
        
        # Print current GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        print(f"   Iteration {i+1}: GPU Memory: {memory_allocated:.2f}GB")
        
        time.sleep(1)
    
    print("✅ GPU test completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_gpu_intensive_operation()
    else:
        monitor_gpu_usage()
