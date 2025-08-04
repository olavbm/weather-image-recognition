#!/usr/bin/env python3
"""
RTX 3090 optimized training launcher for weather image classification
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch training with RTX 3090 optimized settings"""
    
    # Check if dataset exists
    if not Path("dataset").exists():
        print("❌ Dataset directory not found. Please ensure 'dataset' directory exists.")
        return
    
    # RTX 3090 optimized settings
    cmd = [
        sys.executable, "train_pytorch.py",
        
        # Data settings
        "--data-dir", "dataset",
        "--output-dir", "rtx3090_training",
        "--input-size", "224",
        "--val-split", "0.2",
        
        # Model settings  
        "--model-type", "efficientnet_b4",  # Good balance of performance and speed
        "--dropout-rate", "0.5",
        
        # Training settings optimized for RTX 3090
        "--epochs", "10",
        "--batch-size", "64",  # RTX 3090 can handle large batches
        "--lr", "1e-3",
        "--weight-decay", "1e-4",
        "--optimizer", "adamw",
        "--scheduler", "cosine",
        "--label-smoothing", "0.1",
        
        # RTX 3090 optimizations (disabled problematic ones for now)
        "--use-amp",           # Mixed precision for speed
        "--channels-last",     # Better memory layout
        "--compile-model",     # PyTorch 2.0 compilation
        "--num-workers", "8",  # Good for RTX 3090 system
        
        # Logging
        "--log-interval", "25",
        "--save-interval", "10",
        "--keep-checkpoints", "5",
        
        # Reproducibility
        "--seed", "42"
    ]
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 Training completed successfully!")
        print("📁 Results saved in 'rtx3090_training' directory")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")

if __name__ == "__main__":
    main()
