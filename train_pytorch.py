#!/usr/bin/env python3
"""
Clean PyTorch training script for weather image classification
Optimized for single GPU (RTX 3090) training with best practices
"""
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import WeatherDataset, get_transforms
from model import create_model

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(device='cuda') if config.use_amp else None
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        print(f"🚀 Starting training on {self.device}")
        print(f"📁 Output directory: {self.output_dir}")
        
    def setup_logging(self):
        """Setup simple logging to file and console"""
        import logging
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def create_data_loaders(self):
        """Create training and validation data loaders"""
        # Get transforms
        train_transform, val_transform = get_transforms(input_size=self.config.input_size)
        
        # Create full dataset first
        full_dataset = WeatherDataset(self.config.data_dir)
        
        # Create train/val split while preserving class distribution
        indices = list(range(len(full_dataset)))
        labels = [full_dataset.samples[i][1] for i in indices]
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=self.config.val_split, 
            stratify=labels, 
            random_state=self.config.seed
        )
        
        # Create datasets with specific indices
        train_dataset = WeatherDataset(self.config.data_dir, transform=train_transform, subset_indices=train_indices)
        val_dataset = WeatherDataset(self.config.data_dir, transform=val_transform, subset_indices=val_indices)
        
        # Create data loaders with optimized settings for RTX 3090
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=2 if self.config.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=2 if self.config.num_workers > 0 else None
        )
        
        self.num_classes = len(full_dataset.classes)
        self.classes = full_dataset.classes
        
        self.logger.info(f"Dataset info:")
        self.logger.info(f"  Classes: {self.classes}")
        self.logger.info(f"  Train samples: {len(train_dataset)}")
        self.logger.info(f"  Val samples: {len(val_dataset)}")
        self.logger.info(f"  Batch size: {self.config.batch_size}")
        self.logger.info(f"  Workers: {self.config.num_workers}")
        
        return train_loader, val_loader
    
    def create_model(self):
        """Create and setup model"""
        model = create_model(
            model_type=self.config.model_type,
            num_classes=self.num_classes,
            pretrained=True,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Enable optimizations for RTX 3090
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
        
        # Use channels_last memory format for better performance
        if self.config.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile') and self.config.compile_model:
            self.logger.info("🔥 Compiling model with torch.compile")
            model = torch.compile(model)
        
        return model
    
    def create_optimizer_scheduler(self, model):
        """Create optimizer and scheduler"""
        # Optimizer
        if self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        else:  # adam
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        
        # Scheduler
        if self.config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs, eta_min=self.config.lr * 0.01
            )
        elif self.config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.step_size, gamma=self.config.gamma
            )
        elif self.config.scheduler == 'reduce_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        
        # Metrics
        running_loss = 0.0
        num_samples = 0
        num_correct = 0
        
        # Progress tracking
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move to device and convert to channels_last if enabled
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.config.channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)
            
            if self.config.use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            _, predicted = outputs.max(1)
            num_correct += predicted.eq(targets).sum().item()
            
            # Progress report
            if batch_idx % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                samples_per_sec = num_samples / elapsed if elapsed > 0 else 0
                
                self.logger.info(
                    f"Epoch {epoch:3d} [{batch_idx:4d}/{len(train_loader):4d}] "
                    f"Loss: {loss.item():.6f} "
                    f"Acc: {100. * num_correct / num_samples:.2f}% "
                    f"({samples_per_sec:.1f} samples/sec)"
                )
        
        epoch_loss = running_loss / num_samples
        epoch_acc = 100. * num_correct / num_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        
        running_loss = 0.0
        num_samples = 0
        num_correct = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                # Move to device
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.config.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                
                # Forward pass
                if self.config.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                num_samples += images.size(0)
                
                _, predicted = outputs.max(1)
                num_correct += predicted.eq(targets).sum().item()
                
                # Collect predictions for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / num_samples
        epoch_acc = 100. * num_correct / num_samples
        
        return epoch_loss, epoch_acc, all_preds, all_targets
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': best_acc,
            'config': self.config,
            'classes': self.classes
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 New best model saved: {best_path}")
        
        # Keep only last few checkpoints to save space
        if self.config.keep_checkpoints > 0:
            checkpoints = sorted(self.output_dir.glob("checkpoint_epoch_*.pth"))
            if len(checkpoints) > self.config.keep_checkpoints:
                for old_checkpoint in checkpoints[:-self.config.keep_checkpoints]:
                    old_checkpoint.unlink()
    
    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
        ax1.plot(epochs, val_losses, label='Val Loss', color='red', alpha=0.7)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, train_accs, label='Train Acc', color='blue', alpha=0.7)
        ax2.plot(epochs, val_accs, label='Val Acc', color='red', alpha=0.7)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop"""
        self.logger.info("🔥 Starting training...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()
        
        # Create model
        model = self.create_model()
        self.logger.info(f"📊 Model: {self.config.model_type}")
        self.logger.info(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer and scheduler
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer, scheduler = self.create_optimizer_scheduler(model)
        
        # Training tracking
        best_acc = 0.0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            if scheduler:
                if self.config.scheduler == 'reduce_plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Track metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Check if best model
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(model, optimizer, scheduler, epoch, best_acc, is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch:3d}/{self.config.epochs} "
                f"Train: {train_loss:.4f}/{train_acc:.2f}% "
                f"Val: {val_loss:.4f}/{val_acc:.2f}% "
                f"Best: {best_acc:.2f}% "
                f"LR: {lr:.2e} "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            if hasattr(self.config, 'patience') and self.config.patience > 0:
                if epoch - best_acc > self.config.patience:
                    self.logger.info(f"Early stopping after {epoch} epochs")
                    break
        
        total_time = time.time() - start_time
        self.logger.info(f"🎉 Training completed in {total_time/60:.1f} minutes")
        self.logger.info(f"🏆 Best validation accuracy: {best_acc:.4f}%")
        
        # Final evaluation and plots
        self.plot_training_curves(train_losses, train_accs, val_losses, val_accs)
        self.plot_confusion_matrix(val_targets, val_preds)
        
        # Detailed classification report
        report = classification_report(val_targets, val_preds, target_names=self.classes)
        self.logger.info(f"\nFinal Classification Report:\n{report}")
        
        # Save report to file
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Best Validation Accuracy: {best_acc:.4f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        return best_acc

def parse_args():
    parser = argparse.ArgumentParser(description='Weather Image Classification Training')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='dataset',
                      help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory for models and logs')
    parser.add_argument('--val-split', type=float, default=0.2,
                      help='Validation split ratio')
    parser.add_argument('--input-size', type=int, default=224,
                      help='Input image size')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='resnet50',
                      choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'],
                      help='Model architecture')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                      help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                      choices=['adam', 'adamw', 'sgd'],
                      help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['cosine', 'step', 'reduce_plateau', 'none'],
                      help='Learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=30,
                      help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                      help='Gamma for StepLR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                      help='Label smoothing factor')
    
    # Optimization arguments
    parser.add_argument('--use-amp', action='store_true',
                      help='Use Automatic Mixed Precision')
    parser.add_argument('--channels-last', action='store_true',
                      help='Use channels_last memory format')
    parser.add_argument('--compile-model', action='store_true',
                      help='Compile model with torch.compile (PyTorch 2.0+)')
    parser.add_argument('--num-workers', type=int, default=16,
                      help='Number of dataloader workers')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=50,
                      help='Logging interval (batches)')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='Save interval (epochs)')
    parser.add_argument('--keep-checkpoints', type=int, default=3,
                      help='Number of checkpoints to keep')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = False  # Keep False for performance
    torch.backends.cudnn.benchmark = True       # True for performance with consistent input sizes

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create trainer and start training
    trainer = Trainer(args)
    best_acc = trainer.train()
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}%")
    print(f"📁 Results saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()
