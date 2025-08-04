#!/usr/bin/env python3
"""
PyTorch model evaluation script for weather image classification
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

from dataset import WeatherDataset, get_transforms
from model import create_model

class WeatherClassifierEvaluator:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.classes = checkpoint['classes']
        self.best_acc = checkpoint.get('best_acc', 0.0)
        
        # Create model
        self.model = create_model(
            model_type=self.config.model_type,
            num_classes=len(self.classes),
            pretrained=False,  # Don't need pretrained weights when loading checkpoint
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup transforms
        _, self.transform = get_transforms(input_size=self.config.input_size)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model: {self.config.model_type}")
        print(f"🎯 Classes: {len(self.classes)}")
        print(f"🏆 Best training accuracy: {self.best_acc:.2f}%")
    
    def predict_single_image(self, image_path):
        """Predict a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.config, 'use_amp') and self.config.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor)
            
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = self.classes[predicted_class_idx]
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def evaluate_dataset(self, data_dir, split='val', batch_size=64, num_workers=4):
        """Evaluate on validation or test dataset"""
        # Create full dataset first
        full_dataset = WeatherDataset(data_dir)
        
        # Create train/val split
        indices = list(range(len(full_dataset)))
        labels = [full_dataset.samples[i][1] for i in indices]
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=getattr(self.config, 'val_split', 0.2), 
            stratify=labels, 
            random_state=getattr(self.config, 'seed', 42)
        )
        
        # Use appropriate indices based on split
        if split == 'val':
            subset_indices = val_indices
        else:
            subset_indices = train_indices
            
        # Create dataset with specific indices
        dataset = WeatherDataset(data_dir, transform=self.transform, subset_indices=subset_indices)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\n📊 Evaluating on {split} set ({len(dataset)} samples)...")
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if hasattr(self.config, 'use_amp') and self.config.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  Processed batch {batch_idx}/{len(dataloader)}")
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        
        print(f"\n🎯 {split.upper()} Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        report = classification_report(all_targets, all_preds, target_names=self.classes)
        print(f"\nClassification Report:\n{report}")
        
        return all_targets, all_preds, all_probs, accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'Confusion Matrix - {self.config.model_type}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Confusion matrix saved to {save_path}")
        plt.show()
    
    def analyze_predictions(self, data_dir, split='val', num_samples=20):
        """Analyze predictions on random samples"""
        # Create full dataset first
        full_dataset = WeatherDataset(data_dir)
        
        # Create train/val split
        indices = list(range(len(full_dataset)))
        labels = [full_dataset.samples[i][1] for i in indices]
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=getattr(self.config, 'val_split', 0.2), 
            stratify=labels, 
            random_state=getattr(self.config, 'seed', 42)
        )
        
        # Use appropriate indices based on split
        if split == 'val':
            subset_indices = val_indices
        else:
            subset_indices = train_indices
            
        # Create dataset with specific indices
        dataset = WeatherDataset(data_dir, transform=self.transform, subset_indices=subset_indices)
        
        # Get random samples
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        correct = 0
        results = []
        
        print(f"\n🔍 Analyzing {len(sample_indices)} random samples...")
        
        for i, idx in enumerate(sample_indices):
            image_path, true_class_idx = dataset.samples[idx]
            true_class = self.classes[true_class_idx]
            
            # Predict
            predicted_class, confidence, probs = self.predict_single_image(image_path)
            
            is_correct = predicted_class == true_class
            if is_correct:
                correct += 1
            
            results.append({
                'image_path': image_path,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': is_correct
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(indices)} samples")
        
        accuracy = correct / len(results)
        print(f"\n📈 Random Sample Analysis:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show some examples
        print(f"\n📋 Sample Predictions:")
        for i, result in enumerate(results[:10]):
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {Path(result['image_path']).name}: "
                  f"True: {result['true_class']}, "
                  f"Pred: {result['predicted_class']} "
                  f"({result['confidence']:.3f})")
        
        return results
    
    def predict_directory(self, image_dir, output_file=None):
        """Predict all images in a directory"""
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"🔍 Predicting {len(image_paths)} images from {image_dir}")
        
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                predicted_class, confidence, probs = self.predict_single_image(image_path)
                
                results.append({
                    'image_path': str(image_path),
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Save results if requested
        if output_file:
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image_path', 'predicted_class', 'confidence'])
                writer.writeheader()
                writer.writerows(results)
            print(f"💾 Results saved to {output_file}")
        
        # Show summary
        class_counts = {}
        for result in results:
            cls = result['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"\n📊 Prediction Summary:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count} images ({count/len(results)*100:.1f}%)")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Weather Classification Model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--analyze-samples', type=int, default=20, help='Number of samples to analyze')
    parser.add_argument('--predict-dir', type=str, help='Directory with images to predict')
    parser.add_argument('--output-csv', type=str, help='Output CSV file for predictions')
    parser.add_argument('--save-plots', type=str, help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Create evaluator
    evaluator = WeatherClassifierEvaluator(args.checkpoint)
    
    # Evaluate on dataset
    if Path(args.data_dir).exists():
        y_true, y_pred, y_probs, accuracy = evaluator.evaluate_dataset(
            args.data_dir, args.split, args.batch_size, args.num_workers
        )
        
        # Plot confusion matrix
        save_path = None
        if args.save_plots:
            os.makedirs(args.save_plots, exist_ok=True)
            save_path = Path(args.save_plots) / 'confusion_matrix.png'
        
        evaluator.plot_confusion_matrix(y_true, y_pred, save_path)
        
        # Analyze random samples
        evaluator.analyze_predictions(args.data_dir, args.split, args.analyze_samples)
    
    # Predict directory if specified
    if args.predict_dir and Path(args.predict_dir).exists():
        evaluator.predict_directory(args.predict_dir, args.output_csv)
    
    print("\n🎉 Evaluation completed!")

if __name__ == "__main__":
    main()