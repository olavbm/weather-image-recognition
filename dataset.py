import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split

class WeatherDataset(Dataset):
    def __init__(self, data_dir, transform=None, subset_indices=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_name), 
                                       self.class_to_idx[class_name]))
        
        if subset_indices is not None:
            self.samples = [self.samples[i] for i in subset_indices]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

def get_transforms(input_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(data_dir, batch_size=32, val_split=0.2, input_size=224, num_workers=4):
    dataset = WeatherDataset(data_dir)
    
    # Create train/val split while preserving class distribution
    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=42
    )
    
    train_transform, val_transform = get_transforms(input_size)
    
    train_dataset = WeatherDataset(data_dir, transform=train_transform, subset_indices=train_indices)
    val_dataset = WeatherDataset(data_dir, transform=val_transform, subset_indices=val_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, classes = create_data_loaders("dataset/", batch_size=16)
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
        if batch_idx == 0:
            break