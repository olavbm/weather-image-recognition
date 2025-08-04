import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class WeatherCNN(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet18', pretrained=False, dropout_rate=0.5):
        super(WeatherCNN, self).__init__()
        self.backbone = backbone
        
        if backbone == 'resnet18':
            self.features = models.resnet18(pretrained=pretrained)
            feature_dim = self.features.fc.in_features
            self.features = nn.Sequential(*list(self.features.children())[:-1])
        elif backbone == 'resnet34':
            self.features = models.resnet34(pretrained=pretrained)
            feature_dim = self.features.fc.in_features
            self.features = nn.Sequential(*list(self.features.children())[:-1])
        elif backbone == 'resnet50':
            self.features = models.resnet50(pretrained=pretrained)
            feature_dim = self.features.fc.in_features
            self.features = nn.Sequential(*list(self.features.children())[:-1])
        elif backbone == 'resnet101':
            self.features = models.resnet101(pretrained=pretrained)
            feature_dim = self.features.fc.in_features
            self.features = nn.Sequential(*list(self.features.children())[:-1])
        elif backbone == 'efficientnet_b0':
            self.features = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        elif backbone == 'efficientnet_b1':
            self.features = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        elif backbone == 'efficientnet_b2':
            self.features = models.efficientnet_b2(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        elif backbone == 'efficientnet_b3':
            self.features = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        elif backbone == 'efficientnet_b4':
            self.features = models.efficientnet_b4(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        elif backbone == 'efficientnet_b5':
            self.features = models.efficientnet_b5(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        elif backbone == 'efficientnet_b6':
            self.features = models.efficientnet_b6(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        elif backbone == 'efficientnet_b7':
            self.features = models.efficientnet_b7(pretrained=pretrained)
            feature_dim = self.features.classifier[1].in_features
            self.features = self.features.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        if self.backbone.startswith('efficientnet'):
            x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes=11, input_channels=3, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_model(model_type='resnet18', num_classes=11, **kwargs):
    supported_models = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 
        'efficientnet_b6', 'efficientnet_b7'
    ]
    
    if model_type in supported_models:
        return WeatherCNN(num_classes=num_classes, backbone=model_type, **kwargs)
    elif model_type == 'custom':
        return CustomCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    model = create_model('resnet18', num_classes=11)
    print(f"Model: {model}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")