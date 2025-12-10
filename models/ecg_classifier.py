"""
ECG Arrhythmia Classification Model.
Developed by Kuldeep Choksi

1D Convolutional Neural Network for classifying heartbeats
into 5 AAMI classes: Normal, Supraventricular, Ventricular,
Fusion, and Unknown.

Architecture designed for 360-sample (1 second) ECG windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGClassifier(nn.Module):
    """
    1D CNN for ECG beat classification.
    
    Architecture:
        Input: [batch, 1, 360] - single-channel ECG signal
        
        Conv Block 1: 1 -> 32 channels, kernel=7
        Conv Block 2: 32 -> 64 channels, kernel=5
        Conv Block 3: 64 -> 128 channels, kernel=3
        Conv Block 4: 128 -> 256 channels, kernel=3
        
        Global Average Pooling
        Fully Connected: 256 -> 128 -> num_classes
        
    Each conv block: Conv1d -> BatchNorm -> ReLU -> MaxPool -> Dropout
    
    Args:
        num_classes: Number of output classes (default: 5 for AAMI)
        input_size: Length of input signal (default: 360)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_size: int = 360,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Convolutional blocks
        # Block 1: Captures broad features (kernel=7)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Block 2: Medium features (kernel=5)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Block 3: Finer features (kernel=3)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Block 4: Fine details (kernel=3)
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Global average pooling - reduces to [batch, 256, 1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 360]
            
        Returns:
            Logits of shape [batch, num_classes]
        """
        # Convolutional layers
        x = self.conv1(x)  # [batch, 32, 180]
        x = self.conv2(x)  # [batch, 64, 90]
        x = self.conv3(x)  # [batch, 128, 45]
        x = self.conv4(x)  # [batch, 256, 22]
        
        # Global pooling
        x = self.global_pool(x)  # [batch, 256, 1]
        x = x.squeeze(-1)  # [batch, 256]
        
        # Classification
        x = self.fc(x)  # [batch, num_classes]
        
        return x
    
    def predict_proba(self, x):
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x):
        """Get predicted class."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class ECGResidualBlock(nn.Module):
    """Residual block for deeper network."""
    
    def __init__(self, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual  # Skip connection
        out = F.relu(out)
        
        return out


class ECGResNet(nn.Module):
    """
    Deeper ResNet-style model for ECG classification.
    
    Uses residual connections for better gradient flow.
    Better for complex patterns but slower to train.
    """
    
    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Residual blocks
        self.res1 = ECGResidualBlock(64, dropout=dropout)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res2 = ECGResidualBlock(128, dropout=dropout)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.res3 = ECGResidualBlock(256, dropout=dropout)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_in(x)
        
        x = self.res1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.res2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.res3(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        
        return x


# Quick test
if __name__ == '__main__':
    print("="*60)
    print("ECG Classifier Model")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Test basic model
    model = ECGClassifier(num_classes=5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nECGClassifier:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch = torch.randn(8, 1, 360)  # 8 samples, 1 channel, 360 timepoints
    output = model(batch)
    
    print(f"\n  Input shape: {batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output[0].detach().numpy().round(3)}")
    
    # Test predictions
    probs = model.predict_proba(batch)
    preds = model.predict(batch)
    
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Predictions: {preds.tolist()}")
    
    # Test ResNet model
    print("\n" + "-"*60)
    resnet = ECGResNet(num_classes=5)
    resnet_params = sum(p.numel() for p in resnet.parameters())
    print(f"\nECGResNet:")
    print(f"  Total parameters: {resnet_params:,}")
    
    output_res = resnet(batch)
    print(f"  Output shape: {output_res.shape}")