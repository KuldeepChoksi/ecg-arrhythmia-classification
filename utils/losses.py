"""
Loss functions for ECG Arrhythmia Classification.
Developed by Kuldeep Choksi

Includes Focal Loss for handling class imbalance.
Focal Loss down-weights easy examples and focuses on hard ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Focal Loss reduces the loss contribution from easy examples
    and focuses training on hard negatives.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
        p_t = probability of correct class
        alpha_t = class weight
        gamma = focusing parameter (higher = more focus on hard examples)
    
    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self, 
        alpha: torch.Tensor = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits) of shape [batch, num_classes]
            targets: Ground truth labels of shape [batch]
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Instead of hard labels (0 or 1), uses soft labels:
        - True class: 1 - smoothing
        - Other classes: smoothing / (num_classes - 1)
    
    This regularizes the model and prevents overconfidence.
    
    Args:
        num_classes: Number of classes
        smoothing: Smoothing factor (default: 0.1)
    """
    
    def __init__(self, num_classes: int = 5, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = (-smooth_labels * log_probs).sum(dim=1).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Combines Focal Loss with Label Smoothing.
    
    Args:
        alpha: Class weights
        gamma: Focal loss gamma
        smoothing: Label smoothing factor
        focal_weight: Weight for focal loss component
    """
    
    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        focal_weight: float = 0.7
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth = LabelSmoothingLoss(smoothing=smoothing)
        self.focal_weight = focal_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        smooth_loss = self.smooth(inputs, targets)
        return self.focal_weight * focal_loss + (1 - self.focal_weight) * smooth_loss


# Quick test
if __name__ == '__main__':
    print("="*60)
    print("Loss Functions for ECG Classification")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Test focal loss
    batch_size = 8
    num_classes = 5
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Class weights (inverse frequency)
    alpha = torch.tensor([0.25, 6.86, 2.75, 20.68, 2.71])
    
    # Standard CE
    ce_loss = F.cross_entropy(logits, targets)
    print(f"\nCross Entropy Loss: {ce_loss.item():.4f}")
    
    # Focal Loss
    focal = FocalLoss(alpha=alpha, gamma=2.0)
    fl = focal(logits, targets)
    print(f"Focal Loss (gamma=2.0): {fl.item():.4f}")
    
    # Label Smoothing
    smooth = LabelSmoothingLoss(smoothing=0.1)
    sl = smooth(logits, targets)
    print(f"Label Smoothing Loss: {sl.item():.4f}")
    
    # Combined
    combined = CombinedLoss(alpha=alpha, gamma=2.0, smoothing=0.1)
    cl = combined(logits, targets)
    print(f"Combined Loss: {cl.item():.4f}")