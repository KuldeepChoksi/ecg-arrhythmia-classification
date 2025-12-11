"""
Training script for ECG Arrhythmia Classification.
Developed by Kuldeep Choksi

V3: Uses stratified splitting for fair class representation.

Usage:
    python train_v3.py --epochs 50
    python train_v3.py --epochs 50 --focal --oversample
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from models import ECGClassifier, ECGResNet
from utils.dataset_v3 import get_stratified_split
from utils.losses import FocalLoss


CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
               'Fusion (F)', 'Unknown (Q)']


def compute_class_weights(labels):
    """Compute class weights from labels array."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (5 * counts.get(i, 1)) for i in range(5)]
    return torch.FloatTensor(weights)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        signals = batch['signal'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            signals = batch['signal'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100 * np.mean(all_preds == all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class recall
    per_class_recall = []
    for i in range(5):
        mask = all_labels == i
        recall = (all_preds[mask] == i).mean() if mask.sum() > 0 else 0
        per_class_recall.append(recall)
    
    return total_loss / len(val_loader), accuracy, f1_macro, f1_weighted, all_preds, all_labels, per_class_recall


def train(args):
    print("="*60)
    print("ECG ARRHYTHMIA CLASSIFICATION - V3 (Stratified)")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    if args.focal:
        print("\n*** Using FOCAL LOSS ***")
    if args.oversample:
        print("*** Using OVERSAMPLING ***")
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("\nUsing Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")
    
    # Load data with stratified split
    train_dataset, val_dataset, test_dataset = get_stratified_split(
        data_dir=args.data_dir,
        window_size=args.window_size,
        oversample_train=args.oversample
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    print(f"\nCreating model: {args.model}")
    if args.model == 'cnn':
        model = ECGClassifier(num_classes=5, dropout=args.dropout)
    else:
        model = ECGResNet(num_classes=5, dropout=args.dropout)
    
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    class_weights = compute_class_weights(train_dataset.labels).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy().round(2)}")
    
    if args.focal:
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
        print(f"Using Focal Loss (gamma={args.gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Cross Entropy Loss")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    
    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_stratified"
    if args.focal:
        suffix += "_focal"
    if args.oversample:
        suffix += "_oversample"
    results_dir = Path(args.results_dir) / f"ecg_{args.model}{suffix}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print("-"*60)
    
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, f1_macro, f1_weighted, _, _, per_class_recall = validate(model, val_loader, criterion, device)
        
        scheduler.step(f1_macro)
        
        recall_str = " ".join([f"{r*100:.0f}%" for r in per_class_recall])
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {train_acc:.1f}%/{val_acc:.1f}% | "
              f"F1: {f1_macro:.4f} | "
              f"Recall[N,S,V,F,Q]: [{recall_str}]")
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'f1_macro': f1_macro,
                'per_class_recall': per_class_recall,
                'args': vars(args)
            }, results_dir / 'best_model.pth')
            
            print(f"  -> Saved best model (F1: {f1_macro:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("-"*60)
    print(f"Best F1-Macro: {best_f1:.4f} at epoch {best_epoch}")
    
    # Test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(results_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_acc, test_f1_macro, test_f1_weighted, preds, labels, test_recall = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test F1-Macro: {test_f1_macro:.4f}")
    print(f"Test F1-Weighted: {test_f1_weighted:.4f}")
    
    print(f"\nPer-Class Recall:")
    for i, (name, recall) in enumerate(zip(CLASS_NAMES, test_recall)):
        print(f"  {name}: {recall*100:.1f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    # Save results
    torch.save({
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_recall': test_recall,
        'confusion_matrix': confusion_matrix(labels, preds),
        'predictions': preds,
        'labels': labels
    }, results_dir / 'results.pth')
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/mitbih')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'])
    parser.add_argument('--window-size', type=int, default=360)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--focal', action='store_true')
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--results-dir', type=str, default='./results')
    
    args = parser.parse_args()
    train(args)