"""
Training script for ECG Arrhythmia Classification.
Developed by Kuldeep Choksi

V2: Added focal loss and oversampling for better rare class detection.

Usage:
    python train.py --epochs 50 --focal --oversample
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
from utils.dataset_v2 import MITBIHDataset, get_train_val_test_split
from utils.losses import FocalLoss, CombinedLoss


CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
               'Fusion (F)', 'Unknown (Q)']


def compute_class_weights(dataset):
    """Compute class weights inversely proportional to frequency."""
    from collections import Counter
    
    counts = Counter(dataset.labels)
    total = len(dataset.labels)
    
    weights = []
    for i in range(5):
        count = counts.get(i, 1)
        weight = total / (5 * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = [0] * 5
    class_total = [0] * 5
    
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
        
        # Track per-class accuracy
        for i in range(5):
            mask = labels == i
            class_total[i] += mask.sum().item()
            class_correct[i] += (predicted[mask] == i).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model with detailed per-class metrics."""
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
    
    avg_loss = total_loss / len(val_loader)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100 * np.mean(all_preds == all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class recall
    per_class_recall = []
    for i in range(5):
        mask = all_labels == i
        if mask.sum() > 0:
            recall = (all_preds[mask] == i).mean()
        else:
            recall = 0
        per_class_recall.append(recall)
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_preds, all_labels, per_class_recall


def train(args):
    """Main training function."""
    
    print("="*60)
    print("ECG ARRHYTHMIA CLASSIFICATION TRAINING V2")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    if args.focal:
        print("\n*** Using FOCAL LOSS ***")
    if args.oversample:
        print("*** Using OVERSAMPLING ***")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("\nUsing Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")
    
    # Load data
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset = get_train_val_test_split(
        data_dir=args.data_dir,
        window_size=args.window_size,
        oversample_train=args.oversample
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    if args.model == 'cnn':
        model = ECGClassifier(num_classes=5, dropout=args.dropout)
    else:
        model = ECGResNet(num_classes=5, dropout=args.dropout)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss function
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy().round(2)}")
    
    if args.focal:
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
        print(f"Using Focal Loss (gamma={args.gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Cross Entropy Loss")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Scheduler - reduce LR when F1-macro plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if args.focal:
        suffix += "_focal"
    if args.oversample:
        suffix += "_oversample"
    results_dir = Path(args.results_dir) / f"ecg_{args.model}{suffix}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-"*60)
    
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': [],
        'per_class_recall': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, f1_macro, f1_weighted, _, _, per_class_recall = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(f1_macro)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(f1_macro)
        history['val_f1_weighted'].append(f1_weighted)
        history['per_class_recall'].append(per_class_recall)
        
        # Print progress with per-class recall
        recall_str = " ".join([f"{r*100:.0f}%" for r in per_class_recall])
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {train_acc:.1f}%/{val_acc:.1f}% | "
              f"F1: {f1_macro:.4f} | "
              f"Recall[N,S,V,F,Q]: [{recall_str}]")
        
        # Save best model based on F1-macro
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'per_class_recall': per_class_recall,
                'args': vars(args),
                'class_weights': class_weights.cpu()
            }, results_dir / 'best_model.pth')
            
            print(f"  -> Saved best model (F1-Macro: {f1_macro:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    print("-"*60)
    print(f"Training complete!")
    print(f"Best F1-Macro: {best_f1:.4f} at epoch {best_epoch}")
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(results_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1_macro, test_f1_weighted, preds, labels, test_recall = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1-Macro: {test_f1_macro:.4f}")
    print(f"  F1-Weighted: {test_f1_weighted:.4f}")
    
    print(f"\nPer-Class Recall:")
    for i, (name, recall) in enumerate(zip(CLASS_NAMES, test_recall)):
        print(f"  {name}: {recall*100:.1f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Save results
    torch.save({
        'history': history,
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'test_recall': test_recall,
        'confusion_matrix': cm,
        'predictions': preds,
        'labels': labels,
        'args': vars(args)
    }, results_dir / 'results.pth')
    
    print(f"\nResults saved to: {results_dir}")
    
    return model, history, results_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ECG arrhythmia classifier V2')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data/mitbih')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'])
    parser.add_argument('--window-size', type=int, default=360)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15)
    
    # V2 improvements
    parser.add_argument('--focal', action='store_true', help='Use focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--oversample', action='store_true', help='Oversample rare classes')
    
    # Output
    parser.add_argument('--results-dir', type=str, default='./results')
    
    args = parser.parse_args()
    train(args)