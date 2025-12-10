"""
Training script for ECG Arrhythmia Classification.
Developed by Kuldeep Choksi

Trains a 1D CNN to classify heartbeats into 5 AAMI classes.
Handles class imbalance with weighted loss.

Usage:
    python train.py --epochs 50
    python train.py --model resnet --epochs 100
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
from utils import MITBIHDataset, get_train_val_test_split


CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
               'Fusion (F)', 'Unknown (Q)']


def compute_class_weights(dataset):
    """
    Compute class weights inversely proportional to class frequency.
    This helps the model pay more attention to rare classes.
    """
    from collections import Counter
    
    counts = Counter(dataset.labels)
    total = len(dataset.labels)
    
    weights = []
    for i in range(5):
        count = counts.get(i, 1)
        # Inverse frequency weighting
        weight = total / (5 * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        signals = batch['signal'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
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
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100 * np.mean(all_preds == all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_preds, all_labels


def train(args):
    """Main training function."""
    
    print("="*60)
    print("ECG ARRHYTHMIA CLASSIFICATION TRAINING")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
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
        window_size=args.window_size
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
    
    # Loss function with class weights
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy().round(2)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"ecg_{args.model}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-"*60)
    
    best_f1 = 0.0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, f1_macro, f1_weighted, _, _ = validate(
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
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"F1-Macro: {f1_macro:.4f}")
        
        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'args': vars(args),
                'class_weights': class_weights.cpu()
            }, results_dir / 'best_model.pth')
            
            print(f"  -> Saved best model (F1-Macro: {f1_macro:.4f})")
    
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
    
    test_loss, test_acc, test_f1_macro, test_f1_weighted, preds, labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1-Macro: {test_f1_macro:.4f}")
    print(f"  F1-Weighted: {test_f1_weighted:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Save results
    torch.save({
        'history': history,
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'confusion_matrix': cm,
        'predictions': preds,
        'labels': labels,
        'args': vars(args)
    }, results_dir / 'results.pth')
    
    print(f"\nResults saved to: {results_dir}")
    
    return model, history, results_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ECG arrhythmia classifier')
    
    parser.add_argument('--data-dir', type=str, default='./data/mitbih',
                        help='Path to MIT-BIH data')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--window-size', type=int, default=360,
                        help='ECG window size in samples')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    train(args)