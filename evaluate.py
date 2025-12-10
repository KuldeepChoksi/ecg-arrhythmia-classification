"""
Evaluation script for ECG Arrhythmia Classification.
Developed by Kuldeep Choksi

Generates:
- Confusion matrix heatmap
- Per-class metrics
- ECG waveform visualizations with predictions
- ROC curves for each class

Usage:
    python evaluate.py --checkpoint results/ecg_cnn_xxx/best_model.pth
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import ECGClassifier, ECGResNet
from utils import MITBIHDataset, get_train_val_test_split


CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
               'Fusion (F)', 'Unknown (Q)']
CLASS_NAMES_SHORT = ['N', 'S', 'V', 'F', 'Q']


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})
    
    # Create model
    model_type = args.get('model', 'cnn')
    if model_type == 'cnn':
        model = ECGClassifier(num_classes=5, dropout=args.get('dropout', 0.3))
    else:
        model = ECGResNet(num_classes=5, dropout=args.get('dropout', 0.3))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def get_predictions(model, data_loader, device):
    """Get all predictions and probabilities."""
    all_probs = []
    all_preds = []
    all_labels = []
    all_signals = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            signals = batch['signal'].to(device)
            labels = batch['label'].numpy()
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_signals.extend(signals.cpu().numpy())
    
    return (np.array(all_probs), np.array(all_preds), 
            np.array(all_labels), np.array(all_signals))


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.suptitle('ECG Arrhythmia Classification - Kuldeep Choksi', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {output_path}")


def plot_roc_curves(y_true, y_probs, output_path):
    """Plot ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES_SHORT, colors)):
        # Binary labels for this class
        y_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        # Skip if no positive samples
        if y_binary.sum() == 0:
            continue
        
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - ECG Arrhythmia Classification\nDeveloped by Kuldeep Choksi')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC curves to {output_path}")


def plot_ecg_samples(signals, labels, preds, probs, output_path, num_samples=12):
    """Plot sample ECG waveforms with predictions."""
    
    # Get samples from each class
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    axes = axes.flatten()
    
    sample_idx = 0
    
    # Try to get samples from different scenarios
    scenarios = [
        ('Correct Normal', lambda l, p: (l == 0) & (p == 0)),
        ('Correct Ventricular', lambda l, p: (l == 2) & (p == 2)),
        ('Correct Unknown', lambda l, p: (l == 4) & (p == 4)),
        ('Missed Supraventricular', lambda l, p: (l == 1) & (p != 1)),
        ('Missed Ventricular', lambda l, p: (l == 2) & (p != 2)),
        ('False Ventricular', lambda l, p: (l != 2) & (p == 2)),
    ]
    
    for scenario_name, condition in scenarios:
        mask = condition(labels, preds)
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            # Get 2 samples per scenario
            for idx in indices[:2]:
                if sample_idx >= len(axes):
                    break
                    
                ax = axes[sample_idx]
                signal = signals[idx].squeeze()
                true_label = CLASS_NAMES_SHORT[labels[idx]]
                pred_label = CLASS_NAMES_SHORT[preds[idx]]
                confidence = probs[idx, preds[idx]] * 100
                
                # Time axis (assuming 360 Hz)
                time = np.arange(len(signal)) / 360 * 1000  # ms
                
                # Color based on correct/incorrect
                color = 'green' if labels[idx] == preds[idx] else 'red'
                
                ax.plot(time, signal, color='black', linewidth=0.8)
                ax.set_title(f'True: {true_label} | Pred: {pred_label} ({confidence:.0f}%)',
                           color=color, fontsize=10)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                
                sample_idx += 1
    
    # Fill remaining with random samples
    while sample_idx < len(axes):
        idx = np.random.randint(len(signals))
        ax = axes[sample_idx]
        signal = signals[idx].squeeze()
        true_label = CLASS_NAMES_SHORT[labels[idx]]
        pred_label = CLASS_NAMES_SHORT[preds[idx]]
        confidence = probs[idx, preds[idx]] * 100
        
        time = np.arange(len(signal)) / 360 * 1000
        color = 'green' if labels[idx] == preds[idx] else 'red'
        
        ax.plot(time, signal, color='black', linewidth=0.8)
        ax.set_title(f'True: {true_label} | Pred: {pred_label} ({confidence:.0f}%)',
                   color=color, fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        sample_idx += 1
    
    plt.suptitle('ECG Waveform Samples - Kuldeep Choksi', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ECG samples to {output_path}")


def plot_class_distribution(y_true, y_pred, output_path):
    """Plot class distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True distribution
    true_counts = [np.sum(y_true == i) for i in range(5)]
    pred_counts = [np.sum(y_pred == i) for i in range(5)]
    
    x = np.arange(5)
    width = 0.35
    
    axes[0].bar(x - width/2, true_counts, width, label='True', color='steelblue')
    axes[0].bar(x + width/2, pred_counts, width, label='Predicted', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASS_NAMES_SHORT)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Per-class accuracy
    class_acc = []
    for i in range(5):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean() * 100
        else:
            acc = 0
        class_acc.append(acc)
    
    colors = ['green' if acc > 70 else 'orange' if acc > 40 else 'red' for acc in class_acc]
    axes[1].bar(x, class_acc, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES_SHORT)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall (%)')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, acc in enumerate(class_acc):
        axes[1].text(i, acc + 2, f'{acc:.0f}%', ha='center', fontsize=10)
    
    plt.suptitle('ECG Classification Analysis - Kuldeep Choksi', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved class distribution to {output_path}")


def evaluate(args):
    """Main evaluation function."""
    print("="*60)
    print("ECG ARRHYTHMIA CLASSIFICATION EVALUATION")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, device)
    
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  F1-Macro: {checkpoint.get('f1_macro', 0):.4f}")
    
    # Load test data
    print("\nLoading test dataset...")
    _, _, test_dataset = get_train_val_test_split(
        data_dir=args.data_dir,
        window_size=checkpoint.get('args', {}).get('window_size', 360)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )
    
    # Get predictions
    print("\nGenerating predictions...")
    probs, preds, labels, signals = get_predictions(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds) * 100
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    
    # Create output directory
    checkpoint_dir = Path(args.checkpoint).parent
    eval_dir = checkpoint_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_confusion_matrix(labels, preds, eval_dir / "confusion_matrix.png")
    plot_roc_curves(labels, probs, eval_dir / "roc_curves.png")
    plot_ecg_samples(signals, labels, preds, probs, eval_dir / "ecg_samples.png")
    plot_class_distribution(labels, preds, eval_dir / "class_distribution.png")
    
    # Save results text
    with open(eval_dir / "results.txt", 'w') as f:
        f.write("ECG Arrhythmia Classification Results\n")
        f.write("Developed by Kuldeep Choksi\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"F1-Macro: {f1_macro:.4f}\n")
        f.write(f"F1-Weighted: {f1_weighted:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(labels, preds, target_names=CLASS_NAMES))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(labels, preds)))
    
    print(f"\nResults saved to: {eval_dir}")
    
    return accuracy, f1_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ECG classifier')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data/mitbih',
                        help='Path to MIT-BIH data')
    
    args = parser.parse_args()
    evaluate(args)