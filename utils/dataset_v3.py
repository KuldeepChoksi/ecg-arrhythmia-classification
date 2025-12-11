"""
ECG Dataset Loader for MIT-BIH Arrhythmia Database.
Developed by Kuldeep Choksi

V3: Stratified splitting - ensures all classes in train/val/test.

Beat Types (AAMI Standard):
    N - Normal
    S - Supraventricular  
    V - Ventricular
    F - Fusion
    Q - Unknown/Paced
"""

import numpy as np
import wfdb
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


AAMI_CLASSES = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

SYMBOL_TO_AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

IGNORE_SYMBOLS = ['+', '~', '!', '"', '[', ']', '|', 'x']

CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
               'Fusion (F)', 'Unknown (Q)']


class MITBIHDataset(Dataset):
    """PyTorch Dataset for MIT-BIH with pre-loaded beats."""
    
    def __init__(
        self,
        beats: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        oversample: bool = False,
        oversample_factor: float = 0.3
    ):
        self.beats = beats.copy()
        self.labels = labels.copy()
        self.augment = augment
        
        if oversample:
            self._oversample_rare_classes(oversample_factor)
        
        self._print_stats()
    
    def _oversample_rare_classes(self, factor: float):
        """Oversample rare classes."""
        counts = Counter(self.labels)
        max_count = max(counts.values())
        target_count = int(max_count * factor)
        
        print(f"\nOversampling rare classes (target: {target_count})...")
        
        new_beats = list(self.beats)
        new_labels = list(self.labels)
        
        for class_idx in range(5):
            current_count = counts.get(class_idx, 0)
            if current_count >= target_count or current_count == 0:
                continue
            
            class_indices = np.where(self.labels == class_idx)[0]
            samples_to_add = target_count - current_count
            
            print(f"  Class {class_idx}: {current_count} -> {target_count} (+{samples_to_add})")
            
            for _ in range(samples_to_add):
                idx = np.random.choice(class_indices)
                beat = self._augment_beat_strong(self.beats[idx].copy())
                new_beats.append(beat)
                new_labels.append(class_idx)
        
        self.beats = np.array(new_beats, dtype=np.float32)
        self.labels = np.array(new_labels, dtype=np.int64)
    
    def _augment_beat_strong(self, beat: np.ndarray) -> np.ndarray:
        """Strong augmentation for oversampled beats."""
        scale = np.random.uniform(0.8, 1.2)
        beat = beat * scale
        noise = np.random.normal(0, 0.1, len(beat))
        beat = beat + noise
        shift = np.random.randint(-20, 20)
        beat = np.roll(beat, shift)
        return beat
    
    def _augment_beat(self, beat: np.ndarray) -> np.ndarray:
        """Light augmentation."""
        if np.random.random() < 0.5:
            beat = beat * np.random.uniform(0.9, 1.1)
        if np.random.random() < 0.3:
            beat = beat + np.random.normal(0, 0.05, len(beat))
        if np.random.random() < 0.3:
            beat = np.roll(beat, np.random.randint(-10, 10))
        return beat
    
    def _print_stats(self):
        counts = Counter(self.labels)
        total = len(self.labels)
        
        print(f"\nDataset: {total:,} beats")
        for i, name in enumerate(CLASS_NAMES):
            count = counts.get(i, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    def __len__(self):
        return len(self.beats)
    
    def __getitem__(self, idx):
        beat = self.beats[idx].copy()
        if self.augment:
            beat = self._augment_beat(beat)
        return {
            'signal': torch.FloatTensor(beat).unsqueeze(0),
            'label': int(self.labels[idx])
        }


def load_all_beats(data_dir: str = './data/mitbih', window_size: int = 360, lead: int = 0):
    """Load all beats from all records."""
    
    data_path = Path(data_dir)
    
    all_records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119',
        '121', '122', '123', '124',
        '200', '201', '202', '203', '205', '207', '208', '209', '210',
        '212', '213', '214', '215', '217', '219',
        '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    all_beats = []
    all_labels = []
    
    print(f"Loading {len(all_records)} records...")
    
    half_window = window_size // 2
    
    for record_id in all_records:
        try:
            record = wfdb.rdrecord(str(data_path / record_id))
            annotation = wfdb.rdann(str(data_path / record_id), 'atr')
            
            ecg_signal = record.p_signal[:, lead]
            
            for peak, symbol in zip(annotation.sample, annotation.symbol):
                if symbol in IGNORE_SYMBOLS or symbol not in SYMBOL_TO_AAMI:
                    continue
                
                start = peak - half_window
                end = peak + half_window
                
                if start < 0 or end > len(ecg_signal):
                    continue
                
                beat = ecg_signal[start:end].copy()
                
                # Normalize
                mean, std = np.mean(beat), np.std(beat)
                if std > 0:
                    beat = (beat - mean) / std
                else:
                    beat = beat - mean
                
                label = AAMI_CLASSES[SYMBOL_TO_AAMI[symbol]]
                
                all_beats.append(beat)
                all_labels.append(label)
                
        except Exception as e:
            print(f"  Warning: {record_id}: {e}")
    
    return np.array(all_beats, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def get_stratified_split(
    data_dir: str = './data/mitbih',
    window_size: int = 360,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    oversample_train: bool = False
) -> Tuple[MITBIHDataset, MITBIHDataset, MITBIHDataset]:
    """
    Create stratified train/val/test splits.
    
    Each split has the same proportion of each class.
    """
    
    print("="*60)
    print("Loading MIT-BIH with STRATIFIED splitting")
    print("="*60)
    
    # Load all beats
    all_beats, all_labels = load_all_beats(data_dir, window_size)
    
    print(f"\nTotal beats loaded: {len(all_beats):,}")
    
    # Show overall distribution
    counts = Counter(all_labels)
    print("\nOverall distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = counts.get(i, 0)
        print(f"  {name}: {count:,} ({100*count/len(all_labels):.1f}%)")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_beats, all_labels,
        test_size=test_ratio,
        stratify=all_labels,
        random_state=seed
    )
    
    # Second split: separate val from train
    val_size_adjusted = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=seed
    )
    
    print(f"\n--- Stratified Split ---")
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Create datasets
    print("\n--- Training Set ---")
    train_dataset = MITBIHDataset(X_train, y_train, augment=True, oversample=oversample_train)
    
    print("\n--- Validation Set ---")
    val_dataset = MITBIHDataset(X_val, y_val, augment=False, oversample=False)
    
    print("\n--- Test Set ---")
    test_dataset = MITBIHDataset(X_test, y_test, augment=False, oversample=False)
    
    return train_dataset, val_dataset, test_dataset


# Quick test
if __name__ == '__main__':
    print("ECG Dataset V3 - Stratified Split")
    print("Developed by Kuldeep Choksi\n")
    
    train_ds, val_ds, test_ds = get_stratified_split('./data/mitbih', oversample_train=False)