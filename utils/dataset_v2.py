"""
ECG Dataset Loader for MIT-BIH Arrhythmia Database.
Developed by Kuldeep Choksi

Extracts individual heartbeats from continuous ECG recordings
and assigns labels based on annotations.

V2: Added oversampling for rare classes to handle class imbalance.

Beat Types (AAMI Standard):
    N - Normal (includes N, L, R, e, j)
    S - Supraventricular (includes A, a, J, S)
    V - Ventricular (includes V, E)
    F - Fusion (includes F)
    Q - Unknown (includes /, f, Q)
"""

import numpy as np
import wfdb
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from typing import Tuple, List, Dict, Optional


# AAMI standard beat type mapping
AAMI_CLASSES = {
    'N': 0,  # Normal
    'S': 1,  # Supraventricular
    'V': 2,  # Ventricular
    'F': 3,  # Fusion
    'Q': 4,  # Unknown/Paced
}

# MIT-BIH symbol to AAMI class mapping
SYMBOL_TO_AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

IGNORE_SYMBOLS = ['+', '~', '!', '"', '[', ']', '|', 'x']


class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia Database.
    
    Args:
        data_dir: Path to downloaded MIT-BIH data
        records: List of record IDs to use
        window_size: Number of samples per heartbeat window
        lead: Which ECG lead to use (0 = MLII, 1 = V5)
        normalize: Whether to normalize each beat
        augment: Whether to apply data augmentation
        oversample: Whether to oversample rare classes
        oversample_factor: Target ratio relative to majority class
    """
    
    def __init__(
        self,
        data_dir: str = './data/mitbih',
        records: Optional[List[str]] = None,
        window_size: int = 360,
        lead: int = 0,
        normalize: bool = True,
        augment: bool = False,
        oversample: bool = False,
        oversample_factor: float = 0.5
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.lead = lead
        self.normalize = normalize
        self.augment = augment
        self.oversample = oversample
        self.oversample_factor = oversample_factor
        
        if records is None:
            records = [
                '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                '111', '112', '113', '114', '115', '116', '117', '118', '119',
                '121', '122', '123', '124',
                '200', '201', '202', '203', '205', '207', '208', '209', '210',
                '212', '213', '214', '215', '217', '219',
                '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
            ]
        self.records = records
        
        self.beats = []
        self.labels = []
        self._load_all_records()
        
        # Apply oversampling if requested
        if oversample:
            self._oversample_rare_classes()
        
        self.beats = np.array(self.beats, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        self._print_stats()
    
    def _load_all_records(self):
        print(f"Loading {len(self.records)} records...")
        for record_id in self.records:
            try:
                self._load_record(record_id)
            except Exception as e:
                print(f"  Warning: Could not load record {record_id}: {e}")
    
    def _load_record(self, record_id: str):
        record_path = self.data_dir / record_id
        record = wfdb.rdrecord(str(record_path))
        annotation = wfdb.rdann(str(record_path), 'atr')
        
        ecg_signal = record.p_signal[:, self.lead]
        r_peaks = annotation.sample
        symbols = annotation.symbol
        
        half_window = self.window_size // 2
        
        for i, (peak, symbol) in enumerate(zip(r_peaks, symbols)):
            if symbol in IGNORE_SYMBOLS or symbol not in SYMBOL_TO_AAMI:
                continue
            
            start = peak - half_window
            end = peak + half_window
            
            if start < 0 or end > len(ecg_signal):
                continue
            
            beat = ecg_signal[start:end].copy()
            
            if self.normalize:
                beat = self._normalize_beat(beat)
            
            aami_class = SYMBOL_TO_AAMI[symbol]
            label = AAMI_CLASSES[aami_class]
            
            self.beats.append(beat)
            self.labels.append(label)
    
    def _normalize_beat(self, beat: np.ndarray) -> np.ndarray:
        mean = np.mean(beat)
        std = np.std(beat)
        if std > 0:
            beat = (beat - mean) / std
        else:
            beat = beat - mean
        return beat
    
    def _oversample_rare_classes(self):
        """Oversample rare classes to balance the dataset."""
        
        counts = Counter(self.labels)
        max_count = max(counts.values())
        target_count = int(max_count * self.oversample_factor)
        
        print(f"\nOversampling rare classes (target: {target_count} per class)...")
        
        new_beats = list(self.beats)
        new_labels = list(self.labels)
        
        for class_idx in range(5):
            current_count = counts.get(class_idx, 0)
            
            if current_count >= target_count or current_count == 0:
                continue
            
            # Get indices of this class
            class_indices = [i for i, l in enumerate(self.labels) if l == class_idx]
            
            # How many samples to add
            samples_to_add = target_count - current_count
            
            print(f"  Class {class_idx}: {current_count} -> {target_count} (+{samples_to_add})")
            
            # Randomly sample with replacement and add augmented versions
            for _ in range(samples_to_add):
                idx = np.random.choice(class_indices)
                beat = self.beats[idx].copy()
                
                # Apply augmentation to synthetic samples
                beat = self._augment_beat_strong(beat)
                
                new_beats.append(beat)
                new_labels.append(class_idx)
        
        self.beats = new_beats
        self.labels = new_labels
    
    def _augment_beat_strong(self, beat: np.ndarray) -> np.ndarray:
        """Strong augmentation for oversampled beats."""
        
        # Random amplitude scaling (0.8 to 1.2)
        scale = np.random.uniform(0.8, 1.2)
        beat = beat * scale
        
        # Random noise addition
        noise = np.random.normal(0, 0.1, len(beat))
        beat = beat + noise
        
        # Random time shift (up to 20 samples)
        shift = np.random.randint(-20, 20)
        beat = np.roll(beat, shift)
        
        # Random baseline wander
        if np.random.random() < 0.3:
            t = np.linspace(0, 1, len(beat))
            wander = 0.1 * np.sin(2 * np.pi * np.random.uniform(0.5, 2) * t)
            beat = beat + wander
        
        return beat
    
    def _print_stats(self):
        class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
                       'Fusion (F)', 'Unknown (Q)']
        
        counts = Counter(self.labels)
        total = len(self.labels)
        
        print(f"\nDataset Statistics:")
        print(f"  Total beats: {total:,}")
        print(f"  Window size: {self.window_size} samples")
        if self.oversample:
            print(f"  Oversampling: ENABLED")
        print(f"\n  Class Distribution:")
        
        for i, name in enumerate(class_names):
            count = counts.get(i, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"    {name}: {count:,} ({pct:.1f}%)")
    
    def __len__(self):
        return len(self.beats)
    
    def __getitem__(self, idx):
        beat = self.beats[idx].copy()
        label = self.labels[idx]
        
        if self.augment:
            beat = self._augment_beat(beat)
        
        beat_tensor = torch.FloatTensor(beat).unsqueeze(0)
        
        return {
            'signal': beat_tensor,
            'label': label
        }
    
    def _augment_beat(self, beat: np.ndarray) -> np.ndarray:
        """Light augmentation for training."""
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            beat = beat * scale
        
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, len(beat))
            beat = beat + noise
        
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 10)
            beat = np.roll(beat, shift)
        
        return beat


def get_train_val_test_split(
    data_dir: str = './data/mitbih',
    window_size: int = 360,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    oversample_train: bool = False
) -> Tuple[MITBIHDataset, MITBIHDataset, MITBIHDataset]:
    """Create train/val/test splits with optional oversampling."""
    
    np.random.seed(seed)
    
    all_records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119',
        '121', '122', '123', '124',
        '200', '201', '202', '203', '205', '207', '208', '209', '210',
        '212', '213', '214', '215', '217', '219',
        '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    np.random.shuffle(all_records)
    
    n_records = len(all_records)
    n_test = int(n_records * test_ratio)
    n_val = int(n_records * val_ratio)
    
    test_records = all_records[:n_test]
    val_records = all_records[n_test:n_test + n_val]
    train_records = all_records[n_test + n_val:]
    
    print(f"Record split:")
    print(f"  Train: {len(train_records)} records")
    print(f"  Val: {len(val_records)} records")
    print(f"  Test: {len(test_records)} records")
    
    print("\n--- Training Set ---")
    train_dataset = MITBIHDataset(
        data_dir, train_records, window_size, 
        augment=True, oversample=oversample_train, oversample_factor=0.3
    )
    
    print("\n--- Validation Set ---")
    val_dataset = MITBIHDataset(data_dir, val_records, window_size, augment=False)
    
    print("\n--- Test Set ---")
    test_dataset = MITBIHDataset(data_dir, test_records, window_size, augment=False)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    print("="*60)
    print("ECG Dataset Loader V2 (with Oversampling)")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Test with oversampling
    print("\nTesting with oversampling enabled...")
    train_ds, val_ds, test_ds = get_train_val_test_split(
        './data/mitbih', 
        oversample_train=True
    )