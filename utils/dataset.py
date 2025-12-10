"""
ECG Dataset Loader for MIT-BIH Arrhythmia Database.
Developed by Kuldeep Choksi

Extracts individual heartbeats from continuous ECG recordings
and assigns labels based on annotations.

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
# Maps MIT-BIH symbols to 5 AAMI classes
AAMI_CLASSES = {
    'N': 0,  # Normal
    'S': 1,  # Supraventricular
    'V': 2,  # Ventricular
    'F': 3,  # Fusion
    'Q': 4,  # Unknown/Paced
}

# MIT-BIH symbol to AAMI class mapping
SYMBOL_TO_AAMI = {
    # Normal (N)
    'N': 'N',  # Normal beat
    'L': 'N',  # Left bundle branch block
    'R': 'N',  # Right bundle branch block
    'e': 'N',  # Atrial escape beat
    'j': 'N',  # Nodal (junctional) escape beat
    
    # Supraventricular (S)
    'A': 'S',  # Atrial premature beat
    'a': 'S',  # Aberrated atrial premature beat
    'J': 'S',  # Nodal (junctional) premature beat
    'S': 'S',  # Supraventricular premature beat
    
    # Ventricular (V)
    'V': 'V',  # Premature ventricular contraction
    'E': 'V',  # Ventricular escape beat
    
    # Fusion (F)
    'F': 'F',  # Fusion of ventricular and normal beat
    
    # Unknown/Paced (Q)
    '/': 'Q',  # Paced beat
    'f': 'Q',  # Fusion of paced and normal beat
    'Q': 'Q',  # Unclassifiable beat
}

# Symbols to ignore (not actual beats)
IGNORE_SYMBOLS = ['+', '~', '!', '"', '[', ']', '|', 'x']


class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia Database.
    
    Extracts fixed-length windows around each heartbeat and assigns
    AAMI standard labels.
    
    Args:
        data_dir: Path to downloaded MIT-BIH data
        records: List of record IDs to use (default: all)
        window_size: Number of samples per heartbeat window (default: 360 = 1 second)
        lead: Which ECG lead to use (0 = MLII, 1 = V5)
        normalize: Whether to normalize each beat
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        data_dir: str = './data/mitbih',
        records: Optional[List[str]] = None,
        window_size: int = 360,
        lead: int = 0,
        normalize: bool = True,
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.lead = lead
        self.normalize = normalize
        self.augment = augment
        
        # Default records (all 48)
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
        
        # Extract all beats
        self.beats = []
        self.labels = []
        self._load_all_records()
        
        # Convert to numpy arrays
        self.beats = np.array(self.beats, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Print class distribution
        self._print_stats()
    
    def _load_all_records(self):
        """Load and extract beats from all records."""
        
        print(f"Loading {len(self.records)} records...")
        
        for record_id in self.records:
            try:
                self._load_record(record_id)
            except Exception as e:
                print(f"  Warning: Could not load record {record_id}: {e}")
    
    def _load_record(self, record_id: str):
        """Extract beats from a single record."""
        
        record_path = self.data_dir / record_id
        
        # Load signal and annotations
        record = wfdb.rdrecord(str(record_path))
        annotation = wfdb.rdann(str(record_path), 'atr')
        
        # Get signal from selected lead
        ecg_signal = record.p_signal[:, self.lead]
        
        # Get R-peak locations and symbols
        r_peaks = annotation.sample
        symbols = annotation.symbol
        
        half_window = self.window_size // 2
        
        # Extract each beat
        for i, (peak, symbol) in enumerate(zip(r_peaks, symbols)):
            # Skip non-beat symbols
            if symbol in IGNORE_SYMBOLS:
                continue
            
            # Skip if symbol not in our mapping
            if symbol not in SYMBOL_TO_AAMI:
                continue
            
            # Get window boundaries
            start = peak - half_window
            end = peak + half_window
            
            # Skip if window is out of bounds
            if start < 0 or end > len(ecg_signal):
                continue
            
            # Extract beat window
            beat = ecg_signal[start:end].copy()
            
            # Normalize if requested
            if self.normalize:
                beat = self._normalize_beat(beat)
            
            # Get AAMI label
            aami_class = SYMBOL_TO_AAMI[symbol]
            label = AAMI_CLASSES[aami_class]
            
            self.beats.append(beat)
            self.labels.append(label)
    
    def _normalize_beat(self, beat: np.ndarray) -> np.ndarray:
        """Normalize a beat to zero mean and unit variance."""
        mean = np.mean(beat)
        std = np.std(beat)
        if std > 0:
            beat = (beat - mean) / std
        else:
            beat = beat - mean
        return beat
    
    def _print_stats(self):
        """Print dataset statistics."""
        
        class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
                       'Fusion (F)', 'Unknown (Q)']
        
        counts = Counter(self.labels)
        total = len(self.labels)
        
        print(f"\nDataset Statistics:")
        print(f"  Total beats: {total:,}")
        print(f"  Window size: {self.window_size} samples ({self.window_size/360:.2f} seconds)")
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
        
        # Data augmentation
        if self.augment:
            beat = self._augment_beat(beat)
        
        # Convert to tensor and add channel dimension
        beat_tensor = torch.FloatTensor(beat).unsqueeze(0)  # [1, window_size]
        
        return {
            'signal': beat_tensor,
            'label': label
        }
    
    def _augment_beat(self, beat: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a beat."""
        
        # Random amplitude scaling (0.9 to 1.1)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            beat = beat * scale
        
        # Random noise addition
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, len(beat))
            beat = beat + noise
        
        # Random time shift (up to 10 samples)
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 10)
            beat = np.roll(beat, shift)
        
        return beat


def get_train_val_test_split(
    data_dir: str = './data/mitbih',
    window_size: int = 360,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[MITBIHDataset, MITBIHDataset, MITBIHDataset]:
    """
    Create train/val/test splits using record-based splitting.
    
    Records are split to ensure no patient appears in multiple sets.
    This is important for fair evaluation.
    """
    
    np.random.seed(seed)
    
    all_records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119',
        '121', '122', '123', '124',
        '200', '201', '202', '203', '205', '207', '208', '209', '210',
        '212', '213', '214', '215', '217', '219',
        '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    # Shuffle records
    np.random.shuffle(all_records)
    
    # Split records
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
    
    # Create datasets
    print("\n--- Training Set ---")
    train_dataset = MITBIHDataset(data_dir, train_records, window_size, augment=True)
    
    print("\n--- Validation Set ---")
    val_dataset = MITBIHDataset(data_dir, val_records, window_size, augment=False)
    
    print("\n--- Test Set ---")
    test_dataset = MITBIHDataset(data_dir, test_records, window_size, augment=False)
    
    return train_dataset, val_dataset, test_dataset


# Quick test
if __name__ == '__main__':
    print("="*60)
    print("ECG Dataset Loader")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    
    # Test loading
    dataset = MITBIHDataset('./data/mitbih')
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample beat shape: {sample['signal'].shape}")
    print(f"Sample label: {sample['label']}")
    
    # Test train/val/test split
    print("\n" + "="*60)
    print("Testing train/val/test split...")
    print("="*60)
    train_ds, val_ds, test_ds = get_train_val_test_split('./data/mitbih')