"""
Download MIT-BIH Arrhythmia Database.
Developed by Kuldeep Choksi

The MIT-BIH database contains 48 half-hour ECG recordings from 47 subjects.
Each recording has two leads (channels) sampled at 360 Hz.
Annotations mark each heartbeat with its type.

Usage:
    python utils/download_data.py
"""

import os
from pathlib import Path
import wfdb


def download_mitbih(data_dir='./data/mitbih'):
    """
    Download the MIT-BIH Arrhythmia Database from PhysioNet.
    
    The database contains 48 records:
    - Records 100-124: First set (25 records)
    - Records 200-234: Second set (23 records)
    
    Each record has:
    - .dat file: Raw signal data
    - .hea file: Header with metadata
    - .atr file: Annotations (beat labels)
    """
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # MIT-BIH record numbers
    # Note: Not all numbers in range exist (e.g., no 110, 120, etc.)
    records = [
        # 100 series
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119',
        '121', '122', '123', '124',
        # 200 series  
        '200', '201', '202', '203', '205', '207', '208', '209', '210',
        '212', '213', '214', '215', '217', '219',
        '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    print("="*60)
    print("MIT-BIH Arrhythmia Database Download")
    print("Developed by Kuldeep Choksi")
    print("="*60)
    print(f"\nDownloading {len(records)} records to {data_path}")
    print("This may take a few minutes...\n")
    
    downloaded = 0
    failed = []
    
    for record_id in records:
        try:
            # Check if already downloaded
            if (data_path / f"{record_id}.dat").exists():
                print(f"  {record_id}: Already exists, skipping")
                downloaded += 1
                continue
            
            # Download from PhysioNet
            print(f"  {record_id}: Downloading...", end=" ")
            wfdb.dl_database(
                'mitdb',
                dl_dir=str(data_path),
                records=[record_id]
            )
            print("Done")
            downloaded += 1
            
        except Exception as e:
            print(f"Failed ({e})")
            failed.append(record_id)
    
    print("\n" + "="*60)
    print(f"Download complete!")
    print(f"  Successfully downloaded: {downloaded}/{len(records)} records")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Location: {data_path.absolute()}")
    print("="*60)
    
    return downloaded


def verify_download(data_dir='./data/mitbih'):
    """Verify the download by loading one record."""
    
    data_path = Path(data_dir)
    
    # Try to load record 100
    try:
        record = wfdb.rdrecord(str(data_path / '100'))
        annotation = wfdb.rdann(str(data_path / '100'), 'atr')
        
        print("\nVerification - Record 100:")
        print(f"  Sampling frequency: {record.fs} Hz")
        print(f"  Signal length: {record.sig_len} samples ({record.sig_len/record.fs:.1f} seconds)")
        print(f"  Number of channels: {record.n_sig}")
        print(f"  Channel names: {record.sig_name}")
        print(f"  Number of annotated beats: {len(annotation.symbol)}")
        
        # Count beat types
        from collections import Counter
        beat_counts = Counter(annotation.symbol)
        print(f"  Beat type distribution: {dict(beat_counts)}")
        
        return True
        
    except Exception as e:
        print(f"\nVerification failed: {e}")
        return False


if __name__ == '__main__':
    download_mitbih()
    print()
    verify_download()