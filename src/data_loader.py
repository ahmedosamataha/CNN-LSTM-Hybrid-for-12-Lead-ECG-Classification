"""
ECG Data Loading and Preprocessing Module
==========================================
Handles loading PTB-XL dataset, preprocessing 12-lead ECG signals,
and creating PyTorch DataLoaders with patient-level train/val/test splits.

Dataset: PTB-XL (https://physionet.org/content/ptb-xl/1.0.3/)
- 21,837 clinical 12-lead ECG records (10 seconds each)
- 500 Hz sampling rate
- Multi-label annotations for cardiac abnormalities
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample, butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONSTANTS
# ============================================================================
SAMPLING_RATE = 500  # Target sampling rate (Hz)
SIGNAL_LENGTH = 5000  # 10 seconds * 500 Hz
NUM_LEADS = 12
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Superclass mapping (5 diagnostic superclasses)
SUPERCLASS_LABELS = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NUM_CLASSES = len(SUPERCLASS_LABELS)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_ptbxl_data(data_path, sampling_rate=500):
    """
    Load PTB-XL dataset from disk.
    
    Parameters
    ----------
    data_path : str
        Path to the PTB-XL dataset directory (containing records500/ and ptbxl_database.csv)
    sampling_rate : int
        Sampling rate to use (100 or 500 Hz). Default: 500.
    
    Returns
    -------
    X : np.ndarray, shape (n_samples, signal_length, 12)
        ECG signal data
    Y : np.ndarray, shape (n_samples, 5)
        Multi-label binary matrix for 5 diagnostic superclasses
    metadata : pd.DataFrame
        Patient metadata for stratified splitting
    """
    # Load annotation database
    database_path = os.path.join(data_path, 'ptbxl_database.csv')
    df = pd.read_csv(database_path, index_col='ecg_id')
    
    # Parse SCP codes (stored as string representation of dict)
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    
    # Load SCP statements mapping
    scp_path = os.path.join(data_path, 'scp_statements.csv')
    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df['diagnostic'] == 1]  # Keep only diagnostic codes
    
    # Map SCP codes to diagnostic superclasses
    def get_superclass_labels(scp_dict):
        """Map SCP codes to diagnostic superclass labels."""
        labels = set()
        for code, confidence in scp_dict.items():
            if confidence >= 50 and code in scp_df.index:  # Only use confident annotations
                superclass = scp_df.loc[code, 'diagnostic_class']
                if superclass in SUPERCLASS_LABELS:
                    labels.add(superclass)
        return list(labels)
    
    df['diagnostic_superclass'] = df['scp_codes'].apply(get_superclass_labels)
    
    # Filter out samples with no valid labels
    df = df[df['diagnostic_superclass'].apply(len) > 0]
    
    # Binarize labels
    mlb = MultiLabelBinarizer(classes=SUPERCLASS_LABELS)
    Y = mlb.fit_transform(df['diagnostic_superclass'])
    
    # Load ECG signals
    print(f"Loading {len(df)} ECG records at {sampling_rate}Hz...")
    if sampling_rate == 500:
        signal_dir = 'records500'
        signal_len = 5000
    else:
        signal_dir = 'records100'
        signal_len = 1000
    
    X = np.zeros((len(df), signal_len, NUM_LEADS), dtype=np.float32)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        filename = os.path.join(data_path, row['filename_hr'] if sampling_rate == 500 else row['filename_lr'])
        try:
            record = wfdb.rdrecord(filename)
            signal = record.p_signal
            # Ensure correct length
            if signal.shape[0] >= signal_len:
                X[i] = signal[:signal_len]
            else:
                X[i, :signal.shape[0]] = signal
        except Exception as e:
            print(f"Error loading record {idx}: {e}")
            continue
        
        if (i + 1) % 2000 == 0:
            print(f"  Loaded {i+1}/{len(df)} records...")
    
    print(f"Loaded {len(df)} ECG records successfully.")
    
    return X, Y, df


# ============================================================================
# PREPROCESSING
# ============================================================================
def normalize_ecg(X, method='per_lead'):
    """
    Normalize ECG signals.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, signal_length, 12)
        Raw ECG signals
    method : str
        'per_lead': Normalize each lead independently (zero mean, unit variance)
        'per_sample': Normalize each sample globally
        'global': Use global statistics across entire dataset
    
    Returns
    -------
    X_norm : np.ndarray
        Normalized ECG signals
    stats : dict
        Normalization statistics (for applying to test data)
    """
    X_norm = X.copy()
    stats = {}
    
    if method == 'per_lead':
        # Compute mean and std per lead across all samples
        mean = np.mean(X, axis=(0, 1))  # shape: (12,)
        std = np.std(X, axis=(0, 1))    # shape: (12,)
        std[std < 1e-8] = 1.0  # Prevent division by zero
        X_norm = (X - mean) / std
        stats = {'mean': mean, 'std': std}
    
    elif method == 'per_sample':
        for i in range(X.shape[0]):
            mean = np.mean(X[i])
            std = np.std(X[i])
            if std < 1e-8:
                std = 1.0
            X_norm[i] = (X[i] - mean) / std
    
    elif method == 'global':
        mean = np.mean(X)
        std = np.std(X)
        if std < 1e-8:
            std = 1.0
        X_norm = (X - mean) / std
        stats = {'mean': mean, 'std': std}
    
    return X_norm, stats


def apply_normalization(X, stats, method='per_lead'):
    """Apply pre-computed normalization stats to new data."""
    if method == 'per_lead':
        return (X - stats['mean']) / stats['std']
    elif method == 'global':
        return (X - stats['mean']) / stats['std']
    return X


# ============================================================================
# SIGNAL FILTERING (Clinical ECG Preprocessing)
# ============================================================================
def bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=500, order=4):
    """
    Apply Butterworth bandpass filter to ECG signal.
    
    Removes:
    - Baseline wander (< 0.5 Hz): caused by respiration/movement
    - High-frequency noise (> 45 Hz): EMG interference, powerline harmonics
    
    Parameters
    ----------
    signal : np.ndarray, shape (signal_length, num_leads) or (signal_length,)
        Raw ECG signal
    lowcut : float
        Lower cutoff frequency in Hz (default: 0.5 Hz)
    highcut : float
        Upper cutoff frequency in Hz (default: 45 Hz)
    fs : int
        Sampling frequency in Hz
    order : int
        Filter order (higher = sharper cutoff but more ringing)
    
    Returns
    -------
    filtered : np.ndarray
        Bandpass filtered signal
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(np.float32)
    else:
        filtered = np.zeros_like(signal, dtype=np.float32)
        for lead in range(signal.shape[1]):
            filtered[:, lead] = filtfilt(b, a, signal[:, lead])
        return filtered


def notch_filter(signal, freq=50.0, fs=500, quality=30):
    """
    Apply notch filter to remove powerline interference (50/60 Hz).
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal
    freq : float
        Frequency to remove (50 Hz in Europe, 60 Hz in US)
    fs : int
        Sampling frequency
    quality : float
        Quality factor (higher = narrower notch)
    """
    b, a = iirnotch(freq, quality, fs)
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(np.float32)
    else:
        filtered = np.zeros_like(signal, dtype=np.float32)
        for lead in range(signal.shape[1]):
            filtered[:, lead] = filtfilt(b, a, signal[:, lead])
        return filtered


def preprocess_signal(signal, fs=500, apply_bandpass=True, apply_notch=True):
    """
    Full clinical ECG preprocessing pipeline.
    
    Pipeline:
    1. Notch filter (remove 50/60 Hz powerline)
    2. Bandpass filter (0.5-45 Hz)
    
    Parameters
    ----------
    signal : np.ndarray, shape (signal_length, 12)
    fs : int
        Sampling frequency
    
    Returns
    -------
    processed : np.ndarray
        Preprocessed signal
    """
    processed = signal.copy()
    
    if apply_notch:
        processed = notch_filter(processed, freq=50.0, fs=fs)
    
    if apply_bandpass:
        processed = bandpass_filter(processed, lowcut=0.5, highcut=45.0, fs=fs)
    
    return processed


# ============================================================================
# TRAIN/VALIDATION/TEST SPLIT (Patient-Level)
# ============================================================================
def split_data_by_patient(X, Y, df, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data by patient ID to prevent data leakage.
    PTB-XL provides recommended folds (strat_fold column):
    - Folds 1-8: Training
    - Fold 9: Validation
    - Fold 10: Test
    
    Parameters
    ----------
    X, Y : arrays
        Features and labels
    df : pd.DataFrame
        Metadata with 'strat_fold' column
    
    Returns
    -------
    dict with 'train', 'val', 'test' splits
    """
    # Use PTB-XL's recommended stratified folds
    if 'strat_fold' in df.columns:
        train_mask = df['strat_fold'].values <= 8
        val_mask = df['strat_fold'].values == 9
        test_mask = df['strat_fold'].values == 10
        
        splits = {
            'X_train': X[train_mask], 'Y_train': Y[train_mask],
            'X_val': X[val_mask], 'Y_val': Y[val_mask],
            'X_test': X[test_mask], 'Y_test': Y[test_mask],
        }
    else:
        # Fallback: split by patient_id
        patient_ids = df['patient_id'].unique()
        train_ids, temp_ids = train_test_split(
            patient_ids, test_size=val_ratio + test_ratio, random_state=random_seed
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed
        )
        
        train_mask = df['patient_id'].isin(train_ids).values
        val_mask = df['patient_id'].isin(val_ids).values
        test_mask = df['patient_id'].isin(test_ids).values
        
        splits = {
            'X_train': X[train_mask], 'Y_train': Y[train_mask],
            'X_val': X[val_mask], 'Y_val': Y[val_mask],
            'X_test': X[test_mask], 'Y_test': Y[test_mask],
        }
    
    print(f"Data split:")
    print(f"  Train: {splits['X_train'].shape[0]} samples")
    print(f"  Val:   {splits['X_val'].shape[0]} samples")
    print(f"  Test:  {splits['X_test'].shape[0]} samples")
    
    return splits


# ============================================================================
# PYTORCH DATASET
# ============================================================================
class ECGDataset(Dataset):
    """
    PyTorch Dataset for 12-lead ECG data.
    
    Expects data in shape (n_samples, signal_length, num_leads).
    Returns tensors in shape (num_leads, signal_length) for Conv1D.
    """
    
    def __init__(self, X, Y, augment=False):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, signal_length, 12)
            ECG signal data
        Y : np.ndarray, shape (n_samples, num_classes)
            Multi-label binary targets
        augment : bool
            Whether to apply data augmentation during training
        """
        # Transpose to (n_samples, 12, signal_length) for Conv1D
        self.X = torch.FloatTensor(X.transpose(0, 2, 1))
        self.Y = torch.FloatTensor(Y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.augment:
            x = self._augment(x)
        
        return x, y
    
    def _augment(self, x):
        """
        Apply comprehensive augmentations to ECG signal.
        
        Augmentations simulate real-world variability:
        - Amplitude scaling → electrode placement differences
        - Gaussian noise → electrical interference
        - Temporal shift → trigger point variability
        - Lead dropout → missing/noisy electrode simulation
        - Random crop & pad → segment variability
        - Baseline wander simulation → respiratory artifact
        """
        # Random amplitude scaling (±15%)
        if torch.rand(1).item() < 0.5:
            scale = 0.85 + 0.3 * torch.rand(1).item()
            x = x * scale
        
        # Random Gaussian noise (SNR-aware)
        if torch.rand(1).item() < 0.4:
            noise_level = 0.005 + 0.015 * torch.rand(1).item()  # Variable noise
            noise = torch.randn_like(x) * noise_level
            x = x + noise
        
        # Random temporal shift (up to 100 samples = 200ms)
        if torch.rand(1).item() < 0.3:
            shift = int(torch.randint(-100, 100, (1,)).item())
            x = torch.roll(x, shifts=shift, dims=1)
        
        # Lead dropout: randomly zero out 1-2 leads (simulates electrode failure)
        if torch.rand(1).item() < 0.15:
            num_leads_to_drop = int(torch.randint(1, 3, (1,)).item())
            leads_to_drop = torch.randperm(12)[:num_leads_to_drop]
            x[leads_to_drop] = 0.0
        
        # Simulated baseline wander (low-frequency sinusoid)
        if torch.rand(1).item() < 0.2:
            freq = 0.1 + 0.4 * torch.rand(1).item()  # 0.1-0.5 Hz
            amplitude = 0.05 * torch.rand(1).item()
            t = torch.linspace(0, 10, x.shape[1])  # 10 seconds
            wander = amplitude * torch.sin(2 * 3.14159 * freq * t)
            x = x + wander.unsqueeze(0)  # Add to all leads
        
        # Random signal inversion (simulate lead reversal, rare)
        if torch.rand(1).item() < 0.05:
            lead_to_invert = int(torch.randint(0, 12, (1,)).item())
            x[lead_to_invert] = -x[lead_to_invert]
        
        return x


class MixupCollator:
    """
    Mixup data augmentation collator for DataLoader.
    
    Mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
    
    Creates virtual training examples by linear interpolation:
        x_mixed = λ * x_i + (1-λ) * x_j
        y_mixed = λ * y_i + (1-λ) * y_j
    
    This regularizes the model by encouraging linear behavior between training 
    examples, reducing overfitting and improving calibration.
    
    Parameters
    ----------
    alpha : float
        Beta distribution parameter. Higher alpha → more mixing.
        alpha=0.2 (mild), alpha=1.0 (strong), alpha=0 (disabled)
    """
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1 - lam)  # Ensure λ >= 0.5 (closer to original)
            
            batch_size = x_batch.size(0)
            index = torch.randperm(batch_size)
            
            x_mixed = lam * x_batch + (1 - lam) * x_batch[index]
            y_mixed = lam * y_batch + (1 - lam) * y_batch[index]
            
            return x_mixed, y_mixed
        
        return x_batch, y_batch


def create_dataloaders(splits, batch_size=64, augment_train=True, num_workers=0,
                       use_mixup=False, mixup_alpha=0.2):
    """
    Create PyTorch DataLoaders from split data.
    
    Parameters
    ----------
    splits : dict
        Output from split_data_by_patient()
    batch_size : int
        Batch size for training
    augment_train : bool
        Whether to augment training data
    num_workers : int
        Number of dataloader workers (0 for Windows compatibility)
    use_mixup : bool
        Whether to use Mixup augmentation on training data
    mixup_alpha : float
        Beta distribution parameter for Mixup (0.2 = mild mixing)
    
    Returns
    -------
    dict with 'train', 'val', 'test' DataLoaders
    """
    train_dataset = ECGDataset(splits['X_train'], splits['Y_train'], augment=augment_train)
    val_dataset = ECGDataset(splits['X_val'], splits['Y_val'], augment=False)
    test_dataset = ECGDataset(splits['X_test'], splits['Y_test'], augment=False)
    
    # Optional Mixup collator for training
    train_collate = MixupCollator(alpha=mixup_alpha) if use_mixup else None
    
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=True,
                           collate_fn=train_collate),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
    }
    
    print(f"DataLoaders created (batch_size={batch_size}, mixup={use_mixup}):")
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches:   {len(loaders['val'])}")
    print(f"  Test batches:  {len(loaders['test'])}")
    
    return loaders


# ============================================================================
# UTILITY: DOWNLOAD INSTRUCTIONS
# ============================================================================
def print_download_instructions():
    """Print instructions for downloading PTB-XL dataset."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              PTB-XL Dataset Download Instructions                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Option 1: Using wget (recommended)                              ║
    ║  $ wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/  ║
    ║                                                                  ║
    ║  Option 2: Using the PhysioNet API                               ║
    ║  $ pip install wfdb                                              ║
    ║  >>> import wfdb                                                 ║
    ║  >>> wfdb.dl_database('ptb-xl', dl_dir='data/ptb-xl')           ║
    ║                                                                  ║
    ║  Option 3: Manual download                                       ║
    ║  Visit: https://physionet.org/content/ptb-xl/1.0.3/             ║
    ║  Download and extract to: data/ptb-xl/                           ║
    ║                                                                  ║
    ║  Expected structure:                                             ║
    ║  data/ptb-xl/                                                    ║
    ║  ├── ptbxl_database.csv                                          ║
    ║  ├── scp_statements.csv                                          ║
    ║  ├── records100/                                                 ║
    ║  └── records500/                                                 ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    print_download_instructions()
