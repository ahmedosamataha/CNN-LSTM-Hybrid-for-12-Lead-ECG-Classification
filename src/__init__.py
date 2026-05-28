"""
ECG Classification Source Package
CNN-LSTM Hybrid for 12-Lead ECG Classification

Enhanced with:
- Squeeze-and-Excitation channel attention
- Multi-Head Temporal Attention
- Bandpass filtering & signal preprocessing
- Mixup augmentation
- Label smoothing
- EMA weight averaging
- GradCAM interpretability
- Optimal threshold search
- CNN-GRU ablation variant
"""

from .model import CNNClassifier, CNNLSTMClassifier, CNNGRUClassifier, build_model
from .data_loader import (
    load_ptbxl_data, normalize_ecg, split_data_by_patient, create_dataloaders,
    bandpass_filter, notch_filter, preprocess_signal
)
from .train import Trainer, get_default_config, set_seed, compute_pos_weights
from .evaluate import (
    compute_metrics, print_metrics_table, find_optimal_thresholds,
    compute_metrics_with_optimal_threshold, GradCAM1D, delong_test
)
