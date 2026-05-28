"""
Evaluation & Visualization Module
===================================
Implements:
- Per-class AUC-ROC computation and plotting
- Optimal threshold search (per-class threshold optimization)
- Confusion matrices (per-class, multi-label)
- Training curves (loss, AUC, learning rate)
- Attention weight visualization
- GradCAM for CNN interpretability
- ECG signal plotting with model predictions
- Statistical significance testing (bootstrap CI, DeLong test)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from scipy.optimize import minimize_scalar
import torch
import torch.nn.functional as F


# Label names for PTB-XL superclasses
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ============================================================================
# METRICS COMPUTATION
# ============================================================================
def compute_metrics(targets, logits, threshold=0.5):
    """
    Compute comprehensive multi-label classification metrics.
    
    Parameters
    ----------
    targets : np.ndarray, shape (n_samples, n_classes)
        Ground truth binary labels
    logits : np.ndarray, shape (n_samples, n_classes)
        Raw model outputs (before sigmoid)
    threshold : float
        Decision threshold for binary predictions
    
    Returns
    -------
    metrics : dict
        Dictionary with all computed metrics
    """
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    metrics = {}
    
    # Per-class AUC-ROC
    per_class_auc = []
    for i, name in enumerate(CLASS_NAMES):
        try:
            auc_score = roc_auc_score(targets[:, i], probs[:, i])
            per_class_auc.append(auc_score)
            metrics[f'auc_{name}'] = auc_score
        except ValueError:
            per_class_auc.append(0.0)
            metrics[f'auc_{name}'] = 0.0
    
    # Macro and micro AUC
    metrics['auc_macro'] = np.mean(per_class_auc)
    try:
        metrics['auc_micro'] = roc_auc_score(targets, probs, average='micro')
    except ValueError:
        metrics['auc_micro'] = 0.0
    
    # F1, Precision, Recall
    metrics['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(targets, preds, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(targets, preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(targets, preds, average='macro', zero_division=0)
    
    # Per-class F1
    for i, name in enumerate(CLASS_NAMES):
        metrics[f'f1_{name}'] = f1_score(targets[:, i], preds[:, i], zero_division=0)
    
    return metrics


def find_optimal_thresholds(targets, logits, metric='f1'):
    """
    Find optimal classification threshold per class.
    
    Instead of using a fixed 0.5 threshold, search for the threshold that
    maximizes F1 score (or other metric) for each class independently.
    
    This is critical for imbalanced datasets where 0.5 may not be optimal.
    
    Parameters
    ----------
    targets : np.ndarray, shape (n_samples, n_classes)
    logits : np.ndarray, shape (n_samples, n_classes)
    metric : str
        Metric to optimize ('f1' or 'youden')
    
    Returns
    -------
    optimal_thresholds : np.ndarray, shape (n_classes,)
    per_class_scores : dict
    """
    probs = 1 / (1 + np.exp(-logits))
    optimal_thresholds = np.zeros(len(CLASS_NAMES))
    per_class_scores = {}
    
    for i, name in enumerate(CLASS_NAMES):
        best_threshold = 0.5
        best_score = 0.0
        
        # Search over thresholds
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (probs[:, i] >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(targets[:, i], preds, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic = sensitivity + specificity - 1
                tp = ((preds == 1) & (targets[:, i] == 1)).sum()
                tn = ((preds == 0) & (targets[:, i] == 0)).sum()
                fp = ((preds == 1) & (targets[:, i] == 0)).sum()
                fn = ((preds == 0) & (targets[:, i] == 1)).sum()
                sensitivity = tp / (tp + fn + 1e-8)
                specificity = tn / (tn + fp + 1e-8)
                score = sensitivity + specificity - 1
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[i] = best_threshold
        per_class_scores[name] = {
            'threshold': best_threshold,
            'score': best_score
        }
    
    return optimal_thresholds, per_class_scores


def compute_metrics_with_optimal_threshold(targets, logits, val_targets=None, val_logits=None):
    """
    Compute metrics using per-class optimized thresholds.
    
    If validation data provided, finds thresholds on validation set and 
    applies them to the target data (proper evaluation protocol).
    
    Parameters
    ----------
    targets : np.ndarray - test targets
    logits : np.ndarray - test logits
    val_targets : np.ndarray, optional - validation targets (for threshold tuning)
    val_logits : np.ndarray, optional - validation logits (for threshold tuning)
    """
    if val_targets is not None and val_logits is not None:
        # Find thresholds on validation set (proper protocol)
        thresholds, _ = find_optimal_thresholds(val_targets, val_logits)
    else:
        # Find thresholds on test set (for analysis only, not proper evaluation)
        thresholds, _ = find_optimal_thresholds(targets, logits)
    
    probs = 1 / (1 + np.exp(-logits))
    preds = np.zeros_like(probs, dtype=int)
    for i in range(len(CLASS_NAMES)):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
    
    metrics = {}
    metrics['optimal_thresholds'] = {CLASS_NAMES[i]: float(thresholds[i]) for i in range(len(CLASS_NAMES))}
    metrics['f1_macro_optimized'] = f1_score(targets, preds, average='macro', zero_division=0)
    metrics['f1_micro_optimized'] = f1_score(targets, preds, average='micro', zero_division=0)
    
    for i, name in enumerate(CLASS_NAMES):
        metrics[f'f1_{name}_optimized'] = f1_score(targets[:, i], preds[:, i], zero_division=0)
    
    return metrics


class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping for 1D CNN.
    
    Visualizes which temporal regions of the ECG signal are most important 
    for a specific class prediction by computing the gradient of the target
    class w.r.t. the last convolutional layer.
    
    Reference: Selvaraju et al., "Grad-CAM" (2017) — adapted for 1D signals.
    
    Parameters
    ----------
    model : nn.Module
        Trained CNN model
    target_layer : nn.Module
        The convolutional layer to compute CAM for (e.g., last conv block)
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, x, target_class):
        """
        Generate GradCAM heatmap for a specific class.
        
        Parameters
        ----------
        x : torch.Tensor, shape (1, 12, 5000)
            Single ECG input
        target_class : int
            Class index to explain
        
        Returns
        -------
        cam : np.ndarray, shape (signal_length,)
            GradCAM heatmap (upsampled to input resolution)
        """
        self.model.eval()
        x.requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        # Target class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Compute CAM
        # Global average pool the gradients
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        
        # Weighted combination of feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, T')
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Upsample to original signal length
        from scipy.signal import resample
        cam_upsampled = resample(cam, x.shape[2])
        
        return cam_upsampled
    
    def plot_gradcam(self, x, target_class, signal=None, save_path=None):
        """
        Plot GradCAM heatmap overlaid on ECG signal.
        
        Parameters
        ----------
        x : torch.Tensor, shape (1, 12, 5000)
        target_class : int
        signal : np.ndarray, optional, shape (12, 5000)
        """
        cam = self.generate(x, target_class)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        
        # Lead II signal with CAM overlay
        if signal is not None:
            lead_ii = signal[1]  # Lead II
        else:
            lead_ii = x.squeeze()[1].detach().cpu().numpy()
        
        time = np.arange(len(lead_ii)) / 500.0  # seconds
        
        axes[0].plot(time, lead_ii, 'k-', linewidth=0.5)
        axes[0].fill_between(time, lead_ii.min(), lead_ii.max(), 
                            where=cam > 0.5, alpha=0.3, color='red',
                            label='High importance')
        axes[0].set_ylabel('Lead II')
        axes[0].set_title(f'GradCAM: Regions important for {CLASS_NAMES[target_class]} prediction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # CAM heatmap
        axes[1].fill_between(time, 0, cam, alpha=0.7, color='red')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Importance')
        axes[1].set_title('GradCAM Activation Intensity')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def delong_test(targets, probs_model_a, probs_model_b):
    """
    DeLong test for comparing two ROC curves.
    
    Tests whether the AUC of model A is statistically significantly 
    different from the AUC of model B.
    
    Simplified implementation using bootstrap-based p-value estimation.
    
    Parameters
    ----------
    targets : np.ndarray, shape (n_samples,)
        Binary ground truth for a single class
    probs_model_a : np.ndarray, shape (n_samples,)
    probs_model_b : np.ndarray, shape (n_samples,)
    
    Returns
    -------
    dict with 'auc_a', 'auc_b', 'p_value', 'significant'
    """
    auc_a = roc_auc_score(targets, probs_model_a)
    auc_b = roc_auc_score(targets, probs_model_b)
    
    # Bootstrap the difference
    n_bootstrap = 2000
    rng = np.random.RandomState(42)
    n = len(targets)
    diffs = []
    
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            auc_a_boot = roc_auc_score(targets[idx], probs_model_a[idx])
            auc_b_boot = roc_auc_score(targets[idx], probs_model_b[idx])
            diffs.append(auc_a_boot - auc_b_boot)
        except ValueError:
            continue
    
    diffs = np.array(diffs)
    # Two-sided p-value
    observed_diff = auc_a - auc_b
    p_value = np.mean(np.abs(diffs - np.mean(diffs)) >= np.abs(observed_diff))
    
    return {
        'auc_a': auc_a,
        'auc_b': auc_b,
        'diff': observed_diff,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def bootstrap_auc_ci(targets, probs, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence intervals for AUC.
    
    Parameters
    ----------
    targets : np.ndarray, shape (n_samples, n_classes)
    probs : np.ndarray, shape (n_samples, n_classes)
    n_bootstrap : int
    ci : float (e.g., 0.95 for 95% CI)
    
    Returns
    -------
    dict with mean, lower, upper AUC per class
    """
    rng = np.random.RandomState(seed)
    n_samples = len(targets)
    alpha = (1 - ci) / 2
    
    results = {}
    for i, name in enumerate(CLASS_NAMES):
        aucs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n_samples, n_samples)
            try:
                score = roc_auc_score(targets[idx, i], probs[idx, i])
                aucs.append(score)
            except ValueError:
                continue
        
        if aucs:
            results[name] = {
                'mean': np.mean(aucs),
                'lower': np.percentile(aucs, alpha * 100),
                'upper': np.percentile(aucs, (1 - alpha) * 100)
            }
    
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/AUC curves.
    
    Parameters
    ----------
    history : dict
        Training history from Trainer
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC curve
    axes[1].plot(epochs, history['train_auc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_auc'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC (macro)')
    axes[1].set_title('Training & Validation AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    plt.show()


def plot_roc_curves(targets, logits, save_path=None):
    """
    Plot per-class ROC curves.
    
    Parameters
    ----------
    targets : np.ndarray, shape (n_samples, n_classes)
    logits : np.ndarray, shape (n_samples, n_classes)
    """
    probs = 1 / (1 + np.exp(-logits))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{name} (AUC = {auc_score:.3f})')
    
    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Per-Class ROC Curves', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    plt.show()


def plot_precision_recall_curves(targets, logits, save_path=None):
    """Plot per-class Precision-Recall curves."""
    probs = 1 / (1 + np.exp(-logits))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        precision, recall, _ = precision_recall_curve(targets[:, i], probs[:, i])
        ap = average_precision_score(targets[:, i], probs[:, i])
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AP = {ap:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Per-Class Precision-Recall Curves', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(targets, logits, threshold=0.5, save_path=None):
    """
    Plot per-class confusion matrices for multi-label classification.
    """
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, (name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        cm = confusion_matrix(targets[:, i], preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.suptitle('Per-Class Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrices to {save_path}")
    plt.show()


def plot_label_distribution(Y, split_name='Dataset', save_path=None):
    """Plot the distribution of labels across classes."""
    counts = Y.sum(axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_xlabel('Diagnostic Superclass')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Label Distribution ({split_name})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ecg_signal(signal, lead_names=LEAD_NAMES, predictions=None, 
                    true_labels=None, fs=500, save_path=None):
    """
    Plot a 12-lead ECG signal with optional predictions.
    
    Parameters
    ----------
    signal : np.ndarray, shape (12, signal_length) or (signal_length, 12)
        ECG signal data
    predictions : np.ndarray, optional
        Model predictions (probabilities)
    true_labels : np.ndarray, optional
        Ground truth labels
    fs : int
        Sampling frequency
    """
    if signal.shape[0] != 12:
        signal = signal.T  # Transpose if needed
    
    signal_length = signal.shape[1]
    time = np.arange(signal_length) / fs  # Time in seconds
    
    fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
    
    for i, (ax, name) in enumerate(zip(axes, lead_names)):
        ax.plot(time, signal[i], 'k-', linewidth=0.5)
        ax.set_ylabel(name, rotation=0, labelpad=30, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    
    # Add title with predictions
    title = '12-Lead ECG'
    if predictions is not None and true_labels is not None:
        pred_str = ', '.join([f'{CLASS_NAMES[i]}:{predictions[i]:.2f}' 
                            for i in range(len(CLASS_NAMES))])
        true_str = ', '.join([CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) 
                            if true_labels[i] == 1])
        title += f'\nPredicted: {pred_str}\nTrue: {true_str}'
    
    plt.suptitle(title, fontsize=12, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_attention_weights(attention_weights, signal=None, fs=500, save_path=None):
    """
    Visualize attention weights from the CNN-LSTM model.
    
    Parameters
    ----------
    attention_weights : np.ndarray, shape (time_steps,)
        Normalized attention weights
    signal : np.ndarray, optional, shape (12, signal_length)
        Original ECG signal for overlay
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Attention weights
    time_steps = np.arange(len(attention_weights))
    axes[0].bar(time_steps, attention_weights, color='red', alpha=0.6)
    axes[0].set_ylabel('Attention Weight')
    axes[0].set_title('LSTM Attention Weights (which parts of the ECG matter most)')
    axes[0].grid(True, alpha=0.3)
    
    # Original signal (Lead II) if provided
    if signal is not None:
        # Downsample signal to match attention length
        signal_lead_ii = signal[1]  # Lead II
        time = np.linspace(0, len(signal_lead_ii) / fs, len(attention_weights))
        # Resample signal to attention resolution
        from scipy.signal import resample
        resampled = resample(signal_lead_ii, len(attention_weights))
        axes[1].plot(time_steps, resampled, 'b-', linewidth=0.8)
        axes[1].set_ylabel('Lead II Amplitude')
        axes[1].set_xlabel('Feature Time Step')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# ABLATION STUDY HELPERS
# ============================================================================
def compare_models(results_dict, save_path=None):
    """
    Compare multiple model results in a bar chart.
    
    Parameters
    ----------
    results_dict : dict
        {model_name: {'auc_macro': float, 'auc_NORM': float, ...}}
    """
    model_names = list(results_dict.keys())
    
    # Per-class AUC comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(model_names)
    
    for i, (model_name, metrics) in enumerate(results_dict.items()):
        aucs = [metrics.get(f'auc_{name}', 0) for name in CLASS_NAMES]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, aucs, width, label=model_name)
    
    ax.set_xlabel('Diagnostic Class')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model Comparison: Per-Class AUC-ROC')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_metrics_table(metrics, model_name='Model'):
    """Print metrics in a formatted table."""
    print(f"\n{'='*50}")
    print(f"  {model_name} - Evaluation Metrics")
    print(f"{'='*50}")
    print(f"  {'Metric':<25} {'Value':<10}")
    print(f"  {'-'*35}")
    
    # Overall metrics
    print(f"  {'AUC (macro)':<25} {metrics['auc_macro']:.4f}")
    print(f"  {'AUC (micro)':<25} {metrics['auc_micro']:.4f}")
    print(f"  {'F1 (macro)':<25} {metrics['f1_macro']:.4f}")
    print(f"  {'Precision (macro)':<25} {metrics['precision_macro']:.4f}")
    print(f"  {'Recall (macro)':<25} {metrics['recall_macro']:.4f}")
    
    print(f"\n  {'Per-Class AUC:'}")
    print(f"  {'-'*35}")
    for name in CLASS_NAMES:
        print(f"  {name:<25} {metrics[f'auc_{name}']:.4f}")
    
    print(f"{'='*50}\n")


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    n = 500
    targets = np.random.randint(0, 2, (n, 5))
    logits = np.random.randn(n, 5)
    
    metrics = compute_metrics(targets, logits)
    print_metrics_table(metrics, 'Test Model')
