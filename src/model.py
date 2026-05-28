"""
CNN-LSTM Model Architecture for 12-Lead ECG Classification
============================================================
Implements:
1. CNN-only baseline (Conv1D feature extractor + classifier)
2. CNN-LSTM hybrid (Conv1D + Bidirectional LSTM + Multi-Head Attention + classifier)
3. CNN-GRU variant (Conv1D + Bidirectional GRU + classifier) [for ablation]

Enhanced features:
- Squeeze-and-Excitation (SE) blocks for channel (lead) attention
- Multi-Head Attention for richer temporal aggregation
- Residual connections with proper initialization

Architecture (CNN-LSTM):
    Input: (batch, 12 leads, 5000 timesteps)
    → Conv1D blocks with SE attention (feature extraction)
    → Bidirectional LSTM (temporal modeling)
    → Multi-Head Attention (weighted aggregation)
    → Fully Connected (multi-label classification)
    → Sigmoid (independent class probabilities)

Reference: "Deep Neural Network Architectures for ECG Classification" (2026)
           https://arxiv.org/abs/2602.17701
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# BUILDING BLOCKS
# ============================================================================
class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for channel attention.
    
    Learns to re-weight channels (leads) based on global signal content.
    This helps the model focus on the most informative ECG leads for
    each specific sample.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks" (2018)
    
    Architecture:
        Input: (batch, C, L)
        → Global Average Pooling: (batch, C, 1)
        → FC → ReLU → FC → Sigmoid: (batch, C, 1)  [channel weights]
        → Scale input by channel weights
    
    Parameters
    ----------
    channels : int
        Number of input channels
    reduction : int
        Reduction ratio for the bottleneck (default: 4)
    """
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, channels, length)
        
        Returns
        -------
        scaled : torch.Tensor, shape (batch, channels, length)
            Input scaled by learned channel attention weights
        """
        # Squeeze: global average pooling
        b, c, _ = x.size()
        scale = self.squeeze(x).view(b, c)
        
        # Excitation: channel-wise attention
        scale = self.excitation(scale).view(b, c, 1)
        
        return x * scale


# ============================================================================
# BUILDING BLOCKS
# ============================================================================
class ConvBlock(nn.Module):
    """1D Convolutional block: Conv1D → BatchNorm → ReLU → Dropout → SE."""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, 
                 padding='same', dropout=0.1, use_se=True):
        super().__init__()
        # Calculate padding for 'same' output
        if padding == 'same':
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.se(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection and SE attention for 1D signals."""
    
    def __init__(self, channels, kernel_size=7, dropout=0.1, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Temporal Attention for sequence aggregation.
    
    Instead of single attention, uses multiple heads to capture different
    temporal patterns simultaneously (e.g., P-wave timing, QRS morphology, 
    ST-segment features).
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (LSTM output dim)
    num_heads : int
        Number of attention heads (default: 4)
    dropout : float
        Attention dropout rate
    """
    
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.input_dim = input_dim
        
        # Query, Key projections for attention scoring
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)
            LSTM output sequence
        
        Returns
        -------
        context : torch.Tensor, shape (batch, input_dim)
            Attention-weighted aggregation of the sequence
        attn_weights : torch.Tensor, shape (batch, num_heads, seq_len)
            Attention weights per head (for visualization)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project to Q, K, V
        Q = self.query(x)  # (batch, seq_len, input_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (batch, heads, seq, seq)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted aggregation
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        attn_output = self.output_proj(attn_output)
        
        # Residual connection + layer norm
        attn_output = self.layer_norm(attn_output + x)
        
        # Global aggregation: mean pool over sequence
        context = attn_output.mean(dim=1)  # (batch, input_dim)
        
        # Return average attention across heads for visualization
        avg_attn = attn_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len) - averaged
        
        return context, avg_attn


# ============================================================================
# CNN FEATURE EXTRACTOR
# ============================================================================
class CNNFeatureExtractor(nn.Module):
    """
    1D CNN feature extractor for ECG signals with SE attention.
    
    Progressively downsamples the temporal dimension while increasing channels.
    SE blocks learn which channels/leads are most informative.
    Input: (batch, 12, 5000) → Output: (batch, hidden_dim, reduced_time)
    """
    
    def __init__(self, in_channels=12, hidden_dims=[64, 128, 256], 
                 kernel_sizes=[15, 11, 7], dropout=0.2, use_se=True):
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        for i, (h_dim, k_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            # Conv block with stride 2 for downsampling
            layers.append(ConvBlock(prev_channels, h_dim, kernel_size=k_size, 
                                   stride=2, padding=k_size // 2, dropout=dropout,
                                   use_se=use_se))
            # Residual block to refine features
            layers.append(ResidualBlock(h_dim, kernel_size=k_size, dropout=dropout,
                                       use_se=use_se))
            prev_channels = h_dim
        
        self.features = nn.Sequential(*layers)
        self.output_channels = hidden_dims[-1]
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 12, signal_length)
        
        Returns
        -------
        features : torch.Tensor, shape (batch, hidden_dim, reduced_length)
        """
        return self.features(x)


# ============================================================================
# CNN-ONLY MODEL (BASELINE)
# ============================================================================
class CNNClassifier(nn.Module):
    """
    CNN-only baseline for ECG classification.
    Uses Global Average Pooling after CNN feature extraction.
    
    Architecture:
        Conv1D blocks → Global Average Pooling → FC → Sigmoid
    """
    
    def __init__(self, num_leads=12, num_classes=5, hidden_dims=[64, 128, 256],
                 kernel_sizes=[15, 11, 7], dropout=0.3, fc_dim=128):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor(
            in_channels=num_leads,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # Global Average Pooling + Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (batch, channels, 1)
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 12, 5000)
        
        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
        """
        features = self.cnn(x)       # (batch, 256, reduced_time)
        logits = self.classifier(features)  # (batch, num_classes)
        return logits


# ============================================================================
# CNN-LSTM HYBRID MODEL (MAIN)
# ============================================================================
class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM Hybrid model for 12-lead ECG classification.
    
    Architecture:
        1. Conv1D blocks with SE attention extract local features and downsample
        2. Bidirectional LSTM captures temporal dependencies
        3. Multi-Head Attention mechanism weighs LSTM timesteps
        4. Fully connected layers produce multi-label output
    
    This architecture combines CNN's ability to extract morphological 
    features (QRS complexes, ST segments) with LSTM's ability to model
    temporal patterns across the cardiac cycle. SE blocks learn which
    leads are most diagnostically relevant.
    """
    
    def __init__(self, num_leads=12, num_classes=5, 
                 cnn_hidden_dims=[64, 128, 256],
                 cnn_kernel_sizes=[15, 11, 7],
                 lstm_hidden_dim=128, lstm_num_layers=2,
                 fc_dim=128, dropout=0.3, bidirectional=True,
                 num_attention_heads=4, use_se=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 1. CNN Feature Extractor with SE blocks
        self.cnn = CNNFeatureExtractor(
            in_channels=num_leads,
            hidden_dims=cnn_hidden_dims,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            use_se=use_se
        )
        
        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_hidden_dims[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 3. Multi-Head Attention mechanism
        lstm_output_dim = lstm_hidden_dim * self.num_directions
        self.multi_head_attention = MultiHeadAttention(
            input_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Legacy single-head attention (kept for visualization compatibility)
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Classifier head with deeper structure
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_dim),
            nn.GELU(),  # Smoother activation than ReLU
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass of CNN-LSTM model.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 12, 5000)
            12-lead ECG signals (12 channels, 5000 timesteps at 500Hz)
        
        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
            Raw logits for each class (apply sigmoid for probabilities)
        """
        # Step 1: CNN feature extraction with SE channel attention
        # Input: (batch, 12, 5000) → Output: (batch, 256, ~625)
        cnn_features = self.cnn(x)
        
        # Step 2: Prepare for LSTM (transpose to batch_first format)
        # (batch, channels, time) → (batch, time, channels)
        lstm_input = cnn_features.permute(0, 2, 1)
        
        # Step 3: Bidirectional LSTM
        # Output: (batch, time, lstm_hidden * 2)
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        # Step 4: Multi-Head Attention aggregation
        context, _ = self.multi_head_attention(lstm_output)
        
        # Step 5: Classification
        logits = self.classifier(context)
        
        return logits
    
    def get_attention_weights(self, x):
        """
        Get attention weights for visualization/interpretability.
        Uses single-head attention for cleaner visualization.
        
        Returns
        -------
        attn_weights : torch.Tensor, shape (batch, time_steps)
        """
        with torch.no_grad():
            cnn_features = self.cnn(x)
            lstm_input = cnn_features.permute(0, 2, 1)
            lstm_output, _ = self.lstm(lstm_input)
            attn_weights = self.attention(lstm_output)
            attn_weights = F.softmax(attn_weights, dim=1)
        return attn_weights.squeeze(-1)
    
    def get_multihead_attention_weights(self, x):
        """
        Get multi-head attention weights for detailed analysis.
        
        Returns
        -------
        attn_weights : torch.Tensor, shape (batch, seq_len)
            Averaged attention across all heads
        """
        with torch.no_grad():
            cnn_features = self.cnn(x)
            lstm_input = cnn_features.permute(0, 2, 1)
            lstm_output, _ = self.lstm(lstm_input)
            _, attn_weights = self.multi_head_attention(lstm_output)
        return attn_weights


# ============================================================================
# CNN-GRU MODEL (ABLATION VARIANT)
# ============================================================================
class CNNGRUClassifier(nn.Module):
    """
    CNN-GRU model for ECG classification (ablation comparison).
    
    GRU (Gated Recurrent Unit) is a simpler alternative to LSTM:
    - Fewer parameters (2 gates vs 3)
    - No separate cell state
    - Often faster training
    - Sometimes comparable performance
    
    Used to answer: "Is LSTM's extra complexity worth it for ECG data?"
    """
    
    def __init__(self, num_leads=12, num_classes=5,
                 cnn_hidden_dims=[64, 128, 256],
                 cnn_kernel_sizes=[15, 11, 7],
                 gru_hidden_dim=128, gru_num_layers=2,
                 fc_dim=128, dropout=0.3, bidirectional=True,
                 use_se=True):
        super().__init__()
        
        self.num_directions = 2 if bidirectional else 1
        
        # 1. CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(
            in_channels=num_leads,
            hidden_dims=cnn_hidden_dims,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            use_se=use_se
        )
        
        # 2. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=cnn_hidden_dims[-1],
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 3. Attention
        gru_output_dim = gru_hidden_dim * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        cnn_features = self.cnn(x)
        gru_input = cnn_features.permute(0, 2, 1)
        gru_output, _ = self.gru(gru_input)
        
        # Attention-weighted aggregation
        attn_weights = self.attention(gru_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * gru_output, dim=1)
        
        logits = self.classifier(context)
        return logits
    
    def get_attention_weights(self, x):
        with torch.no_grad():
            cnn_features = self.cnn(x)
            gru_input = cnn_features.permute(0, 2, 1)
            gru_output, _ = self.gru(gru_input)
            attn_weights = self.attention(gru_output)
            attn_weights = F.softmax(attn_weights, dim=1)
        return attn_weights.squeeze(-1)


# ============================================================================
# MODEL SUMMARY UTILITY
# ============================================================================
def count_parameters(model):
    """Count trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def model_summary(model, input_shape=(1, 12, 5000)):
    """Print model summary with parameter counts."""
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Estimated model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    output = model(dummy_input)
    print(f"\nInput shape:  {list(dummy_input.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"{'='*60}\n")
    
    return total_params


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def build_model(model_type='cnn_lstm', num_classes=5, **kwargs):
    """
    Factory function to build ECG classification models.
    
    Parameters
    ----------
    model_type : str
        'cnn_only', 'cnn_lstm', or 'cnn_gru'
    num_classes : int
        Number of output classes
    **kwargs : dict
        Additional model hyperparameters
    
    Returns
    -------
    model : nn.Module
    """
    if model_type == 'cnn_only':
        model = CNNClassifier(num_classes=num_classes, **kwargs)
    elif model_type == 'cnn_lstm':
        model = CNNLSTMClassifier(num_classes=num_classes, **kwargs)
    elif model_type == 'cnn_gru':
        model = CNNGRUClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'cnn_only', 'cnn_lstm', or 'cnn_gru'.")
    
    print(f"Built {model_type} model")
    model_summary(model)
    
    return model


if __name__ == '__main__':
    # Quick test
    print("Testing CNN-only model...")
    cnn_model = build_model('cnn_only')
    
    print("\nTesting CNN-LSTM model...")
    lstm_model = build_model('cnn_lstm')
    
    print("\nTesting CNN-GRU model...")
    gru_model = build_model('cnn_gru')
    
    # Test with dummy data
    dummy = torch.randn(4, 12, 5000)
    out_cnn = cnn_model(dummy)
    out_lstm = lstm_model(dummy)
    out_gru = gru_model(dummy)
    print(f"\nCNN output: {out_cnn.shape}")
    print(f"CNN-LSTM output: {out_lstm.shape}")
    print(f"CNN-GRU output: {out_gru.shape}")
