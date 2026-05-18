"""
ml_predictor.py  (PyTorch 2.6 compatible, warning-free)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class PanicDetector(nn.Module):
    def __init__(self, input_dim=6, seq_len=30, hidden=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        # dropout=0.0 on single-layer LSTMs (PyTorch only applies LSTM dropout
        # between layers, so it's a no-op on num_layers=1 and causes a warning).
        # We apply dropout explicitly after each LSTM instead.
        self.lstm1      = nn.LSTM(128, hidden, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm1_drop = nn.Dropout(dropout)
        self.lstm2      = nn.LSTM(hidden * 2, hidden, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm2_drop = nn.Dropout(dropout)

        self.attention  = TemporalSelfAttention(hidden * 2, num_heads=4)
        self.pool_norm  = nn.LayerNorm(hidden * 4)
        self.dropout    = nn.Dropout(dropout)
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 4, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden * 4, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        cnn_out      = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm1_out, _ = self.lstm1(cnn_out)
        lstm1_out    = self.lstm1_drop(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out    = self.lstm2_drop(lstm2_out)
        attn_out     = self.attention(lstm2_out)
        avg_pool     = attn_out.mean(dim=1)
        max_pool     = attn_out.max(dim=1).values
        pooled       = self.pool_norm(torch.cat([avg_pool, max_pool], dim=-1))
        pooled       = self.dropout(pooled)
        score        = self.score_head(pooled).squeeze(-1) * 100.0
        logit        = self.cls_head(pooled).squeeze(-1)
        return score, logit


class MLPanicPredictor:
    MODEL_PATH = "model/panic_model.pt"
    WINDOW     = 30

    def __init__(self, model_path=None):
        self.model_path = model_path or self.MODEL_PATH
        self._model     = None
        self._scaler    = None
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded    = False
        self._try_load()

    def _try_load(self):
        if not os.path.exists(self.model_path):
            print(f"[MLPredictor] No model at '{self.model_path}'. Using formula fallback.")
            return
        try:
            checkpoint  = torch.load(self.model_path, map_location=self._device, weights_only=False)
            input_dim   = checkpoint.get("input_dim", 6)
            seq_len     = checkpoint.get("seq_len",   30)
            self._model = PanicDetector(input_dim=input_dim, seq_len=seq_len).to(self._device)
            self._model.load_state_dict(checkpoint["model_state"], strict=False)
            self._model.eval()
            self._scaler = checkpoint["scaler"]
            self._loaded = True
            print(f"[MLPredictor] Model loaded from '{self.model_path}' on {self._device}.")
        except Exception as e:
            print(f"[MLPredictor] Failed to load model: {e}. Using formula fallback.")

    @property
    def is_loaded(self):
        return self._loaded

    def predict(self, feature_buffer: list):
        if not self._loaded or len(feature_buffer) < self.WINDOW:
            return None
        window = np.array(feature_buffer[-self.WINDOW:], dtype=np.float32)
        flat   = self._scaler.transform(window.reshape(-1, window.shape[-1]))
        tensor = torch.tensor(flat.reshape(1, self.WINDOW, -1), dtype=torch.float32).to(self._device)
        with torch.no_grad():
            score, _ = self._model(tensor)
        return float(np.clip(score.item(), 0, 100))


predictor = MLPanicPredictor()