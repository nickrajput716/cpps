"""
train_model.py
──────────────
Trains a Temporal CNN → BiLSTM → Self-Attention model on the prepared dataset.

Architecture:
  Input (batch, 30, 6)
  → Conv1D × 2  (temporal pattern extraction)
  → BiLSTM × 2  (sequence modelling)
  → Self-Attention (weight recent/important frames more)
  → Dense head  (regression: 0-100 score  +  binary: panic/no-panic)

Run:
    pip install torch scikit-learn --break-system-packages
    python train_model.py --dataset dataset.npz --out model/panic_model.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# ── Dataset ──────────────────────────────────────────────────────────────────

class PanicDataset(Dataset):
    def __init__(self, X, y_bin, y_score):
        self.X       = torch.tensor(X,       dtype=torch.float32)
        self.y_bin   = torch.tensor(y_bin,   dtype=torch.float32)
        self.y_score = torch.tensor(y_score, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_bin[idx], self.y_score[idx]


# ── Self-Attention ────────────────────────────────────────────────────────────

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


# ── Main Model ────────────────────────────────────────────────────────────────

class PanicDetector(nn.Module):
    """
    Input:  (batch, seq_len=30, features=6)
    Output: score (batch,) in [0,100] and logit (batch,) for BCE
    """
    def __init__(self, input_dim=6, seq_len=30, hidden=128, dropout=0.3):
        super().__init__()

        # ── Temporal CNN: extracts local motion patterns across frames
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── BiLSTM: models long-range temporal dependencies
        self.lstm1 = nn.LSTM(128, hidden, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden * 2, hidden, batch_first=True, bidirectional=True, dropout=dropout)

        # ── Self-attention: highlights the most panic-indicative frames
        self.attention = TemporalSelfAttention(hidden * 2, num_heads=4)

        # ── Aggregation: global average + max pooling concatenated
        self.pool_norm = nn.LayerNorm(hidden * 4)
        self.dropout   = nn.Dropout(dropout)

        # ── Regression head: outputs 0–100 panic score
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 4, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),      # → [0,1], scaled to [0,100] at inference
        )

        # ── Classification head: binary panic / no-panic
        self.cls_head = nn.Sequential(
            nn.Linear(hidden * 4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # raw logit for BCEWithLogitsLoss
        )

    def forward(self, x):
        # x: (B, T, F)
        # CNN expects (B, F, T)
        cnn_out = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)   # → (B, T, 128)

        lstm1_out, _ = self.lstm1(cnn_out)                          # → (B, T, 256)
        lstm2_out, _ = self.lstm2(lstm1_out)                        # → (B, T, 256)

        attn_out = self.attention(lstm2_out)                        # → (B, T, 256)

        # Pooling: average + max over time
        avg_pool = attn_out.mean(dim=1)                             # (B, 256)
        max_pool = attn_out.max(dim=1).values                       # (B, 256)
        pooled   = torch.cat([avg_pool, max_pool], dim=-1)          # (B, 512)

        pooled = self.pool_norm(pooled)
        pooled = self.dropout(pooled)

        score  = self.score_head(pooled).squeeze(-1) * 100.0        # (B,) in [0,100]
        logit  = self.cls_head(pooled).squeeze(-1)                  # (B,) raw

        return score, logit


# ── Focal Loss for imbalanced classes ────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt   = torch.exp(-bce)
        w    = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = w * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data     = np.load(args.dataset)
    X        = data["X"]           # (N, 30, 6)
    y_bin    = data["y"]           # (N,) binary
    y_score  = data["y_score"]     # (N,) 0-100

    # Normalise features across time and batch
    B, T, F  = X.shape
    X_flat   = X.reshape(-1, F)
    scaler   = StandardScaler()
    X_flat   = scaler.fit_transform(X_flat)
    X        = X_flat.reshape(B, T, F).astype(np.float32)

    # Train / val split
    idx      = np.arange(B)
    tr_idx, va_idx = train_test_split(idx, test_size=0.15, stratify=y_bin, random_state=42)

    tr_ds = PanicDataset(X[tr_idx], y_bin[tr_idx], y_score[tr_idx])
    va_ds = PanicDataset(X[va_idx], y_bin[va_idx], y_score[va_idx])

    # Weighted sampler to handle class imbalance
    class_counts = np.bincount(y_bin[tr_idx].astype(int))
    weights      = 1.0 / class_counts[y_bin[tr_idx].astype(int)]
    sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler,   num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,     num_workers=2, pin_memory=True)

    # Model, optimiser, schedulers
    model     = PanicDetector(input_dim=F, seq_len=T).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    mse_loss   = nn.MSELoss()
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"\nTraining {B} samples  |  {len(tr_idx)} train  |  {len(va_idx)} val")
    print(f"Batch {args.batch}  |  LR {args.lr}  |  Epochs {args.epochs}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for X_b, y_b, ys_b in tr_loader:
            X_b, y_b, ys_b = X_b.to(device), y_b.to(device), ys_b.to(device)
            optimizer.zero_grad()

            score, logit = model(X_b)

            # Combined loss: regression (score) + classification (binary)
            loss_reg = mse_loss(score, ys_b)
            loss_cls = focal_loss(logit, y_b)
            loss     = 0.6 * loss_reg / 100.0 + 0.4 * loss_cls   # normalise scales

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item()

        scheduler.step()

        # ── Validate ──
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b, ys_b in va_loader:
                X_b, y_b, ys_b = X_b.to(device), y_b.to(device), ys_b.to(device)
                score, logit   = model(X_b)
                loss_reg       = mse_loss(score, ys_b)
                loss_cls       = focal_loss(logit, y_b)
                va_loss       += (0.6 * loss_reg / 100.0 + 0.4 * loss_cls).item()
                preds          = (torch.sigmoid(logit) > 0.5).float()
                va_correct    += (preds == y_b).sum().item()
                va_total      += len(y_b)

        avg_tr  = tr_loss / len(tr_loader)
        avg_va  = va_loss  / len(va_loader)
        va_acc  = 100.0 * va_correct / va_total

        print(f"Epoch {epoch:3d}/{args.epochs}  |  "
              f"train {avg_tr:.4f}  |  val {avg_va:.4f}  |  val_acc {va_acc:.1f}%  |  "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if avg_va < best_val_loss:
            best_val_loss = avg_va
            torch.save({
                "model_state":  model.state_dict(),
                "scaler":       scaler,
                "input_dim":    F,
                "seq_len":      T,
            }, args.out)
            print(f"  ✓  Saved best model → {args.out}")

    print(f"\nDone.  Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.npz",          help=".npz file from prepare_dataset.py")
    parser.add_argument("--out",     default="model/panic_model.pt", help="Output model path")
    parser.add_argument("--epochs",  type=int,   default=60)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=3e-4)
    args = parser.parse_args()
    train(args)
