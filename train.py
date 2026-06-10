"""
Training script for Thalia CNN-LSTM thesis.
Tests both baseline (ResNet50) and CNN-LSTM models.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import timm

# ============================================================================
# SETUP
# ============================================================================

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")

CFG = {
    'device': DEVICE,
    'task': 'classification',
    'num_classes': 2,
    'epochs': 3,  # 3 for sanity check, 90 for full training
    'batch_size': 2,
    'lr': 1e-5,
    'weight_decay': 1e-2,
    'gradient_clip': 1.0,
    'timeseries_length': 3,
    'n_channels_per_timestep': 9,  # 3 geo + 6 atm
    'model_name': 'cnn_lstm',  # 'baseline' or 'cnn_lstm'
}

print(f"Config: epochs={CFG['epochs']}, lr={CFG['lr']}, batch_size={CFG['batch_size']}")

# ============================================================================
# MODELS
# ============================================================================

class BaselineResNet50(nn.Module):
    """Paper's approach: sees all timesteps as channels."""
    def __init__(self, in_channels=27, num_classes=2):
        super().__init__()
        self.model = timm.create_model(
            'resnet50',
            pretrained=True,
            num_classes=num_classes,
            in_chans=in_channels
        )

    def forward(self, x):
        # x: (B, T*C, H, W) = (B, 27, 512, 512)
        return self.model(x)


class CNNLSTMClassifier(nn.Module):
    """Thesis contribution: CNN on each frame, LSTM over time."""
    def __init__(
        self,
        backbone='resnet50',
        in_channels_per_frame=9,
        timeseries_len=3,
        lstm_hidden=256,
        num_classes=2,
        dropout=0.3,
        pretrained=True
    ):
        super().__init__()
        self.T = timeseries_len
        self.C = in_channels_per_frame

        # CNN backbone
        self.cnn = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
            in_chans=in_channels_per_frame
        )
        cnn_out = self.cnn.num_features  # 2048 for ResNet50

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T*C, H, W)
        B, TC, H, W = x.shape
        T, C = self.T, self.C
        assert TC == T * C

        # Reshape to (B, T, C, H, W)
        x = x.view(B, T, C, H, W)

        # Process each frame through CNN
        x_flat = x.view(B * T, C, H, W)
        feats = self.cnn(x_flat)  # (B*T, cnn_out)
        feats = feats.view(B, T, -1)  # (B, T, cnn_out)

        # LSTM
        lstm_out, _ = self.lstm(feats)  # (B, T, lstm_hidden)

        # Use last hidden state
        last = lstm_out[:, -1, :]

        return self.head(last)


# ============================================================================
# DATA LOADING
# ============================================================================

def create_loaders(shuffle_frames=False):
    """Import and use the fixed data loader."""
    from src.data_loader_fixed import create_loaders as _create_loaders

    data_root  = "/Users/bg/deepLearning/MITDeepLearning/volcano-thesis/data/webdatasets/temporal/3"
    stats_path = "/Users/bg/deepLearning/MITDeepLearning/volcano-thesis/Thalia/statistics.json"

    return _create_loaders(
        data_root=data_root,
        stats_path=stats_path,
        timeseries_length=3,
        batch_size=CFG['batch_size'],
        num_workers=0,
        seed=42,
        shuffle_frames=shuffle_frames,
    )


# ============================================================================
# TRAINING
# ============================================================================

def compute_metrics(all_labels, all_probs):
    """Compute F1, Precision, Recall, AUROC."""
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    if len(np.unique(all_labels)) < 2:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'auroc': 50.0}

    return {
        'precision': precision_score(all_labels, all_preds, zero_division=0) * 100,
        'recall': recall_score(all_labels, all_preds, zero_division=0) * 100,
        'f1': f1_score(all_labels, all_preds, zero_division=0) * 100,
        'auroc': roc_auc_score(all_labels, all_probs) * 100,
    }


def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue
        images, labels, _ = batch
        images = images.to(CFG['device'])
        labels = labels.to(CFG['device'])

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 10 == 0:
            print(f"  Epoch {epoch} | Batch {n_batches:3d} | loss: {loss.item():.4f}")

    print(f"  Epoch {epoch} | Avg loss: {total_loss / n_batches:.4f}")
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_labels, all_probs = [], []

    for batch in loader:
        if batch is None:
            continue
        images, labels, _ = batch
        images = images.to(CFG['device'])
        labels = labels.to(CFG['device'])

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        total_loss += loss.item()
        n_batches += 1

    metrics = compute_metrics(all_labels, all_probs)
    metrics['loss'] = total_loss / max(n_batches, 1)
    return metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default='cnn_lstm', choices=['baseline', 'cnn_lstm'])
    parser.add_argument('--epochs',  type=int, default=CFG['epochs'])
    parser.add_argument('--shuffle', action='store_true', help='Shuffle frames (ablation)')
    args = parser.parse_args()

    CFG['model_name'] = args.model
    CFG['epochs']     = args.epochs

    Path("outputs").mkdir(exist_ok=True)

    shuffle_tag = "_shuffled" if args.shuffle else ""
    run_name    = f"{CFG['model_name']}{shuffle_tag}"

    print("\n" + "=" * 70)
    print(f"Training: {run_name.upper()}  |  epochs={CFG['epochs']}")
    if args.shuffle:
        print("⚠️  ABLATION MODE: frames are shuffled (temporal order destroyed)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_loaders(shuffle_frames=args.shuffle)
    print("✅ Data loaders created")

    # Create model
    print(f"\nCreating {CFG['model_name']} model...")
    if CFG['model_name'] == 'baseline':
        model = BaselineResNet50(
            in_channels=27,
            num_classes=CFG['num_classes']
        )
    elif CFG['model_name'] == 'cnn_lstm':
        model = CNNLSTMClassifier(
            backbone='resnet50',
            in_channels_per_frame=9,
            timeseries_len=3,
            lstm_hidden=256,
            num_classes=CFG['num_classes'],
        )
    else:
        raise ValueError(f"Unknown model: {CFG['model_name']}")

    model = model.to(CFG['device'])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Model created ({n_params:.1f}M parameters)")

    # Optimizer & loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay']
    )

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70 + "\n")

    best_f1 = 0.0
    train_start = time.time()
    for epoch in range(1, CFG['epochs'] + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_metrics = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{CFG['epochs']}")
        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_metrics['loss']:.4f}")
        print(f"  F1         : {val_metrics['f1']:6.2f}%")
        print(f"  Precision  : {val_metrics['precision']:6.2f}%")
        print(f"  Recall     : {val_metrics['recall']:6.2f}%")
        print(f"  AUROC      : {val_metrics['auroc']:6.2f}%")
        print(f"  Time       : {elapsed:.0f}s\n")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), f"outputs/best_{run_name}.pth")
            print(f"  ✅ Best checkpoint saved (F1={best_f1:.1f}%)\n")

    total_elapsed = time.time() - train_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    print("=" * 70)
    print(f"Training complete! Best F1: {best_f1:.1f}%")
    print(f"Total training time: {total_mins}m {total_secs}s")
    print("=" * 70)
