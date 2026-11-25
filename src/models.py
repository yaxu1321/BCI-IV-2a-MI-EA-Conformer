"""Motor Imagery decoding models:
1. FBCSP + Riemannian Tangent Space (traditional strong baseline, 4-40 Hz filter bank)
2. ShallowConvNet (Schirrmeister et al., 2017)
3. EEGNet (Lawhern et al., 2018)
4. EEG-Conformer (Song et al., 2023): Temporal-Spatial Conv + Multi-Head Self-Attention
"""

from __future__ import annotations

import copy
from typing import List, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.alignment_aug import augment_eeg_batch


class FBCSPRiemannianDecoder:
    """Filter Bank (4-40 Hz) + Riemannian Tangent Space + Logistic Regression baseline."""

    def __init__(
        self,
        sfreq: float = 250.0,
        filter_bands: Sequence[Sequence[float]] = (
            (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
            (20.0, 24.0), (24.0, 28.0), (28.0, 32.0), (32.0, 36.0), (36.0, 40.0),
        ),
        cov_estimator: str = "lwf",
        classifier_c: float = 1.0,
    ) -> None:
        self.sfreq = float(sfreq)
        self.filter_bands = [(float(b[0]), float(b[1])) for b in filter_bands]
        self.cov_estimator = cov_estimator
        self.classifier_c = float(classifier_c)

        nyq = self.sfreq / 2.0
        self.sos_bands = [
            butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
            for lo, hi in self.filter_bands
        ]
        self.cov_estimators_ = []
        self.ts_transformers_ = []
        self.scaler_ = StandardScaler()
        self.clf_ = LogisticRegression(C=self.classifier_c, max_iter=1000, solver="lbfgs")

    def _extract_band_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        feats = []
        if fit:
            self.cov_estimators_ = []
            self.ts_transformers_ = []

        for idx, sos in enumerate(self.sos_bands):
            Xb = sosfiltfilt(sos, X, axis=-1)
            if fit:
                cov_est = Covariances(estimator=self.cov_estimator)
                covs = cov_est.fit_transform(Xb)
                ts = TangentSpace(metric="riemann")
                z = ts.fit_transform(covs)
                self.cov_estimators_.append(cov_est)
                self.ts_transformers_.append(ts)
            else:
                covs = self.cov_estimators_[idx].transform(Xb)
                z = self.ts_transformers_[idx].transform(covs)
            feats.append(z)
        return np.concatenate(feats, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPRiemannianDecoder":
        Z = self._extract_band_features(X, fit=True)
        Z = self.scaler_.fit_transform(Z)
        self.clf_.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._extract_band_features(X, fit=False)
        Z = self.scaler_.transform(Z)
        return self.clf_.predict(Z)


def build_shallow_convnet(n_channels: int, n_times: int, n_classes: int = 4, dropout: float = 0.5):
    import torch
    import torch.nn as nn

    class ShallowConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal_conv = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False)
            self.spatial_conv = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False)
            self.bn = nn.BatchNorm2d(40)
            self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
            self.drop = nn.Dropout(dropout)
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                out = self._forward_features(dummy)
                feat_dim = out.flatten(1).shape[1]
            self.classifier = nn.Linear(feat_dim, n_classes)

        def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.bn(x)
            x = x * x
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            return self.drop(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self._forward_features(x).flatten(1))

    return ShallowConvNet()


def build_eegnet_mi(n_channels: int, n_times: int, n_classes: int = 4, dropout: float = 0.5):
    import torch
    import torch.nn as nn

    class EEGNetMI(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal_conv = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(8),
            )
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
            )
            self.separable_conv = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
                nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                feat_dim = self.separable_conv(self.spatial_conv(self.temporal_conv(dummy))).flatten(1).shape[1]
            self.classifier = nn.Linear(feat_dim, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            return self.classifier(x.flatten(1))

    return EEGNetMI()


def build_eeg_conformer(
    n_channels: int,
    n_times: int,
    n_classes: int = 4,
    emb_size: int = 40,
    depth: int = 3,
    n_heads: int = 4,
    dropout: float = 0.5,
):
    """EEG-Conformer: Temporal-Spatial Conv Patch Embedding + Multi-Head Self-Attention."""
    import torch
    import torch.nn as nn

    class EEGConformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal_conv = nn.Conv2d(1, emb_size, kernel_size=(1, 25), stride=(1, 1), bias=False)
            self.spatial_conv = nn.Conv2d(emb_size, emb_size, kernel_size=(n_channels, 1), stride=(1, 1), bias=False)
            self.patch_norm_pool = nn.Sequential(
                nn.BatchNorm2d(emb_size),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
                nn.Dropout(dropout),
            )
            self.proj = nn.Conv2d(emb_size, emb_size, kernel_size=(1, 1))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=n_heads,
                dim_feedforward=emb_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.norm = nn.LayerNorm(emb_size)

            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                seq_len = self._patch_embed(dummy).shape[1]

            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, emb_size) * 0.02)
            flat_dim = seq_len * emb_size
            self.head = nn.Sequential(
                nn.Linear(flat_dim, 128),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(128, n_classes),
            )

        def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.patch_norm_pool(x)
            x = self.proj(x)  # (B, E, 1, S)
            return x.squeeze(2).transpose(1, 2)  # (B, S, E)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tokens = self._patch_embed(x) + self.pos_embedding
            tokens = self.norm(self.transformer(tokens))
            return self.head(tokens.flatten(1))

    return EEGConformer()


class DeepMIDecoder:
    """Unified PyTorch trainer for ShallowConvNet, EEGNet, and EEG-Conformer under LOSO."""

    def __init__(
        self,
        arch: str,
        n_classes: int = 4,
        epochs: int = 80,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        val_ratio: float = 0.15,
        patience: int = 15,
        use_aug: bool = False,
        aug_cfg: dict | None = None,
        conformer_cfg: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.arch = arch
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_ratio = val_ratio
        self.patience = patience
        self.use_aug = use_aug
        self.aug_cfg = aug_cfg or {}
        self.conformer_cfg = conformer_cfg or {}
        self.seed = seed

        self.model = None
        self.device_ = None
        self.mean_ = None
        self.std_ = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean_ = X.mean(axis=(0, 2), keepdims=True)
            self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return (X - self.mean_) / self.std_

    def _init_model(self, n_channels: int, n_times: int):
        if self.arch == "ShallowConvNet":
            return build_shallow_convnet(n_channels, n_times, self.n_classes)
        if self.arch == "EEGNet":
            return build_eegnet_mi(n_channels, n_times, self.n_classes)
        if self.arch == "EEGConformer":
            return build_eeg_conformer(
                n_channels,
                n_times,
                self.n_classes,
                emb_size=int(self.conformer_cfg.get("emb_size", 40)),
                depth=int(self.conformer_cfg.get("depth", 3)),
                n_heads=int(self.conformer_cfg.get("n_heads", 4)),
                dropout=float(self.conformer_cfg.get("dropout", 0.5)),
            )
        raise ValueError(f"Unknown architecture: {self.arch}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepMIDecoder":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_norm = self._standardize(X, fit=True).astype(np.float32)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.seed)
        tr_idx, va_idx = next(splitter.split(X_norm, y))

        X_tr, y_tr = X_norm[tr_idx], y[tr_idx]
        X_va, y_va = X_norm[va_idx], y[va_idx]

        self.model = self._init_model(X.shape[1], X.shape[2]).to(self.device_)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr[:, None, :, :]), torch.from_numpy(y_tr.astype(np.int64))),
            batch_size=self.batch_size,
            shuffle=True,
        )
        X_va_t = torch.from_numpy(X_va[:, None, :, :]).to(self.device_)

        best_acc = -1.0
        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                if self.use_aug:
                    xb = augment_eeg_batch(
                        xb,
                        yb=yb,
                        sr_prob=float(self.aug_cfg.get("sr_prob", 0.5)),
                        n_segments=int(self.aug_cfg.get("n_segments", 8)),
                        time_mask_prob=float(self.aug_cfg.get("time_mask_prob", 0.2)),
                        time_mask_max_ratio=float(self.aug_cfg.get("time_mask_max_ratio", 0.08)),
                        freq_mask_prob=float(self.aug_cfg.get("freq_mask_prob", 0.0)),
                        noise_std=float(self.aug_cfg.get("noise_std", 0.02)),
                    )
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_va_t).argmax(dim=1).cpu().numpy()
            val_acc = float(np.mean(val_preds == y_va))
            if val_acc > best_acc + 1e-4:
                best_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        X_norm = self._standardize(X, fit=False).astype(np.float32)
        preds = []
        with torch.no_grad():
            for s in range(0, len(X_norm), 128):
                xb = torch.from_numpy(X_norm[s : s + 128, None, :, :]).to(self.device_)
                preds.append(self.model(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
