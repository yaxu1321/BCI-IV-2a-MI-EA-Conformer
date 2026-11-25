"""Unsupervised Euclidean Alignment (EA) and Time-Frequency EEG Augmentation.

Reference:
- He & Wu (2019), Transfer Learning for Brain-Computer Interfaces: A Euclidean Space
  Data Alignment Approach, IEEE Transactions on Biomedical Engineering.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def euclidean_alignment(X: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    """Apply unsupervised Euclidean Alignment (EA) independently to one subject's trials.

    Computes the arithmetic mean covariance matrix across all trials of a subject:
        R_bar = (1 / N) * sum_i (X_i @ X_i.T) / T
    Then whitens every trial via R_bar^(-1/2) @ X_i so the subject's mean covariance
    becomes the identity matrix I_C, eliminating inter-subject baseline distribution shift
    without using any class labels.

    Args:
        X: EEG array of shape (n_trials, n_channels, n_times).
        reg: Trace-normalized shrinkage regularization for numerical stability.

    Returns:
        X_aligned: shape (n_trials, n_channels, n_times).
    """
    n_trials, n_ch, n_times = X.shape
    covs = np.einsum("nct,ndt->ncd", X, X) / float(max(n_times, 1))
    r_bar = covs.mean(axis=0)
    r_bar = 0.5 * (r_bar + r_bar.T)

    trace_scale = np.trace(r_bar) / max(n_ch, 1)
    r_bar_reg = r_bar + reg * trace_scale * np.eye(n_ch, dtype=np.float64)

    eigvals, eigvecs = eigh(r_bar_reg)
    eigvals = np.maximum(eigvals, 1e-10)
    r_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

    return np.einsum("cd,ndt->nct", r_inv_sqrt, X)


def augment_eeg_batch(
    xb,
    yb=None,
    sr_prob: float = 0.5,
    n_segments: int = 8,
    time_mask_prob: float = 0.2,
    time_mask_max_ratio: float = 0.08,
    freq_mask_prob: float = 0.0,
    noise_std: float = 0.02,
):
    """Apply intra-class Segmentation & Reconstruction (S&R, Song et al., 2023),
    gentle temporal masking, and weak Gaussian noise on a PyTorch batch.

    Args:
        xb: PyTorch tensor of shape (B, 1, C, T) on CPU or GPU.
        yb: Optional class labels tensor of shape (B,) for intra-class S&R.
    """
    import torch

    x = xb.clone()
    B, _, C, T = x.shape

    # 0. Intra-class Segmentation & Reconstruction (Song et al., 2023 EEG-Conformer)
    if yb is not None and sr_prob > 0 and n_segments > 1 and B >= 4:
        seg_edges = torch.linspace(0, T, steps=n_segments + 1, dtype=torch.long)
        x_sr = x.clone()
        for cls in torch.unique(yb):
            cls_idx = torch.where(yb == cls)[0]
            if len(cls_idx) < 2:
                continue
            for i in cls_idx:
                if torch.rand(1).item() < sr_prob:
                    donors = cls_idx[torch.randint(0, len(cls_idx), (n_segments,), device=x.device)]
                    for s_i in range(n_segments):
                        st, ed = int(seg_edges[s_i].item()), int(seg_edges[s_i + 1].item())
                        x_sr[i, :, :, st:ed] = x[donors[s_i], :, :, st:ed]
        x = x_sr

    # 1. Weak Gaussian noise injection
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    # 2. Temporal random masking
    if time_mask_prob > 0:
        max_len = max(1, int(T * time_mask_max_ratio))
        for i in range(B):
            if torch.rand(1).item() < time_mask_prob:
                mask_len = int(torch.randint(1, max_len + 1, (1,)).item())
                start = int(torch.randint(0, max(1, T - mask_len), (1,)).item())
                x[i, :, :, start : start + mask_len] = 0.0

    # 3. Frequency-domain random band masking via real FFT
    if freq_mask_prob > 0:
        x_fft = torch.fft.rfft(x, dim=-1)
        n_freqs = x_fft.shape[-1]
        max_f = max(1, n_freqs // 12)
        for i in range(B):
            if torch.rand(1).item() < freq_mask_prob:
                f_len = int(torch.randint(1, max_f + 1, (1,)).item())
                f_start = int(torch.randint(1, max(2, n_freqs - f_len), (1,)).item())
                x_fft[i, :, :, f_start : f_start + f_len] = 0.0
        x = torch.fft.irfft(x_fft, n=T, dim=-1)

    return x
