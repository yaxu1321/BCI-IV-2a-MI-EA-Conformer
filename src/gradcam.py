"""Grad-CAM channel-wise interpretability analysis for MI deep networks.

Quantifies gradient-weighted channel importance across all 22 EEG electrodes
(particularly sensorimotor cortex C3 / Cz / C4) as numerical CSV statistics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def compute_gradcam_channel_importance(
    decoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ch_names: Sequence[str],
    max_samples: int = 128,
) -> List[Dict[str, float | str]]:
    """Compute Grad-CAM channel saliency from temporal_conv output (B, F, C, T')."""
    import torch

    model = decoder.model
    device = decoder.device_
    model.eval()

    X_sub = decoder._standardize(X_test[:max_samples], fit=False).astype(np.float32)
    y_sub = y_test[:max_samples]

    activations = []
    gradients = []

    def fwd_hook(_mod, _inp, out):
        activations.append(out)

    def bwd_hook(_mod, _grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.temporal_conv
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        xb = torch.from_numpy(X_sub[:, None, :, :]).to(device)
        yb = torch.from_numpy(y_sub.astype(np.int64)).to(device)
        xb.requires_grad_(True)

        logits = model(xb)
        target_scores = logits.gather(1, yb[:, None]).sum()
        model.zero_grad()
        target_scores.backward()

        act = activations[0].detach()  # (B, F, C, T')
        grad = gradients[0].detach()   # (B, F, C, T')

        # Global average pooling of gradients over time -> channel-feature weights
        weights = grad.mean(dim=-1, keepdim=True)  # (B, F, C, 1)
        cam = torch.relu((weights * act).sum(dim=1))  # (B, C, T')
        ch_scores = cam.mean(dim=(0, 2)).cpu().numpy()  # (C,)
        total = float(ch_scores.sum()) + 1e-12
        ch_norm = ch_scores / total
    finally:
        h_fwd.remove()
        h_bwd.remove()

    rows: List[Dict[str, float | str]] = []
    for ch, sc in zip(ch_names, ch_norm):
        rows.append({"channel": str(ch), "gradcam_weight": round(float(sc), 6)})
    return rows
