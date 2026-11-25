"""BCI Competition IV-2a (BNCI2014_001) data loader via MOABB & MNE-Python."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger("mi_conformer")

CLASS_ORDER = ["left_hand", "right_hand", "feet", "tongue"]


def configure_mne_data_dir(data_dir: str | Path) -> Path:
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(data_path)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(data_path)
    try:
        import mne

        mne.set_config("MNE_DATA", str(data_path), set_env=True)
        mne.set_config("MNE_DATASETS_BNCI_PATH", str(data_path), set_env=True)
    except Exception as exc:
        logger.warning("Could not persist MNE_DATA: %s", exc)
    return data_path


def load_bci_iv_2a_all_subjects(
    subjects: Sequence[int],
    cfg: Dict,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], List[str]]:
    """Load and preprocess BCI Competition IV-2a (22-ch, 4-class MI, 4-40 Hz) for all subjects.

    Uses MOABB MotorImagery paradigm on BNCI2014_001, ensuring consistent channel order,
    4-40 Hz bandpass filtering, and 250 Hz sampling rate.
    """
    try:
        from moabb.datasets import BNCI2014_001 as BCIIV2aDataset
    except ImportError:
        from moabb.datasets import BNCI2014001 as BCIIV2aDataset
    from moabb.paradigms import MotorImagery

    configure_mne_data_dir(cfg["project"]["data_dir"])
    dcfg = cfg["dataset"]

    dataset = BCIIV2aDataset()
    paradigm = MotorImagery(
        n_classes=int(dcfg["n_classes"]),
        fmin=float(dcfg["l_freq"]),
        fmax=float(dcfg["h_freq"]),
        tmin=float(dcfg["tmin"]),
        tmax=float(dcfg["tmax"]),
        resample=float(dcfg["sfreq"]),
    )

    label_to_idx = {name: i for i, name in enumerate(CLASS_ORDER)}
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    ch_names: List[str] = list(dcfg["channels"])

    # Probe exact channel names from subject 1 raw object once
    try:
        raw_s1 = dataset.get_data(subjects=[subjects[0]])[subjects[0]]
        first_sess = next(iter(raw_s1.values()))
        first_run = next(iter(first_sess.values()))
        eeg_chs = [c for c in first_run.ch_names if first_run.get_channel_types(picks=[c])[0] == "eeg"]
        if len(eeg_chs) == 22:
            ch_names = eeg_chs
    except Exception:
        pass

    for subj in subjects:
        logger.info("[BCI IV-2a] Loading Subject %02d...", subj)
        X, y_str, _meta = paradigm.get_data(dataset=dataset, subjects=[subj])
        # Convert microvolt/volt scale consistently to microvolts for numerical stability
        if np.std(X) < 1e-3:
            X = X * 1e6
        y = np.array([label_to_idx[str(lbl)] for lbl in y_str], dtype=np.int64)
        subject_data[int(subj)] = (X.astype(np.float64), y)
        logger.info("[BCI IV-2a] Subject %02d loaded: X=%s, class_counts=%s", subj, X.shape, np.bincount(y).tolist())

    return subject_data, ch_names
