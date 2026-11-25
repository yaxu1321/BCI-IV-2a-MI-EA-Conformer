"""Download and cache BCI Competition IV-2a (BNCI2014_001) via MOABB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import configure_mne_data_dir
from src.utils import load_config, save_json, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BCI Competition IV-2a dataset (9 subjects).")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("mi_download")

    data_dir = Path(cfg["project"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    resolved_dir = configure_mne_data_dir(data_dir)

    try:
        from moabb.datasets import BNCI2014_001 as BCIIV2aDataset
    except ImportError:
        from moabb.datasets import BNCI2014001 as BCIIV2aDataset

    ds = BCIIV2aDataset()
    subjects = [int(s) for s in cfg["dataset"]["subjects"]]
    logger.info("Downloading BNCI2014_001 (BCI IV-2a) for all %d subjects to %s...", len(subjects), resolved_dir)
    ds.download(subject_list=subjects)
    save_json({"data_dir": str(resolved_dir), "subjects": subjects}, resolved_dir.parent / "dataset_manifest.json")
    logger.info("Download completed.")


if __name__ == "__main__":
    main()
