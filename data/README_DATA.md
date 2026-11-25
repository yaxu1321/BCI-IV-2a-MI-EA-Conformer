# BCI Competition IV-2a 数据集目录说明

本目录用于缓存通过 MOABB (`BNCI2014_001`) 自动下载的 BCI Competition IV-2a 公开数据集（9 名被试、22 导 EEG、四分类运动想象：左手 / 右手 / 双脚 / 舌头，250 Hz 采样率）。

运行 `python scripts/download_dataset.py` 时，数据将自动缓存至 `data/mne_data/`（已加入 `.gitignore`，不随代码库提交）。
