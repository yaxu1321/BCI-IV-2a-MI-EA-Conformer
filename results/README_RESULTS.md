# 实验结果文件说明

本目录包含 9 名被试严格留一被试交叉验证（LOSO）零校准消融评测输出：
- `benchmark_report.md`：7 组模型与欧氏对齐消融结果及 Grad-CAM 电极归因报告
- `loso_summary.csv` / `loso_per_subject.csv`：FBCSP+黎曼、ShallowConvNet、EEGNet、EEG-Conformer 在对齐与增强消融下的准确率、Cohen's Kappa 与 Macro-F1
- `gradcam_channel_importance.csv` / `gradcam_per_subject.csv`：22 导电极 Grad-CAM 通道归因权重统计
- `experiment_summary.json`：结构化汇总指标
