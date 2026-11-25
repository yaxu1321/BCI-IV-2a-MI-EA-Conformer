# BCI Competition IV-2a 跨被试（LOSO）运动想象解码与消融汇总

- 评估协议：严格留一被试交叉验证（LOSO，评估目标被试数 `9`）

## 1. 模型与欧氏对齐（EA）消融结果汇总

| model                 | use_ea   | use_aug   |   accuracy_pct_mean |   accuracy_pct_std |   kappa_mean |   macro_f1_mean |
|:----------------------|:---------|:----------|--------------------:|-------------------:|-------------:|----------------:|
| EEGConformer_EA_Aug   | True     | True      |              51.543 |             15.864 |        0.354 |           0.505 |
| EEGConformer_NoEA     | False    | False     |              43.172 |             17.683 |        0.242 |           0.391 |
| EEGNet_EA             | True     | False     |              50.077 |             16.337 |        0.334 |           0.481 |
| EEGNet_NoEA           | False    | False     |              43.73  |             16.845 |        0.25  |           0.402 |
| FBCSP_Riemannian_EA   | True     | False     |              41.416 |             10.195 |        0.219 |           0.403 |
| FBCSP_Riemannian_NoEA | False    | False     |              33.874 |              6.901 |        0.118 |           0.29  |
| ShallowConvNet_EA     | True     | False     |              49.364 |             14.741 |        0.325 |           0.482 |

> **最优跨被试结果（EEGConformer_EA_Aug）**：四分类跨被试准确率 由 43.2% 提升至 **51.5%**（Cohen's Kappa **0.354**）。

## 2. Grad-CAM 脑电极重要性 Top-5（验证感觉运动皮层 C3/Cz/C4 聚焦）

| channel   |   gradcam_weight_mean |
|:----------|----------------------:|
| Cz        |               0.05913 |
| C3        |               0.05902 |
| CP3       |               0.05482 |
| C1        |               0.05417 |
| C4        |               0.05327 |
