# 基于无监督欧氏域对齐与改进型 EEG-Conformer 的跨被试运动想象解码

本项目针对运动想象（Motor Imagery, MI）脑电跨个体协方差分布偏移大、新用户零校准（Zero-Calibration）直接迁移性能显著退化的问题，基于经典公开数据集 **BCI Competition IV-2a（9 名被试，22 导四分类：左手 / 右手 / 双脚 / 舌头）** 开展前端无监督欧氏空间对齐（EA）、同类试次时序分段重组增强（S&R）、改进型 `EEG-Conformer` 混合架构消融及 Grad-CAM 通道可解释性研究。

---

## 1. 核心算法与架构改进

1. **纯净运动想象时间窗与多谱系基线（`src/data_pipeline.py`, `src/models.py`）**：
   - 基于 MOABB（`BNCI2014_001`）与 MNE-Python 进行 4–40 Hz 零相位带通滤波，截取提示后 `[0.5s, 4.0s]` 纯净运动想象段（250 Hz 采样），剔除 `[0.0s, 0.5s]` 视觉提示诱发电位伪迹；
   - 构建 9 子带（`4–8 Hz` 至 `36–40 Hz`，4 Hz 步长）`Ledoit-Wolf 收缩协方差 + 黎曼切空间` 传统几何基线，以及 `ShallowConvNet`、`EEGNet` 深度对照模型。
2. **无监督欧氏对齐与同类分段重组增强（`src/alignment_aug.py`）**：
   - **欧氏空间对齐（EA）**：按每名被试独立计算全试次均值协方差 $\bar{R}$ 并执行白化变换 $\tilde{X}_i = \bar{R}^{-1/2} X_i$，使各被试二阶空间协方差统一对齐至单位阵（全程不使用任何类别标签，严格满足零泄露跨被试假设）；
   - **时序分段重组（Segmentation & Reconstruction, S&R）**：在训练批内将同类试次等分为 8 段并在对应时相随机重组，配合轻量随机时域掩码与微弱高斯噪声，有效扩充跨被试样本多样性。
3. **保相位展平头改进型 `EEG-Conformer`（`src/models.py`）**：
   - 采用时间卷积与全导联空间卷积构成 Patch 嵌入层提取局部 ERD/ERS 节律表征，接入 Pre-LN 多头自注意力 Transformer 编码器建模全局时序依赖；
   - 使用**保相位时序展平分类头（Phase-Preserving Flattened Head）**替代全局时间均值池化，完整保留 3.5s 运动想象窗口内的时相结构。
4. **Grad-CAM 电极通道归因（`src/gradcam.py`）**：
   - 对测试被试计算梯度加权通道激活映射，量化 22 导电极在运动想象判别中的贡献权重。

---

## 2. 严格 9 折留一被试（LOSO）消融实验结果

完整评测明细见 [`results/benchmark_report.md`](./results/benchmark_report.md)、[`results/loso_summary.csv`](./results/loso_summary.csv) 与 [`results/gradcam_channel_importance.csv`](./results/gradcam_channel_importance.csv)。

### 2.1 模型架构与欧氏对齐消融对比（9 折 LOSO 均值 ± 标准差，四分类随机基线 25.0%）

| 实验配置 | 无监督 EA 对齐 | S&R 时序增强 | 跨被试准确率 (%) | Cohen's Kappa | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `FBCSP_Riemannian_NoEA` | 否 | 否 | 33.87 ± 6.90% | 0.118 | 0.290 |
| `FBCSP_Riemannian_EA` | 是 | 否 | 41.42 ± 10.20% | 0.219 | 0.403 |
| `EEGNet_NoEA` | 否 | 否 | 43.73 ± 16.85% | 0.250 | 0.402 |
| `EEGConformer_NoEA` | 否 | 否 | 43.17 ± 17.68% | 0.242 | 0.391 |
| `ShallowConvNet_EA` | 是 | 否 | 49.36 ± 14.74% | 0.325 | 0.482 |
| `EEGNet_EA` | 是 | 否 | 50.08 ± 16.34% | 0.334 | 0.481 |
| **`EEGConformer_EA_Aug`** | **是** | **是** | **51.54 ± 15.86%** | **0.354** | **0.505** |

> 引入无监督 EA 对齐与 S&R 时序增强后，`EEG-Conformer` 四分类零校准跨被试准确率由未对齐的 **43.2% 提升至 51.5%**（+8.4%），相较传统未对齐 `FBCSP + 黎曼` 基线（33.9%）提升 **+17.6%**。

### 2.2 Grad-CAM 脑电极重要性 Top-5 排名

| 排名 | 导联名称 | 所属皮层区域 | 全 9 折平均归因权重 |
| :---: | :---: | :--- | :---: |
| 1 | **`Cz`** | 中央中线感觉运动皮层（下肢/双脚代表区） | 0.05913 |
| 2 | **`C3`** | 左侧感觉运动皮层（右手代表区） | 0.05902 |
| 3 | **`CP3`** | 左侧中央顶区感觉运动联合皮层 | 0.05482 |
| 4 | **`C1`** | 左侧中央旁中线运动皮层 | 0.05417 |
| 5 | **`C4`** | 右侧感觉运动皮层（左手代表区） | 0.05327 |

---

## 3. 运行方式

```bash
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/run_experiments.py --mode full
```
