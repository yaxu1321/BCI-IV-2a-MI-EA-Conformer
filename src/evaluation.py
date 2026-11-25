"""Strict Leave-One-Subject-Out (LOSO) evaluation and ablation engine for BCI IV-2a."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.alignment_aug import euclidean_alignment
from src.data_pipeline import load_bci_iv_2a_all_subjects
from src.gradcam import compute_gradcam_channel_importance
from src.models import DeepMIDecoder, FBCSPRiemannianDecoder
from src.utils import compute_mi_metrics, save_json

logger = logging.getLogger("mi_conformer")


def _write_benchmark_report(
    summary_df: pd.DataFrame,
    gradcam_df: pd.DataFrame,
    n_folds: int,
    out_path: Path,
) -> None:
    lines = [
        "# BCI Competition IV-2a 跨被试（LOSO）运动想象解码与消融汇总",
        "",
        f"- 评估协议：严格留一被试交叉验证（LOSO，评估目标被试数 `{n_folds}`）",
        "",
        "## 1. 模型与欧氏对齐（EA）消融结果汇总",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    if not summary_df.empty:
        best = summary_df.sort_values("accuracy_pct_mean", ascending=False).iloc[0]
        base = summary_df[summary_df["model"] == "EEGConformer_NoEA"]
        base_str = f"由 {base.iloc[0]['accuracy_pct_mean']:.1f}% 提升至 " if not base.empty else ""
        lines.append(
            f"> **最优跨被试结果（{best['model']}）**：四分类跨被试准确率 {base_str}"
            f"**{best['accuracy_pct_mean']:.1f}%**（Cohen's Kappa **{best['kappa_mean']:.3f}**）。"
        )
        lines.append("")

    if not gradcam_df.empty:
        top5 = gradcam_df.sort_values("gradcam_weight_mean", ascending=False).head(5)
        lines.extend(["## 2. Grad-CAM 脑电极重要性 Top-5（验证感觉运动皮层 C3/Cz/C4 聚焦）", ""])
        lines.append(top5.to_markdown(index=False))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_loso_benchmark(
    target_subjects: Sequence[int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run strict LOSO cross-subject benchmark with EA & architecture ablations."""
    results_dir = Path(cfg["project"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_subjects = [int(s) for s in cfg["dataset"]["subjects"]]
    subject_data_raw, ch_names = load_bci_iv_2a_all_subjects(all_subjects, cfg)

    # Pre-compute unsupervised EA independently per subject (zero label usage, zero leakage)
    ea_reg = float(cfg["alignment"]["ea_regularization"])
    subject_data_ea = {
        s: (euclidean_alignment(X, reg=ea_reg), y)
        for s, (X, y) in subject_data_raw.items()
    }

    tcfg = cfg["training"]
    fcfg = cfg["fbcsp_riemannian"]
    acfg = cfg["augmentation"]
    ccfg = cfg["conformer"]
    seed = int(cfg["project"]["seed"])
    sfreq = float(cfg["dataset"]["sfreq"])

    ablation_specs = [
        ("FBCSP_Riemannian_NoEA", "fbcsp", False, False),
        ("FBCSP_Riemannian_EA", "fbcsp", True, False),
        ("ShallowConvNet_EA", "ShallowConvNet", True, False),
        ("EEGNet_NoEA", "EEGNet", False, False),
        ("EEGNet_EA", "EEGNet", True, False),
        ("EEGConformer_NoEA", "EEGConformer", False, False),
        ("EEGConformer_EA_Aug", "EEGConformer", True, True),
    ]

    records: List[Dict[str, Any]] = []
    gradcam_rows: List[Dict[str, Any]] = []

    for test_subj in target_subjects:
        test_subj = int(test_subj)
        train_subjs = [s for s in all_subjects if s != test_subj]
        logger.info("=== LOSO Fold: Target Subject %02d (Train on %d subjects) ===", test_subj, len(train_subjs))

        for exp_name, arch, use_ea, use_aug in ablation_specs:
            pool = subject_data_ea if use_ea else subject_data_raw
            X_train = np.concatenate([pool[s][0] for s in train_subjs], axis=0)
            y_train = np.concatenate([pool[s][1] for s in train_subjs], axis=0)
            X_test, y_test = pool[test_subj]

            if arch == "fbcsp":
                model = FBCSPRiemannianDecoder(
                    sfreq=sfreq,
                    filter_bands=fcfg["filter_bands"],
                    cov_estimator=fcfg["cov_estimator"],
                    classifier_c=fcfg["classifier_c"],
                )
            else:
                model = DeepMIDecoder(
                    arch=arch,
                    n_classes=int(cfg["dataset"]["n_classes"]),
                    epochs=int(tcfg["epochs"]),
                    batch_size=int(tcfg["batch_size"]),
                    lr=float(tcfg["lr"]),
                    weight_decay=float(tcfg["weight_decay"]),
                    val_ratio=float(tcfg["val_ratio"]),
                    patience=int(tcfg["patience"]),
                    use_aug=use_aug,
                    aug_cfg=acfg,
                    conformer_cfg=ccfg,
                    seed=seed,
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_mi_metrics(y_test, y_pred)
            row = {
                "target_subject": test_subj,
                "model": exp_name,
                "use_ea": use_ea,
                "use_aug": use_aug,
                **metrics,
            }
            records.append(row)
            logger.info(
                "[Subject %02d] %-24s | Acc: %5.2f%% | Kappa: %.4f | F1: %.4f",
                test_subj, exp_name, metrics["accuracy_pct"], metrics["kappa"], metrics["macro_f1"],
            )

            # Extract Grad-CAM channel saliency for the proposed EEGConformer_EA_Aug
            if exp_name == "EEGConformer_EA_Aug":
                gcam = compute_gradcam_channel_importance(model, X_test, y_test, ch_names)
                for g in gcam:
                    g["target_subject"] = test_subj
                gradcam_rows.extend(gcam)

        # Checkpoint save after every target subject
        pd.DataFrame(records).to_csv(results_dir / "loso_per_subject.csv", index=False)

    df_all = pd.DataFrame(records)
    summary_df = (
        df_all.groupby(["model", "use_ea", "use_aug"], as_index=False)
        .agg(
            accuracy_pct_mean=("accuracy_pct", "mean"),
            accuracy_pct_std=("accuracy_pct", "std"),
            kappa_mean=("kappa", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
        )
        .round(3)
    )
    summary_df.to_csv(results_dir / "loso_summary.csv", index=False)

    gradcam_summary = pd.DataFrame()
    if gradcam_rows:
        df_g = pd.DataFrame(gradcam_rows)
        df_g.to_csv(results_dir / "gradcam_per_subject.csv", index=False)
        gradcam_summary = (
            df_g.groupby("channel", as_index=False)
            .agg(gradcam_weight_mean=("gradcam_weight", "mean"))
            .sort_values("gradcam_weight_mean", ascending=False)
            .round(5)
        )
        gradcam_summary.to_csv(results_dir / "gradcam_channel_importance.csv", index=False)

    payload = {
        "target_subjects": [int(s) for s in target_subjects],
        "loso_summary": summary_df.to_dict(orient="records"),
        "gradcam_top5": gradcam_summary.head(5).to_dict(orient="records") if not gradcam_summary.empty else [],
    }
    save_json(payload, results_dir / "experiment_summary.json")
    _write_benchmark_report(summary_df, gradcam_summary, len(target_subjects), results_dir / "benchmark_report.md")
    logger.info("All BCI IV-2a results saved to %s", results_dir)
    return payload
