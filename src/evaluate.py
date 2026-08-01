"""
Computes a consistent set of evaluation metrics for every model and
produces a comparison table + bar chart saved to OUTPUT_DIR.
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve,
)

import config


def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def save_confusion_matrix(y_true, y_proba, model_name, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Sepsis", "Sepsis"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Sepsis", "Sepsis"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_curves(results, y_true_dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name, y_proba in results.items():
        fpr, tpr, _ = roc_curve(y_true_dict[model_name], y_proba)
        auc = roc_auc_score(y_true_dict[model_name], y_proba)
        ax.plot(fpr, tpr, label=f"{model_name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "roc_curves_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_bar_chart(metrics_df):
    metrics_to_plot = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Accuracy"]
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.set_index("Model")[metrics_to_plot].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Metrics")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "metrics_comparison_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_comparison_report(all_predictions, all_labels):
    """
    all_predictions: dict {model_name: y_proba_array}
    all_labels: dict {model_name: y_true_array}  (test labels used by that model
                 -- tabular models and the LSTM use different row orderings,
                 so each model keeps its own aligned y_true)
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    rows = []
    for model_name, y_proba in all_predictions.items():
        y_true = all_labels[model_name]
        m = compute_metrics(y_true, y_proba)
        m["Model"] = model_name
        rows.append(m)
        save_confusion_matrix(y_true, y_proba, model_name)

    metrics_df = pd.DataFrame(rows)[["Model", "AUROC", "AUPRC", "F1", "Precision", "Recall", "Accuracy"]]
    metrics_df = metrics_df.sort_values("AUROC", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(config.OUTPUT_DIR, "model_comparison_results.csv")
    metrics_df.to_csv(csv_path, index=False)

    save_roc_curves(all_predictions, all_labels)
    save_metrics_bar_chart(metrics_df)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)
    print(f"\nFull results saved to: {config.OUTPUT_DIR}/")

    return metrics_df
