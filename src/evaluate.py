"""
Evaluation and Metrics Visualization Module.
Calculates Accuracy, Precision, Recall, F1-Score, generates classification reports,
confusion matrices, and model comparison bar charts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Compute core classification performance metrics.
    Assumes binary or two-class classification (e.g. FAKE=0, REAL=1 or vice versa).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Binary specific F1 for positive class if applicable
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "F1 (Macro)": round(f1_macro, 4)
    }


def generate_classification_summary(y_true, y_pred, target_names=None):
    """Return text classification report string."""
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)


def plot_confusion_matrices(y_true, models_predictions: dict, class_names, output_path: str):
    """
    Plot side-by-side confusion matrices for all evaluated models.
    models_predictions: dict of {model_name: y_pred_array}
    """
    n_models = len(models_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    sns.set_theme(style="white")
    
    for ax, (model_name, y_pred) in zip(axes, models_predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        # Normalize for percentage annotations
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Annotations with counts and percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
                
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={"size": 11, "fontweight": "bold"}
        )
        ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("True Label", fontsize=10, labelpad=8)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrices plot saved to: {output_path}")


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str):
    """
    Plot comparative grouped bar chart for all models across metrics.
    """
    melted_df = pd.melt(
        metrics_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
    ax = sns.barplot(
        data=melted_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette
    )
    
    plt.title("Model Performance Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Machine Learning Model", fontsize=12, labelpad=10)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12, labelpad=10)
    plt.ylim(0.70, 1.02)  # Focus on the high-performance range for contrast
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    
    # Annotate bars with score values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=0,
                xytext=(0, 3),
                textcoords="offset points"
            )
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Model comparison plot saved to: {output_path}")
