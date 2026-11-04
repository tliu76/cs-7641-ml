"""
Classification metrics for the UL Report's downstream neural network (Part 4).

Required metrics mirror SL/OL: test Accuracy, Macro-F1, balanced accuracy, and
per-class precision/recall/F1. Macro-F1 and balanced accuracy are central because
Covertype is multiclass and imbalanced -- accuracy alone hides weak minority-class
behaviour, which is exactly what dimensionality reduction may damage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import config as C


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    support: np.ndarray
    confusion: np.ndarray
    val_loss: float | None = None
    extra: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        row = {
            "val_loss": self.val_loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
        }
        for k in range(C.N_CLASSES):
            row[f"f1_c{k+1}"] = float(self.per_class_f1[k])
            row[f"recall_c{k+1}"] = float(self.per_class_recall[k])
        row.update(self.extra)
        return row


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    val_loss: float | None = None) -> ClassificationMetrics:
    labels = list(range(C.N_CLASSES))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        per_class_precision=prec,
        per_class_recall=rec,
        per_class_f1=f1,
        support=sup,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        val_loss=val_loss,
    )
