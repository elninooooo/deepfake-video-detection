"""Evaluation metrics: Acc, Precision, Recall, F1, AUC, EER."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


@dataclass
class EvalReport:
    acc: float
    precision: float
    recall: float
    f1: float
    auc: float
    eer: float
    threshold_eer: float
    n_samples: int

    def to_dict(self):
        return self.__dict__.copy()

    def __str__(self):
        return (f"n={self.n_samples}  "
                f"Acc={self.acc:.4f}  P={self.precision:.4f}  R={self.recall:.4f}  "
                f"F1={self.f1:.4f}  AUC={self.auc:.4f}  EER={self.eer:.4f}")


def compute_eer(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thr[idx])


def evaluate(y_true, y_score, threshold: float = 0.5) -> EvalReport:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    y_pred = (y_score >= threshold).astype(int)

    if y_true.size == 0:
        return EvalReport(0, 0, 0, 0, 0, 0, 0.5, 0)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        # AUC / EER undefined → return safe defaults
        return EvalReport(
            acc=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            auc=float("nan"),
            eer=float("nan"),
            threshold_eer=float("nan"),
            n_samples=int(y_true.size),
        )

    auc = float(roc_auc_score(y_true, y_score))
    eer, thr_eer = compute_eer(y_true, y_score)
    return EvalReport(
        acc=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=auc,
        eer=eer,
        threshold_eer=thr_eer,
        n_samples=int(y_true.size),
    )
