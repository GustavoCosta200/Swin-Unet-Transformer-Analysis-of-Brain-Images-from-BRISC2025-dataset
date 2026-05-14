from __future__ import annotations

from typing import Dict
import torch


def _binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> Dict[str, float]:
    preds = _binary_predictions(logits, threshold)
    targets = (targets >= 0.5).float()

    dims = tuple(range(1, preds.ndim))
    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1 - targets)).sum(dim=dims)
    fn = ((1 - preds) * targets).sum(dim=dims)
    tn = ((1 - preds) * (1 - targets)).sum(dim=dims)

    dice = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean()
    pixel_acc = ((tp + tn + eps) / (tp + tn + fp + fn + eps)).mean()
    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()

    return {
        "dice": float(dice.detach().cpu()),
        "iou": float(iou.detach().cpu()),
        "pixel_accuracy": float(pixel_acc.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
    }


class MetricAverager:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.count = 0

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self.totals[k] = self.totals.get(k, 0.0) + float(v) * n
        self.count += n

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.totals.items()}
