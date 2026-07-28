"""Model evaluation: predictions, headline metrics and per-class breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """Everything the plots and reports need from one evaluated split."""

    split_name: str
    class_names: List[str]
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray
    loss: float = float("nan")
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(
            self.true_labels,
            self.predicted_labels,
            labels=list(range(len(self.class_names))),
        )

    def classification_report(self) -> str:
        return classification_report(
            self.true_labels,
            self.predicted_labels,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )

    def summary(self) -> str:
        lines = [f"{self.split_name} set:"]
        if not np.isnan(self.loss):
            lines.append(f"  loss                {self.loss:.4f}")
        for key, value in self.metrics.items():
            lines.append(f"  {key:<19} {value:.4f}")
        return "\n".join(lines)


def _true_labels(generator, count: int) -> np.ndarray:
    """True labels in the generator's current (unshuffled) order."""
    labels = getattr(generator, "ordered_labels", None)
    if labels is None:
        labels = generator.classes
    return np.asarray(labels)[:count]


def evaluate(
    model: tf.keras.Model,
    generator,
    class_names: Sequence[str] | None = None,
    split_name: str = "Test",
    verbose: int = 1,
) -> EvaluationResult:
    """Predict over a split and compute accuracy, balanced accuracy and kappa.

    The generator must be unshuffled, otherwise predictions and labels will not
    line up.
    """
    if hasattr(generator, "reset"):
        generator.reset()

    print(f"\n{'=' * 70}\nEVALUATING: {split_name}\n{'=' * 70}")
    probabilities = model.predict(generator, verbose=verbose)
    predicted = np.argmax(probabilities, axis=1)
    true = _true_labels(generator, len(predicted))

    names = list(class_names or sorted(generator.class_indices))
    metrics = {
        "accuracy": float(np.mean(predicted == true)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "cohens_kappa": float(cohen_kappa_score(true, predicted)),
        "quadratic_kappa": float(
            cohen_kappa_score(true, predicted, weights="quadratic")
        ),
    }

    try:
        metrics["macro_auc"] = float(
            roc_auc_score(
                tf.keras.utils.to_categorical(true, num_classes=len(names)),
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        # Raised when a class is missing from the split; leave AUC out.
        pass

    if hasattr(generator, "reset"):
        generator.reset()
    evaluated = model.evaluate(generator, verbose=0, return_dict=True)

    result = EvaluationResult(
        split_name=split_name,
        class_names=names,
        true_labels=true,
        predicted_labels=predicted,
        probabilities=probabilities,
        loss=float(evaluated.get("loss", float("nan"))),
        metrics=metrics,
    )

    print("\n" + result.summary())
    print("\nClassification report:")
    print(result.classification_report())
    return result


def per_class_accuracy(result: EvaluationResult) -> Dict[str, float]:
    """Recall for each class, keyed by class name."""
    cm = result.confusion_matrix
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(
            np.diag(cm), support, out=np.zeros(len(support), dtype=float), where=support > 0
        )
    return dict(zip(result.class_names, recall.tolist()))
