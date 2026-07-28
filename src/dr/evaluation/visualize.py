"""Plots for training curves, evaluation results and model internals.

Every function takes an ``output_dir`` and returns the path it wrote to, so
callers decide where figures land instead of relying on a global constant.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from dr.evaluation.evaluate import EvaluationResult

DPI = 300


def _save(fig: plt.Figure, output_dir: str | os.PathLike[str], filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def plot_training_history(
    history, output_dir: str | os.PathLike[str], filename: str = "training_history.png"
) -> Path:
    """Loss, accuracy, balanced accuracy and learning rate over epochs."""
    records = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    panels = [
        (axes[0, 0], "loss", "val_loss", "Loss", 1.0),
        (axes[0, 1], "accuracy", "val_accuracy", "Accuracy (%)", 100.0),
        (
            axes[1, 0],
            "balanced_accuracy",
            "val_balanced_accuracy",
            "Balanced accuracy (%)",
            100.0,
        ),
    ]

    for ax, train_key, val_key, ylabel, scale in panels:
        for key, label in ((train_key, "Train"), (val_key, "Validation")):
            if key in records:
                ax.plot([v * scale for v in records[key]], label=label, linewidth=2)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(ylabel.split(" (")[0], fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    lr_key = next((k for k in ("learning_rate", "lr") if k in records), None)
    if lr_key:
        ax.plot(records[lr_key], linewidth=2, color="purple")
        ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Learning rate", fontweight="bold")
    ax.set_title("Learning rate schedule", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_confusion_matrix(
    result: EvaluationResult,
    output_dir: str | os.PathLike[str],
    normalize: bool = False,
) -> Path:
    """Confusion matrix heatmap, raw counts or row-normalised."""
    cm = result.confusion_matrix.astype(float)
    fmt = "d"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm if normalize else cm.astype(int),
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=result.class_names,
        yticklabels=result.class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        ax=ax,
    )
    ax.set_title(
        f"Confusion matrix - {result.split_name}"
        + (" (normalised)" if normalize else ""),
        fontweight="bold",
    )
    ax.set_ylabel("True label", fontweight="bold")
    ax.set_xlabel("Predicted label", fontweight="bold")
    fig.tight_layout()

    suffix = "_normalized" if normalize else ""
    return _save(
        fig,
        output_dir,
        f"confusion_matrix_{result.split_name.lower()}{suffix}.png",
    )


def plot_roc_curves(
    result: EvaluationResult, output_dir: str | os.PathLike[str]
) -> Path:
    """One-vs-rest ROC curve per class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for index, class_name in enumerate(result.class_names):
        binary_true = (result.true_labels == index).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            continue  # AUC is undefined when a class is absent from the split
        scores = result.probabilities[:, index]
        fpr, tpr, _ = roc_curve(binary_true, scores)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} (AUC = {roc_auc_score(binary_true, scores):.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate", fontweight="bold")
    ax.set_title(f"ROC curves - {result.split_name}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return _save(fig, output_dir, f"roc_curves_{result.split_name.lower()}.png")


def plot_per_class_predictions(
    result: EvaluationResult, output_dir: str | os.PathLike[str]
) -> Path:
    """For each true class, how its samples were distributed across predictions."""
    class_names = result.class_names
    num_classes = len(class_names)
    columns = 3
    rows = int(np.ceil(num_classes / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for index, class_name in enumerate(class_names):
        ax = axes[index]
        mask = result.true_labels == index
        predictions = result.predicted_labels[mask]

        if predictions.size == 0:
            ax.text(0.5, 0.5, f"No samples\nfor {class_name}", ha="center", va="center")
            ax.axis("off")
            continue

        counts = np.bincount(predictions, minlength=num_classes)
        colors = ["#2ecc71" if j == index else "#e74c3c" for j in range(num_classes)]
        bars = ax.bar(range(num_classes), counts, color=colors, alpha=0.8,
                      edgecolor="black")

        correct = int(counts[index])
        accuracy = correct / predictions.size
        ax.set_title(
            f"{class_name}\nrecall {accuracy * 100:.2f}% ({correct}/{predictions.size})",
            fontweight="bold",
        )
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylabel("Count", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, count in zip(bars, counts):
            if count:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    for ax in axes[num_classes:]:
        ax.axis("off")

    fig.tight_layout()
    return _save(
        fig, output_dir, f"per_class_analysis_{result.split_name.lower()}.png"
    )


def plot_sample_predictions(
    model,
    generator,
    class_names: Sequence[str],
    output_dir: str | os.PathLike[str],
    num_samples: int = 16,
) -> Path:
    """Grid of test images captioned with true label, prediction and confidence."""
    if hasattr(generator, "reset"):
        generator.reset()
    images, labels = generator[0]
    images, labels = images[:num_samples], labels[:num_samples]

    probabilities = model.predict(images, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    true = np.argmax(labels, axis=1)

    columns = 4
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for index, ax in enumerate(axes):
        if index >= len(images):
            ax.axis("off")
            continue
        ax.imshow(_to_displayable(images[index]))
        correct = predicted[index] == true[index]
        ax.set_title(
            f"True: {class_names[true[index]]}\n"
            f"Pred: {class_names[predicted[index]]}\n"
            f"Conf: {probabilities[index][predicted[index]]:.2f}",
            color="green" if correct else "red",
            fontweight="bold",
            fontsize=10,
        )
        ax.axis("off")

    fig.tight_layout()
    return _save(fig, output_dir, "sample_predictions.png")


def plot_feature_maps(
    model,
    image: np.ndarray,
    layer_names: Sequence[str],
    output_dir: str | os.PathLike[str],
    title: str = "",
    channels_per_layer: int = 8,
    filename: str = "feature_maps.png",
) -> Path:
    """Activations of selected layers for a single image.

    Useful for checking that the attention gates fire on lesions rather than
    on the black background.

    Args:
        image: A single image with a leading batch dimension, ``(1, H, W, 3)``.
        layer_names: Layers to tap, e.g. ``att5_multiply`` or ``d2_relu2``.
    """
    from tensorflow.keras.models import Model

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected a single image of shape (1, H, W, 3), got {image.shape}")

    activation_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(name).output for name in layer_names],
    )
    activations = activation_model.predict(image, verbose=0)
    if len(layer_names) == 1:
        activations = [activations]

    fig = plt.figure(figsize=(20, (len(layer_names) + 1) * 3))
    grid = fig.add_gridspec(len(layer_names) + 1, 1, hspace=0.4)

    ax = fig.add_subplot(grid[0])
    ax.imshow(_to_displayable(image[0]))
    ax.set_title("Input image", fontsize=14, fontweight="bold")
    ax.axis("off")

    for row, (layer_name, activation) in enumerate(zip(layer_names, activations), 1):
        if activation.ndim == 4:
            shown = min(channels_per_layer, activation.shape[-1])
            inner = grid[row].subgridspec(1, shown, wspace=0.05)
            for column in range(shown):
                ax = fig.add_subplot(inner[0, column])
                ax.imshow(activation[0, :, :, column], cmap="viridis")
                ax.axis("off")
                if column == 0:
                    ax.set_title(
                        f"{layer_name} "
                        f"({activation.shape[1]}x{activation.shape[2]}"
                        f"x{activation.shape[-1]})",
                        loc="left",
                        fontsize=11,
                        pad=10,
                    )
        else:
            ax = fig.add_subplot(grid[row])
            ax.bar(range(activation.shape[-1]), activation[0])
            ax.set_title(
                f"{layer_name} ({activation.shape[-1]} features)", fontsize=11
            )
            ax.set_xticks([])

    if title:
        fig.suptitle(title, fontsize=16, y=0.995)

    return _save(fig, output_dir, filename)


def _to_displayable(image: np.ndarray) -> np.ndarray:
    """Map an image tensor into the 0-1 range matplotlib expects.

    Handles both the ``rescale=1/255`` Keras generators and the ImageNet
    normalisation used by the Albumentations pipelines.
    """
    image = np.asarray(image, dtype=np.float32)
    minimum, maximum = float(image.min()), float(image.max())
    if minimum >= 0.0 and maximum <= 1.0:
        return image
    span = maximum - minimum
    if span <= 0:
        return np.zeros_like(image)
    return (image - minimum) / span
