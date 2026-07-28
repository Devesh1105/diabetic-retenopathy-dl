"""Dataset preparation, inspection and generators.

``dr.data.analysis`` is plain Python and imports cheaply. The generator and
preprocessing helpers pull in TensorFlow, scikit-learn and Pillow, so they are
resolved lazily: ``from dr.data import create_generators`` still works, but
merely importing this package does not load the deep-learning stack.
"""

from importlib import import_module
from typing import Any

from dr.data.analysis import (
    class_distribution,
    find_minority_classes,
    imbalance_report,
    minority_class_indices,
    verify_labels,
)

_LAZY_EXPORTS = {
    "AlbumentationsGenerator": "dr.data.generators",
    "Generators": "dr.data.generators",
    "compute_class_weights": "dr.data.generators",
    "create_generators": "dr.data.generators",
    "prepare_dataset": "dr.data.preprocessing",
    "verify_dimensions": "dr.data.preprocessing",
}

__all__ = [
    "class_distribution",
    "find_minority_classes",
    "imbalance_report",
    "minority_class_indices",
    "verify_labels",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        return getattr(import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
