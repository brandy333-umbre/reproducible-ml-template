
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_synthetic_classification(
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    class_sep: float,
) -> DatasetBundle:
    X, y = make_classification(
        n_samples=n_train + n_val + n_test,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=seed,
    )

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(n_val + n_test), random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=n_test, random_state=seed, stratify=y_tmp
    )

    return DatasetBundle(X_train, y_train, X_val, y_val, X_test, y_test)
