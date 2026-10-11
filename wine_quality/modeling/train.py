"""Definición del modelo base: escalado y regresión logística en un solo Pipeline."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def construir_modelo(C: float, class_weight: str, max_iter: int, semilla: int) -> Pipeline:
    """Arma el Pipeline sin entrenar. El escalado va dentro para ajustarse solo con train."""
    return Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", LogisticRegression(
            C=C,
            class_weight=None if class_weight == "none" else class_weight,
            max_iter=max_iter,
            random_state=semilla,
        )),
    ])
