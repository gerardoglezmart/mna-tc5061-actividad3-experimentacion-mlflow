"""Métricas y artefactos de evaluación sobre el conjunto de prueba."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)

ETIQUETAS = ["no bueno", "bueno"]


def evaluar(y_test, pred, proba) -> dict:
    """Con 86 % de vinos no buenos, el accuracy solo no basta: se acompaña de recall, F1 y ROC-AUC."""
    return {
        "accuracy": accuracy_score(y_test, pred),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def guardar_artefactos(y_test, pred, carpeta: Path) -> list[Path]:
    """Guarda la matriz de confusión y el classification report para registrarlos en MLflow."""
    matriz = carpeta / "matriz_confusion.png"
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=ETIQUETAS, ax=ax,
                                            colorbar=False)
    fig.tight_layout()
    fig.savefig(matriz, dpi=110)
    plt.close(fig)

    reporte = carpeta / "classification_report.txt"
    reporte.write_text(classification_report(y_test, pred, target_names=ETIQUETAS,
                                             zero_division=0))
    return [matriz, reporte]
