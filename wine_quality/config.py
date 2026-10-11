"""Rutas y constantes del proyecto. Ningún otro módulo escribe una ruta a mano."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_FILE = PROJ_ROOT / "data" / "raw" / "winequality-red.csv"

# Un vino es bueno si los catadores le dieron 7 u 8, el mismo corte de la semana 3
TARGET = "buen_vino"
UMBRAL_CALIDAD = 7
TEST_SIZE = 0.2

EXPERIMENT_NAME = "wine-quality-experimentacion"
TRACKING_URI = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
