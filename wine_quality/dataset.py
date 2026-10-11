"""Carga, limpieza y división del dataset Wine Quality (vino tinto)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from wine_quality.config import RAW_DATA_FILE, TARGET, TEST_SIZE, UMBRAL_CALIDAD


def cargar_datos(ruta=RAW_DATA_FILE) -> pd.DataFrame:
    """Lee el CSV original del UCI, que viene separado por punto y coma."""
    return pd.read_csv(ruta, sep=";")


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados y crea la variable objetivo binaria.

    No hay faltantes ni tipos incorrectos, y los valores extremos se conservan porque son
    concentraciones plausibles. El análisis está en el notebook de la semana 3:
    notebooks/TC5061_Notebook_Baseline_GonzalezMartinezGerardo.ipynb.
    """
    limpio = df.drop_duplicates().reset_index(drop=True)
    limpio[TARGET] = (limpio["quality"] >= UMBRAL_CALIDAD).astype(int)
    return limpio


def dividir_datos(df: pd.DataFrame, semilla: int):
    """Separa entrenamiento y prueba conservando la proporción de vinos buenos."""
    X = df.drop(columns=["quality", TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=semilla, stratify=y)
