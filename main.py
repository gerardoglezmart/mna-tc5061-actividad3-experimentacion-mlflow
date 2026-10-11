"""Punto de entrada: entrena el modelo base y registra la corrida en MLflow.

    python main.py
    python main.py --C 0.1 --class-weight balanced
"""

import argparse
from pathlib import Path
import tempfile
import warnings

import mlflow
import mlflow.sklearn

from wine_quality.config import EXPERIMENT_NAME, TRACKING_URI
from wine_quality.dataset import cargar_datos, dividir_datos, limpiar_datos
from wine_quality.modeling.evaluate import evaluar, guardar_artefactos
from wine_quality.modeling.train import construir_modelo

# La salida del modelo es la clase 0/1 en enteros y no tiene faltantes, así que el aviso de
# MLflow sobre columnas enteras en la firma no aplica
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una regresión logística sobre Wine Quality y registra la "
                    "corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--C", type=float, default=1.0,
                        help="inverso de la fuerza de regularización; menor valor, más regulariza")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="'balanced' da más peso a la clase minoritaria, los vinos buenos")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="máximo de iteraciones del optimizador lbfgs")
    parser.add_argument("--seed", type=int, default=42,
                        help="semilla de la división train/test y del modelo")
    return parser.parse_args()


def main() -> None:
    args = leer_argumentos()

    # Mismo orden del notebook de la semana 3: datos, modelo, evaluación
    X_train, X_test, y_train, y_test = dividir_datos(limpiar_datos(cargar_datos()), args.seed)
    modelo = construir_modelo(args.C, args.class_weight, args.max_iter, args.seed)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    metricas = evaluar(y_test, pred, modelo.predict_proba(X_test)[:, 1])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    nombre = f"C={args.C}_cw={args.class_weight}_iter={args.max_iter}"

    with mlflow.start_run(run_name=nombre) as run:
        for parametro, valor in vars(args).items():
            mlflow.log_param(parametro, valor)
        for metrica, valor in metricas.items():
            mlflow.log_metric(metrica, valor)
        with tempfile.TemporaryDirectory() as carpeta:
            for ruta in guardar_artefactos(y_test, pred, Path(carpeta)):
                mlflow.log_artifact(str(ruta))
        mlflow.sklearn.log_model(modelo, name="model", input_example=X_train.head(3))

    print(f"{nombre}  " + "  ".join(f"{k}={v:.3f}" for k, v in metricas.items()))
    print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
