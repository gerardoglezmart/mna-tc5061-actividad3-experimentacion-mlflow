# Actividad 3: Código de experimentación base con MLflow

**Nombre:** Gerardo González Martínez  
**Matrícula:** A01840096  
**Curso:** TC5061.10, Operaciones de aprendizaje automático (MLOps)  
**Programa:** Maestría en Inteligencia Artificial Aplicada  
**Entregable:** [`reports/TC5061_Experimentacion_MLFlow_GonzalezMartinezGerardo.pdf`](reports/TC5061_Experimentacion_MLFlow_GonzalezMartinezGerardo.pdf)

El modelo base del notebook de la semana 3 se migró a un script parametrizado que registra
cada corrida en MLflow. El dataset es Wine Quality (vino tinto) del UCI, donde `quality` es la
calificación de 0 a 10 que asignaron catadores expertos. Como en la semana 3, se trata como
clasificación binaria: un vino es bueno si su calidad es 7 u 8 (13.5 % de los registros tras
eliminar duplicados), un corte habitual para este dataset que se define en `wine_quality/config.py` (`UMBRAL_CALIDAD`).

## Ejecución

Requiere Python 3.13 (probado en Ubuntu 24.04 con Python 3.13.15). Todos los comandos se corren
desde la raíz del repositorio.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py                                   # valores por defecto (baseline de la semana 3)
python main.py --class-weight balanced --C 1.0   # cualquier combinación de argumentos
mlflow ui --port 5000                            # ver las corridas en http://127.0.0.1:5000
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización: a menor valor, más regulariza |
| `--class-weight` | `none` | `balanced` da más peso a la clase minoritaria, los vinos buenos |
| `--max-iter` | `1000` | Máximo de iteraciones del optimizador lbfgs |
| `--seed` | `42` | Semilla de la partición train/test (con el solver lbfgs no altera el modelo) |

`python main.py --help` muestra la misma ayuda.

## Estructura

Generada con Cookiecutter Data Science (`cookiecutter-data-science` 2.3.0, instalado fuera del
ambiente del proyecto porque solo se usa una vez) y recortada a lo que usa la actividad:

```
├── main.py                  # punto de entrada: argumentos, orden de ejecución y registro en MLflow
├── wine_quality/
│   ├── config.py            # rutas, variable objetivo, experimento y tracking URI
│   ├── dataset.py           # carga, limpieza y división estratificada
│   └── modeling/
│       ├── train.py         # Pipeline de StandardScaler y LogisticRegression
│       └── evaluate.py      # métricas, matriz de confusión y classification report
├── notebooks/               # notebook de la semana 3, solo exploración de datos
├── data/raw/                # winequality-red.csv, 1,599 registros
├── reports/                 # PDF entregable, su fuente LaTeX, capturas y corridas_mlflow.csv
├── requirements.txt         # dependencias con versiones fijadas
└── pyproject.toml           # paquete y versión de Python
```

Se quitaron `Makefile`, `.env`, `docs/`, `references/`, `models/` (el modelo se registra en
MLflow), las carpetas de `data/` distintas de `raw/` y los módulos de ejemplo `features.py`,
`plots.py` y `predict.py`. Los módulos generados se reescribieron con `argparse`. `mlflow.db` y
`mlruns/` se crean al correr y no se versionan.

<details>
<summary>Comando exacto de <code>ccds</code></summary>

```bash
pip install cookiecutter-data-science==2.3.0
ccds https://github.com/drivendata/cookiecutter-data-science --no-input \
  project_name=mna-tc5061-actividad3-experimentacion-mlflow module_name=wine_quality \
  "author_name=Gerardo González Martínez" \
  "description=Código de experimentación base con MLflow sobre Wine Quality" \
  python_version_number=3.13 \
  environment_manager=virtualenv dependency_file=requirements.txt pydata_packages=none \
  testing_framework=none linting_and_formatting=ruff "open_source_license=No license file" \
  docs=none include_code_scaffold=Yes
```

</details>

## Corridas y resultados

Las siete corridas del análisis, todas en el experimento `wine-quality-experimentacion`:

| Comando | accuracy | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| `python main.py` | 0.875 | 0.243 | 0.346 | 0.886 |
| `python main.py --C 0.01` | 0.868 | 0.054 | 0.100 | 0.878 |
| `python main.py --C 100` | 0.875 | 0.243 | 0.346 | 0.886 |
| `python main.py --class-weight balanced` | 0.790 | 0.811 | **0.513** | 0.885 |
| `python main.py --class-weight balanced --C 0.1` | 0.787 | 0.811 | 0.508 | 0.884 |
| `python main.py --max-iter 5` | 0.882 | 0.270 | 0.385 | 0.882 |
| `python main.py` (repetida) | 0.875 | 0.243 | 0.346 | 0.886 |

Entre las configuraciones probadas, la mejor por F1 es `--class-weight balanced`. La última
corrida repite la primera y da las mismas métricas sin redondear, como muestra la exportación de
MLflow con sus IDs en `reports/corridas_mlflow.csv`. El análisis completo está en el PDF.

## Declaración de uso de herramientas de inteligencia artificial

En esta entrega se utilizó Claude Code de Anthropic, con el modelo Claude Opus 5.5, para la
generación parcial de código, orientada a mejorar el código propuesto y a plantear
alternativas basadas en decisiones de optimización.

Anthropic. (2026). Claude Code (Claude Opus 5.5) [Modelo de lenguaje grande], utilizado para
generación parcial de código, mejora del código propuesto y alternativas de optimización.
https://claude.com/claude-code
