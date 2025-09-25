# Mlflow_MCT
Claro, aquí tienes un archivo `README.md` que explica la integración de MLflow y Model Card Toolkit (MCT) para un caso de uso de IA generativa y otro de Machine Learning tradicional. Incluye ejemplos de código y también explica cómo realizar el registro sin un modelado explícito en MLflow.

---

# Integración de MLflow y Model Card Toolkit para la Gobernanza de Modelos

Este repositorio demuestra cómo integrar **MLflow** para el seguimiento y registro de modelos con **Model Card Toolkit (MCT)** de Google para la documentación y gobernanza de los mismos. Se presentan dos casos de uso:

1.  **Machine Learning Tradicional:** Un modelo de clasificación para predecir la especie de pinguinos.
2.  **IA Generativa:** Un modelo de generación de texto simple.

Para cada caso, se muestra cómo generar una "Model Card" de dos maneras:
*   **A través de Código:** Integrando el seguimiento de MLflow y la generación de la Model Card en un script de entrenamiento.
*   **Sin Entrenamiento Explícito en MLflow:** Creando una Model Card para un modelo que no fue entrenado y registrado directamente a través de una ejecución de MLflow.

## ¿Por qué integrar MLflow y Model Card Toolkit?

*   **MLflow** es excelente para el **seguimiento de experimentos y el registro de modelos**. Permite versionar modelos, gestionar sus etapas (Staging, Producción) y mantener un linaje claro de cómo se crearon.
*   **Model Card Toolkit (MCT)** se enfoca en la **documentación y transparencia**. Ayuda a crear informes estructurados sobre el propósito, el rendimiento, las limitaciones y las consideraciones éticas de un modelo.

Al combinarlos, se obtiene un ciclo de vida de MLOps robusto, donde los modelos no solo están versionados y son desplegables, sino que también están bien documentados para una gobernanza y un uso responsable.

## Prerrequisitos

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install mlflow scikit-learn pandas model-card-toolkit transformers torch
```

## Caso de Uso 1: Machine Learning Tradicional (Clasificación de Pingüinos)

### A. Integración a través de Código

En este escenario, entrenamos un modelo de clasificación, lo registramos en MLflow y generamos su Model Card en un solo script.

**`train_penguin_classifier.py`**```python
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import model_card_toolkit as mct

# --- 1. Entrenamiento y Registro en MLflow ---

# Cargar datos
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
penguins = pd.read_csv(url).dropna()

# Preparar datos
X = pd.get_dummies(penguins.drop('species', axis=1), drop_first=True)
y = penguins['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar una ejecución de MLflow
with mlflow.start_run() as run:
    # Parámetros del modelo
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)

    # Entrenar el modelo
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluar y registrar métricas
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Registrar el modelo
    mlflow.sklearn.log_model(clf, "penguin-classifier-rf")
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")


# --- 2. Generación de la Model Card ---

# Inicializar el Model Card Toolkit
mct_store = mct.ModelCardToolkit()
model_card = mct_store.scaffold_assets()

# Rellenar información del modelo
model_card.model_details.name = 'Clasificador de Especies de Pingüinos'
model_card.model_details.overview = (
    'Este modelo es un RandomForestClassifier entrenado para predecir la especie '
    'de un pingüino (Adelie, Chinstrap, o Gentoo) basándose en sus '
    'características físicas.'
)
model_card.model_details.owners = [
    mct.Owner(name='Equipo de Ciencia de Datos', contact='equipo-ds@example.com')
]
# Enlazar a la ejecución de MLflow
model_card.model_details.references = [
    mct.Reference(reference=f'MLflow Run ID: {run_id}')
]

# Rellenar datos de evaluación
model_card.quantitative_analysis.performance_metrics = [
    mct.PerformanceMetric(type='accuracy', value=str(accuracy))
]

# Generar el HTML de la Model Card
html_content = mct.export.export_format_to_html_string(model_card)
with open("penguin_model_card.html", "w") as f:
    f.write(html_content)

print("Model Card generada en 'penguin_model_card.html'")
```

### B. Creación de Model Card sin Entrenamiento Explícito en MLflow

Imagina que ya tienes un modelo entrenado (`.pkl`) y quieres documentarlo, aunque no fue producto de una ejecución de `mlflow.start_run()`.

```python
import pickle
import model_card_toolkit as mct
from sklearn.ensemble import RandomForestClassifier # Necesario para cargar el pkl

# Supongamos que ya tenemos un modelo guardado
# from train_penguin_classifier import clf
# with open('penguin_model.pkl', 'wb') as f:
#     pickle.dump(clf, f)

# --- Creación de la Model Card ---

mct_store = mct.ModelCardToolkit()
model_card = mct_store.scaffold_assets()

# Rellenar información del modelo (puede ser manual o de un archivo de config)
model_card.model_details.name = 'Clasificador de Pingüinos (Versión Legacy)'
model_card.model_details.overview = (
    'Modelo de clasificación de pingüinos entrenado previamente. '
    'No está asociado a una ejecución de MLflow.'
)
model_card.model_details.version.name = 'v1.0-legacy'

# Rellenar datos de evaluación (supongamos que los conocemos)
model_card.quantitative_analysis.performance_metrics = [
    mct.PerformanceMetric(type='accuracy', value='0.98')
]

# Consideraciones de uso
model_card.considerations.users = ['Biólogos', 'Investigadores de la Antártida']
model_card.considerations.use_cases = ['Clasificación rápida de especies en campo a partir de mediciones.']
model_card.considerations.limitations = ['El modelo solo fue entrenado con datos de 3 especies.']


# Generar el HTML
html_content = mct.export.export_format_to_html_string(model_card)
with open("penguin_model_card_legacy.html", "w") as f:
    f.write(html_content)

print("Model Card para modelo legacy generada en 'penguin_model_card_legacy.html'")```

---

## Caso de Uso 2: IA Generativa (Generador de Texto Simple)

### A. Integración a través de Código

Aquí, "entrenar" puede significar simplemente cargar un modelo pre-entrenado y registrarlo en MLflow como una versión específica para nuestra aplicación.

**`register_generative_model.py`**
```python
import mlflow
from transformers import pipeline
import model_card_toolkit as mct

# --- 1. Carga y Registro del Modelo en MLflow ---

# Usaremos un modelo pequeño pre-entrenado de Hugging Face
generator = pipeline('text-generation', model='gpt2')
model_name = "distilgpt2-text-generator"

# Iniciar una ejecución de MLflow para registrar el artefacto
with mlflow.start_run() as run:
    # Registrar el modelo usando el "flavor" de transformers
    mlflow.transformers.log_model(
        transformers_model=generator,
        artifact_path=model_name,
        registered_model_name=model_name # Registrar en el Model Registry
    )
    run_id = run.info.run_id
    print(f"Modelo registrado en MLflow con Run ID: {run_id}")


# --- 2. Generación de la Model Card ---

mct_store = mct.ModelCardToolkit()
model_card = mct_store.scaffold_assets()

# Rellenar la información del modelo
model_card.model_details.name = 'Generador de Texto Básico'
model_card.model_details.overview = (
    'Este modelo utiliza la arquitectura distilgpt2 de Hugging Face para generar '
    'texto a partir de un prompt inicial.'
)
model_card.model_details.owners = [mct.Owner(name='Equipo de IA Generativa')]
model_card.model_details.references = [
    # Mlflow_MCT

    Este repositorio contiene ejemplos y utilidades para integrar MLflow (seguimiento y registro de modelos) con Model Card Toolkit (MCT) para documentar y gobernar modelos de Machine Learning y modelos de IA generativa.

    Contenido principal:

    - `train_penguin_classifier.py`: Entrena un RandomForest sobre el dataset de penguins, registra la ejecución en MLflow (opcional) y genera una Model Card.
    - `register_generative_model.py`: Ejemplo para registrar un modelo generativo (Hugging Face) en MLflow y generar su Model Card. Soporta `--dry-run` para evitar descargas pesadas.
    - `create_legacy_model_card.py`: Genera una Model Card para un modelo preexistente (pkl) sin necesidad de MLflow.
    - `create_api_model_card.py`: Genera una Model Card para un servicio externo (por ejemplo, OpenAI) describiendo el uso y limitaciones.

    Recomendaciones y mejoras incluidas:

    - Todos los scripts incluyen una opción `--dry-run` o similares para permitir ejecuciones rápidas en entornos de desarrollo.
    - Se agregó `.gitignore` y `requirements.txt` con versiones mínimas para reproducibilidad.
    - Se añadieron tests de smoke para validar los scripts en `--dry-run`.

    Requisitos

    Instala dependencias (se recomienda un venv):

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

    Cómo ejecutar

    - Ejecutar entrenamiento de pingüinos (modo rápido):

    ```bash
    python train_penguin_classifier.py --dry-run
    ```

    - Registrar modelo generativo en modo simulado:

    ```bash
    python register_generative_model.py --dry-run
    ```

    - Generar model card legacy o para API:

    ```bash
    python create_legacy_model_card.py
    python create_api_model_card.py --name gpt-4-0613
    ```

    Ver Model Cards

    Los scripts generan archivos `.html` en la raíz del proyecto. Ábrelos con tu navegador.

    Tests

    Ejecuta los tests de smoke con:

    ```bash
    pytest -q
    ```

    Próximos pasos sugeridos

    - Añadir CI que ejecute los tests y genere artefactos.
    - Documentar pasos para registrar modelos en un MLflow remoto o en un registro central.
    - Añadir ejemplos de despliegue y evaluación más formales (validación cruzada, métricas por clase, fairness).

    ---
    Generado y mejorado automáticamente a partir del README original. Si quieres que añada más ejemplos, CI o despliegues (Docker/GHA), dime qué prefieres.
