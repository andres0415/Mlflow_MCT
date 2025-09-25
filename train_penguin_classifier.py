"""Entrena un clasificador de pingüinos, lo registra en MLflow y genera una Model Card.

Modo de uso:
    python train_penguin_classifier.py [--dry-run]

--dry-run: ejecuta el flujo sin registrar modelos en MLflow ni descargar grandes dependencias.
"""
import argparse
import os
import sys

DRY = False

def main(dry_run: bool = False):
    if dry_run:
        # No requerir dependencias pesadas en dry-run; crear HTML mínimo
        print("Modo dry-run: generando Model Card mínima sin dependencias pesadas.")
        html = "<html><body><h1>Penguin Model Card (dry-run)</h1><p>Simulación.</p></body></html>"
        out = "penguin_model_card.html"
        with open(out, "w") as f:
            f.write(html)
        print(f"Model Card generada en: {out}")
        return

    # Modo completo — importar dependencias pesadas aquí
    import mlflow
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import model_card_toolkit as mct

    print("Cargando datos de penguins...")
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
    penguins = pd.read_csv(url).dropna()

    X = pd.get_dummies(penguins.drop('species', axis=1), drop_first=True)
    y = penguins['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = 100

    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", float(accuracy))

        mlflow.sklearn.log_model(clf, "penguin-classifier-rf")
        run_id = run.info.run_id

        print(f"Run ID: {run_id} - accuracy: {accuracy:.4f}")

    # Generar Model Card usando MCT
    mct_store = mct.ModelCardToolkit()
    model_card = mct_store.scaffold_assets()
    model_card.model_details.name = 'Clasificador de Especies de Pingüinos'
    model_card.model_details.overview = (
        'RandomForestClassifier entrenado para predecir la especie de un pingüino.'
    )
    model_card.model_details.references = [mct.Reference(reference=f'MLflow Run ID: {run_id}')]
    model_card.quantitative_analysis.performance_metrics = [mct.PerformanceMetric(type='accuracy', value=str(round(float(accuracy), 4)))]

    html = mct.export.export_format_to_html_string(model_card)
    out = "penguin_model_card.html"
    with open(out, "w") as f:
        f.write(html)

    print(f"Model Card generada en: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Evita operaciones pesadas y registros en MLflow')
    args = parser.parse_args()
    DRY = args.dry_run
    main(dry_run=DRY)
