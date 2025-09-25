"""Carga un modelo generativo pre-entrenado (o simula en dry-run), lo registra en MLflow y crea una Model Card."""
import argparse

def main(dry_run: bool = False):
    if dry_run:
        print("Dry-run: generando Model Card mínima para modelo generativo sin dependencias.")
        html = "<html><body><h1>Generative Model Card (dry-run)</h1><p>Simulación.</p></body></html>"
        out = 'generative_model_card.html'
        with open(out, 'w') as f:
            f.write(html)
        print(f'Model Card generada en: {out}')
        return

    import mlflow
    import model_card_toolkit as mct
    from transformers import pipeline

    generator = pipeline('text-generation', model='gpt2')
    model_name = "gpt2-text-generator"

    with mlflow.start_run() as run:
        try:
            mlflow.transformers.log_model(transformers_model=generator, artifact_path=model_name, registered_model_name=model_name)
        except Exception:
            mlflow.log_param('model_name', model_name)
        run_id = run.info.run_id
        print(f"Registro MLflow Run ID: {run_id}")

    # Generar Model Card
    mct_store = mct.ModelCardToolkit()
    model_card = mct_store.scaffold_assets()
    model_card.model_details.name = 'Generador de Texto Básico'
    model_card.model_details.overview = 'Modelo pre-entrenado de Hugging Face para generación de texto.'
    model_card.model_details.references = [mct.Reference(reference=f'MLflow Run ID: {run_id}')]
    model_card.considerations.limitations = ['No usar para decisiones críticas; revisar contenido generado.']

    html = mct.export.export_format_to_html_string(model_card)
    out = 'generative_model_card.html'
    with open(out, 'w') as f:
        f.write(html)

    print(f'Model Card generada en: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Evita descargas pesadas')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
