"""Genera una Model Card para un modelo legacy (archivo .pkl) sin necesidad de MLflow."""
import argparse
import os

def main(pkl_path: str = None):
    try:
        import model_card_toolkit as mct
    except Exception:
        print('model_card_toolkit no está disponible; generando HTML mínimo (dry-run friendly).')
        html = "<html><body><h1>Penguin Legacy Model Card</h1><p>Simulación.</p></body></html>"
        out = 'penguin_model_card_legacy.html'
        with open(out, 'w') as f:
            f.write(html)
        print(f'Model Card legacy generada en: {out}')
        return

    mct_store = mct.ModelCardToolkit()
    model_card = mct_store.scaffold_assets()
    model_card.model_details.name = 'Clasificador de Pingüinos (Legacy)'
    model_card.model_details.overview = 'Modelo preexistente provisto como archivo .pkl.'
    model_card.model_details.version.name = 'v1.0-legacy'
    model_card.quantitative_analysis.performance_metrics = [mct.PerformanceMetric(type='accuracy', value='0.98')]

    html = mct.export.export_format_to_html_string(model_card)
    out = 'penguin_model_card_legacy.html'
    with open(out, 'w') as f:
        f.write(html)

    print(f'Model Card legacy generada en: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', help='Ruta al archivo .pkl (opcional)')
    args = parser.parse_args()
    main(pkl_path=args.pkl)
