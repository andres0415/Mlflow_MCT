"""Genera una Model Card para un servicio externo de IA (por ejemplo OpenAI)"""
import argparse

def main(model_name: str = 'external-api-model'):
    try:
        import model_card_toolkit as mct
    except Exception:
        print('model_card_toolkit no disponible; generando HTML mínimo.')
        html = f"<html><body><h1>API Model Card: {model_name}</h1><p>Simulación.</p></body></html>"
        out = 'generative_api_model_card.html'
        with open(out, 'w') as f:
            f.write(html)
        print(f'Model Card de API externa generada en: {out}')
        return

    mct_store = mct.ModelCardToolkit()
    model_card = mct_store.scaffold_assets()
    model_card.model_details.name = 'Integración con API Externa'
    model_card.model_details.overview = 'Uso de un servicio externo para generación de texto.'
    model_card.model_details.version.name = model_name
    model_card.considerations.use_cases = ['Generación de borradores', 'Resumen de transcripciones']
    model_card.considerations.limitations = ['Salida no siempre verificada; revisar antes de publicar.']

    html = mct.export.export_format_to_html_string(model_card)
    out = 'generative_api_model_card.html'
    with open(out, 'w') as f:
        f.write(html)

    print(f'Model Card de API externa generada en: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gpt-4-0613', help='Nombre o versión del modelo/servicio')
    args = parser.parse_args()
    main(model_name=args.name)
