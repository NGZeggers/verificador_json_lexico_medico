# Verificador de Léxico Médico Mexicano

Este proyecto contiene un verificador de JSON para validar y analizar la unificación de conceptos médicos, especialmente enfocado en el léxico médico mexicano procesado por modelos LLM.

## Características

- Detección de duplicados por ID y términos canónicos
- Análisis de similitud usando distancia Jaccard
- Detección de cruces entre canónicos y sinónimos
- Verificación de integridad de datos y metadata
- Generación de reportes detallados
- Soporte para colores en la salida de terminal
- Modo verbose para debugging

## Uso

```bash
# Verificación básica
python verificador_json.py resultado.json

# Verificación con umbral de Jaccard personalizado
python verificador_json.py resultado.json --jaccard 0.6

# Verificación detallada con reporte
python verificador_json.py resultado.json --verbose --export-report reporte.txt

# Verificación rápida
python verificador_json.py resultado.json --quick
```

## Requisitos

- Python 3.6+
- Paquetes opcionales:
  - unidecode (para normalización de texto)
  - tqdm (para barras de progreso)

## Estructura del JSON esperado

El verificador espera un archivo JSON con la siguiente estructura:

```json
{
    "metadata": {
        "total_conceptos": 1000,
        "validados_cie10": 500,
        "validados_cemece": 600,
        ...
    },
    "conceptos": [
        {
            "concepto": "término médico",
            "canonico": "término canónico",
            "concept_id": "ID único",
            "tipo": "tipo de concepto",
            "sinonimos": ["sinónimo 1", "sinónimo 2"],
            "sinonimos_etiquetados": [...],
            ...
        },
        ...
    ]
}
```

## Licencia

MIT
