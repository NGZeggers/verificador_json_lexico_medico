# Unificador y Verificador de Léxico Médico Mexicano

Este proyecto contiene un conjunto de herramientas para unificar, procesar y verificar conceptos médicos en español, con énfasis especial en el léxico médico mexicano. Incluye procesamiento con modelos LLM y verificación de calidad.

## Componentes Principales

### 1. Unificador Híbrido (`unificar_llm_hibrido.py`)
- Versión optimizada para procesar múltiples archivos eficientemente
- Incluye procesamiento por lotes, pre-filtrado por tipo, cache de Jaccard
- Soporte para procesamiento paralelo
- Barra de progreso para monitoreo

### 2. Unificador con LLM (`uni_json_llm.py`)
- Utiliza GPT-OSS:20B para análisis sofisticado de conceptos
- Detección avanzada de duplicados usando LLM
- Análisis semántico de sinónimos
- Cache de consultas LLM

### 3. Unificador Base (`uni_json.py`)
- Versión base sin dependencia de LLM
- Usa similitud de texto y reglas para detectar duplicados
- Procesamiento eficiente de grandes conjuntos de datos

### 4. Verificador de Resultados (`verificador_json.py`)
- Detección de duplicados por ID y términos canónicos
- Análisis de similitud usando distancia Jaccard
- Verificación de integridad de datos y metadata
- Generación de reportes detallados

### 5. Merge de Conceptos (`merge_conceptos.py`)
- Funciones auxiliares para fusionar conceptos
- Manejo de sinónimos etiquetados
- Preservación de metadata y validaciones

## Uso

### Unificador Híbrido
```bash
python unificar_llm_hibrido.py *.json -o resultado.json
python unificar_llm_hibrido.py *.json -o resultado.json --lotes 5 --use-llm
python unificar_llm_hibrido.py *.json -o resultado.json --parallel --workers 4
```

### Unificador con LLM
```bash
python uni_json_llm.py archivos/*.json -o unificado_llm.json
python uni_json_llm.py archivos/*.json -o unificado_llm.json --no-llm
```

### Verificador
```bash
python verificador_json.py resultado.json
python verificador_json.py resultado.json --jaccard 0.6 --verbose
python verificador_json.py resultado.json --export-report reporte.txt
```

## Estructura del JSON

### Formato de Entrada/Salida
```json
{
    "metadata": {
        "total_conceptos": 1000,
        "validados_cie10": 500,
        "validados_cemece": 600,
        "score_mexicanidad_promedio": "0.750",
        "version": "4.0.0",
        ...
    },
    "conceptos": [
        {
            "concepto": "término médico",
            "canonico": "término canónico",
            "concept_id": "ID único",
            "tipo": "tipo de concepto",
            "sinonimos": ["sinónimo 1", "sinónimo 2"],
            "sinonimos_etiquetados": [
                {
                    "texto": "sinónimo",
                    "etiquetas": ["coloquial", "mx"],
                    "score_mx": 0.85,
                    ...
                }
            ],
            "validado_cie10": true,
            "validado_cemece": true,
            "metricas_mx": {
                "promedio_score": 0.75,
                "max_score": 0.85,
                ...
            }
        },
        ...
    ]
}
```

## Requisitos

- Python 3.6+
- Paquetes requeridos:
  - requests (para LLM)
  - tqdm (para barras de progreso)
  - unidecode (para normalización de texto)

Paquetes opcionales:
```bash
pip install unidecode tqdm requests
```

## Características Avanzadas

- Cache global para optimizar rendimiento
- Procesamiento por lotes para archivos grandes
- Detección inteligente de tipos compatibles
- Métricas de mexicanidad
- Soporte para validación CIE-10 y CEMECE
- Fusión inteligente de sinónimos etiquetados
- Preservación de metadata y proveniencia

## Licencia

MIT