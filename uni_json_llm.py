#!/usr/bin/env python3
"""
Unificador de JSONs médicos con detección de duplicados usando LLM.
Versión con GPT-OSS:20B para análisis más sofisticado de conceptos y sinónimos.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import argparse
import requests
from time import sleep


class LLMAnalyzer:
    """Clase para análisis de conceptos usando LLM."""
    
    def __init__(self, api_url: str = "http://localhost:11434/api/generate", 
                 model: str = "gpt-oss:20b"):
        """
        Inicializa el analizador LLM.
        
        Args:
            api_url: URL de la API del LLM (Ollama por defecto)
            model: Modelo a usar
        """
        self.api_url = api_url
        self.model = model
        self.cache_analisis = {}  # Cache para evitar llamadas repetidas
    
    def consultar_llm(self, prompt: str, temperatura: float = 0.1) -> str:
        """
        Consulta al LLM con un prompt específico.
        
        Args:
            prompt: Texto del prompt
            temperatura: Temperatura para la generación
            
        Returns:
            Respuesta del LLM
        """
        # Verificar cache
        cache_key = hash(prompt)
        if cache_key in self.cache_analisis:
            return self.cache_analisis[cache_key]
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperatura,
                "stream": False,
                "format": "json"
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                respuesta = response.json().get('response', '{}')
                self.cache_analisis[cache_key] = respuesta
                return respuesta
            else:
                print(f"Error en LLM: {response.status_code}")
                return '{}'
                
        except Exception as e:
            print(f"Error conectando con LLM: {e}")
            return '{}'
    
    def son_conceptos_duplicados_llm(self, concepto1: Dict, concepto2: Dict) -> Tuple[bool, float, str]:
        """
        Usa el LLM para determinar si dos conceptos son duplicados.
        
        Args:
            concepto1: Primer concepto
            concepto2: Segundo concepto
            
        Returns:
            Tupla (son_duplicados, confianza, razon)
        """
        prompt = f"""Analiza si estos dos conceptos médicos son duplicados o se refieren a lo mismo.
        
Concepto 1:
- Nombre: {concepto1.get('concepto', '')}
- Canónico: {concepto1.get('canonico', '')}
- Tipo: {concepto1.get('tipo', '')}
- Sinónimos principales: {', '.join(concepto1.get('sinonimos', [])[:5])}

Concepto 2:
- Nombre: {concepto2.get('concepto', '')}
- Canónico: {concepto2.get('canonico', '')}
- Tipo: {concepto2.get('tipo', '')}
- Sinónimos principales: {', '.join(concepto2.get('sinonimos', [])[:5])}

Responde en formato JSON con esta estructura exacta:
{{
    "son_duplicados": true/false,
    "confianza": 0.0-1.0,
    "razon": "explicación breve",
    "concepto_principal": "nombre del concepto que debería mantenerse"
}}

Considera que pueden ser duplicados si:
- Se refieren a la misma estructura anatómica
- Uno es variante del otro (ej: singular/plural, masculino/femenino)
- Son formas diferentes de referirse a lo mismo
- Comparten la mayoría de sinónimos

NO son duplicados si:
- Son conceptos relacionados pero distintos (ej: abdomen vs abdominal)
- Uno es parte del otro pero no son lo mismo
- Tienen significados médicos diferentes"""

        respuesta = self.consultar_llm(prompt)
        
        try:
            resultado = json.loads(respuesta)
            return (
                resultado.get('son_duplicados', False),
                resultado.get('confianza', 0.0),
                resultado.get('razon', '')
            )
        except:
            return (False, 0.0, 'Error en análisis')
    
    def analizar_sinonimos_duplicados(self, sinonimos1: List[str], sinonimos2: List[str]) -> Dict:
        """
        Analiza qué sinónimos son realmente únicos entre dos listas.
        
        Args:
            sinonimos1: Primera lista de sinónimos
            sinonimos2: Segunda lista de sinónimos
            
        Returns:
            Diccionario con análisis de sinónimos
        """
        prompt = f"""Analiza estas dos listas de sinónimos médicos y determina cuáles son únicos y cuáles son duplicados o variantes.

Lista 1: {json.dumps(sinonimos1, ensure_ascii=False)}
Lista 2: {json.dumps(sinonimos2, ensure_ascii=False)}

Identifica:
1. Sinónimos que son claramente el mismo término (idénticos o con mínimas variaciones)
2. Sinónimos que son variantes del mismo concepto (ej: barriga/barriguita)
3. Sinónimos verdaderamente únicos en cada lista

Responde en formato JSON:
{{
    "duplicados_exactos": ["lista de términos idénticos"],
    "variantes": [
        {{"termino1": "término de lista 1", "termino2": "término de lista 2", "relacion": "explicación"}}
    ],
    "unicos_lista1": ["términos únicos de lista 1"],
    "unicos_lista2": ["términos únicos de lista 2"],
    "sinonimos_unificados": ["lista final sin duplicados reales"]
}}"""

        respuesta = self.consultar_llm(prompt)
        
        try:
            return json.loads(respuesta)
        except:
            # Fallback si falla el LLM
            return {
                "sinonimos_unificados": list(set(sinonimos1 + sinonimos2))
            }


class ConceptoUnificadorLLM:
    """Clase para unificar conceptos médicos usando LLM."""
    
    def __init__(self, usar_llm: bool = True, umbral_confianza: float = 0.7):
        """
        Inicializa el unificador con LLM.
        
        Args:
            usar_llm: Si usar o no el LLM
            umbral_confianza: Umbral de confianza del LLM para considerar duplicados
        """
        self.usar_llm = usar_llm
        self.umbral_confianza = umbral_confianza
        self.conceptos_unificados = []
        self.metadata_global = None
        
        if usar_llm:
            self.llm = LLMAnalyzer()
            print("✓ LLM inicializado (GPT-OSS:20B)")
        else:
            self.llm = None
            print("⚠ Modo sin LLM - usando análisis básico")
    
    def cargar_archivos_json(self, rutas_archivos: List[str]) -> List[Dict]:
        """Carga múltiples archivos JSON."""
        archivos_datos = []
        for ruta in rutas_archivos:
            if os.path.exists(ruta):
                with open(ruta, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    archivos_datos.append(data)
                print(f"✓ Archivo cargado: {ruta}")
            else:
                print(f"✗ Archivo no encontrado: {ruta}")
        return archivos_datos
    
    def son_conceptos_duplicados(self, concepto1: Dict, concepto2: Dict) -> Tuple[bool, str]:
        """
        Determina si dos conceptos son duplicados usando LLM o análisis básico.
        
        Returns:
            Tupla (son_duplicados, explicacion)
        """
        # Primero verificación rápida sin LLM
        if concepto1.get('canonico') == concepto2.get('canonico'):
            return (True, "Conceptos canónicos idénticos")
        
        if concepto1.get('concept_id') and concepto1.get('concept_id') == concepto2.get('concept_id'):
            return (True, "IDs de concepto idénticos")
        
        # Si tenemos LLM, hacer análisis profundo
        if self.usar_llm and self.llm:
            son_dup, confianza, razon = self.llm.son_conceptos_duplicados_llm(concepto1, concepto2)
            if confianza >= self.umbral_confianza:
                return (son_dup, f"LLM ({confianza:.2f}): {razon}")
        
        # Fallback a comparación simple
        concepto1_lower = concepto1.get('concepto', '').lower()
        concepto2_lower = concepto2.get('concepto', '').lower()
        
        if concepto1_lower == concepto2_lower:
            return (True, "Nombres de concepto idénticos")
        
        return (False, "No se detectaron duplicados")
    
    def fusionar_sinonimos_con_llm(self, concepto1: Dict, concepto2: Dict) -> Dict:
        """
        Fusiona dos conceptos usando análisis LLM para sinónimos.
        """
        concepto_fusionado = concepto1.copy()
        
        # Obtener listas de sinónimos simples
        sinonimos1 = concepto1.get('sinonimos', [])
        sinonimos2 = concepto2.get('sinonimos', [])
        
        # Análisis con LLM si está disponible
        if self.usar_llm and self.llm:
            analisis = self.llm.analizar_sinonimos_duplicados(sinonimos1, sinonimos2)
            sinonimos_unificados = analisis.get('sinonimos_unificados', 
                                               list(set(sinonimos1 + sinonimos2)))
            
            # Log de decisiones del LLM
            if analisis.get('variantes'):
                print(f"    LLM detectó variantes: {len(analisis['variantes'])}")
            if analisis.get('duplicados_exactos'):
                print(f"    LLM detectó duplicados exactos: {len(analisis['duplicados_exactos'])}")
        else:
            # Sin LLM, unión simple
            sinonimos_unificados = list(set(sinonimos1 + sinonimos2))
        
        concepto_fusionado['sinonimos'] = sorted(sinonimos_unificados)
        
        # Fusionar sinónimos etiquetados
        self.fusionar_sinonimos_etiquetados(concepto_fusionado, concepto1, concepto2)
        
        # Fusionar notas
        notas1 = concepto1.get('notas', '')
        notas2 = concepto2.get('notas', '')
        if notas1 and notas2 and notas1 != notas2:
            concepto_fusionado['notas'] = f"{notas1} | FUSIONADO CON: {notas2}"
        elif notas2 and not notas1:
            concepto_fusionado['notas'] = notas2
        
        # Actualizar métricas
        self.actualizar_metricas_mx(concepto_fusionado)
        
        # Mantener las mejores validaciones
        concepto_fusionado['validado_cie10'] = (concepto1.get('validado_cie10', False) or 
                                               concepto2.get('validado_cie10', False))
        concepto_fusionado['validado_cemece'] = (concepto1.get('validado_cemece', False) or 
                                                concepto2.get('validado_cemece', False))
        
        # Agregar metadata de fusión
        concepto_fusionado['fusion_info'] = {
            'conceptos_fusionados': [
                concepto1.get('concepto'),
                concepto2.get('concepto')
            ],
            'metodo': 'LLM' if self.usar_llm else 'basico',
            'fecha_fusion': datetime.now().isoformat()
        }
        
        return concepto_fusionado
    
    def fusionar_sinonimos_etiquetados(self, concepto_fusionado: Dict, 
                                      concepto1: Dict, concepto2: Dict) -> None:
        """Fusiona los sinónimos etiquetados de dos conceptos."""
        sinonimos_etiq1 = concepto1.get('sinonimos_etiquetados', [])
        sinonimos_etiq2 = concepto2.get('sinonimos_etiquetados', [])
        
        sinonimos_map = {}
        
        # Procesar primera lista
        for sin in sinonimos_etiq1:
            texto_lower = sin['texto'].lower()
            sinonimos_map[texto_lower] = sin.copy()
        
        # Fusionar segunda lista
        for sin in sinonimos_etiq2:
            texto_lower = sin['texto'].lower()
            if texto_lower in sinonimos_map:
                # Fusionar información
                existente = sinonimos_map[texto_lower]
                
                # Mantener mejor score_mx
                if sin.get('score_mx', 0) > existente.get('score_mx', 0):
                    existente['score_mx'] = sin['score_mx']
                
                # Fusionar etiquetas
                etiquetas = set(existente.get('etiquetas', []))
                etiquetas.update(sin.get('etiquetas', []))
                existente['etiquetas'] = sorted(list(etiquetas))
                
                # Fusionar notas
                if sin.get('notas') and sin['notas'] != existente.get('notas'):
                    nota_existente = existente.get('notas', '')
                    if nota_existente:
                        existente['notas'] = f"{nota_existente}; {sin['notas']}"
                    else:
                        existente['notas'] = sin['notas']
            else:
                sinonimos_map[texto_lower] = sin.copy()
        
        concepto_fusionado['sinonimos_etiquetados'] = list(sinonimos_map.values())
    
    def actualizar_metricas_mx(self, concepto: Dict) -> None:
        """Actualiza las métricas de mexicanidad de un concepto."""
        sinonimos_etiq = concepto.get('sinonimos_etiquetados', [])
        if not sinonimos_etiq:
            return
        
        scores = [s.get('score_mx', 0) for s in sinonimos_etiq]
        total = len(scores)
        
        if total > 0:
            promedio = sum(scores) / total
            max_score = max(scores) if scores else 0
            count_alto = sum(1 for s in scores if s >= 0.15)
            
            concepto['metricas_mx'] = {
                'promedio_score': promedio,
                'max_score': max_score,
                'count_alto': count_alto,
                'total_sinonimos': total
            }
    
    def unificar_archivos(self, archivos_datos: List[Dict]) -> Dict:
        """Unifica múltiples archivos JSON en uno solo."""
        todos_conceptos = []
        
        # Recopilar todos los conceptos
        for archivo in archivos_datos:
            conceptos = archivo.get('conceptos', [])
            todos_conceptos.extend(conceptos)
        
        print(f"\nTotal de conceptos encontrados: {len(todos_conceptos)}")
        
        # Agrupar conceptos duplicados
        print("\nAnalizando duplicados...")
        grupos_duplicados = []
        conceptos_procesados = set()
        
        for i, concepto1 in enumerate(todos_conceptos):
            if i in conceptos_procesados:
                continue
            
            grupo_actual = [concepto1]
            conceptos_procesados.add(i)
            
            for j, concepto2 in enumerate(todos_conceptos[i+1:], i+1):
                if j in conceptos_procesados:
                    continue
                
                son_dup, explicacion = self.son_conceptos_duplicados(concepto1, concepto2)
                
                if son_dup:
                    grupo_actual.append(concepto2)
                    conceptos_procesados.add(j)
                    print(f"  → Duplicado: '{concepto1['concepto']}' ≈ '{concepto2['concepto']}'")
                    print(f"    Razón: {explicacion}")
            
            grupos_duplicados.append(grupo_actual)
        
        # Fusionar grupos
        print("\nFusionando conceptos duplicados...")
        for grupo in grupos_duplicados:
            if len(grupo) > 1:
                print(f"  Fusionando grupo de {len(grupo)} conceptos: {grupo[0]['concepto']}")
                concepto_fusionado = grupo[0]
                for concepto in grupo[1:]:
                    concepto_fusionado = self.fusionar_sinonimos_con_llm(
                        concepto_fusionado, concepto
                    )
                self.conceptos_unificados.append(concepto_fusionado)
            else:
                self.conceptos_unificados.append(grupo[0])
        
        print(f"\nConceptos después de unificación: {len(self.conceptos_unificados)}")
        print(f"Conceptos fusionados: {len(todos_conceptos) - len(self.conceptos_unificados)}")
        
        # Actualizar metadata
        self.actualizar_metadata_global(archivos_datos)
        
        return {
            'metadata': self.metadata_global,
            'conceptos': self.conceptos_unificados
        }
    
    def actualizar_metadata_global(self, archivos_datos: List[Dict]) -> None:
        """Actualiza los metadatos globales."""
        if archivos_datos:
            self.metadata_global = archivos_datos[0].get('metadata', {}).copy()
        else:
            self.metadata_global = {}
        
        # Calcular estadísticas
        validados_cie10_exactos = sum(
            1 for c in self.conceptos_unificados 
            if c.get('cie10_match_calidad') == 'exacto'
        )
        validados_cie10_parciales = sum(
            1 for c in self.conceptos_unificados 
            if c.get('cie10_match_calidad') == 'parcial'
        )
        validados_cemece = sum(
            1 for c in self.conceptos_unificados 
            if c.get('validado_cemece', False)
        )
        validados_ambas = sum(
            1 for c in self.conceptos_unificados 
            if c.get('validado_cie10', False) and c.get('validado_cemece', False)
        )
        
        # Score mexicanidad
        scores_mx = []
        terminos_bien_mexicanizados = 0
        conceptos_fusionados = 0
        
        for concepto in self.conceptos_unificados:
            if 'metricas_mx' in concepto:
                score_promedio = concepto['metricas_mx'].get('promedio_score', 0)
                scores_mx.append(score_promedio)
                if score_promedio >= 0.15:
                    terminos_bien_mexicanizados += 1
            
            if 'fusion_info' in concepto:
                conceptos_fusionados += 1
        
        score_mx_promedio = sum(scores_mx) / len(scores_mx) if scores_mx else 0
        
        # Tipos encontrados
        tipos = set()
        for concepto in self.conceptos_unificados:
            if 'tipo' in concepto:
                tipos.add(concepto['tipo'])
        
        # Actualizar metadata
        self.metadata_global.update({
            'fecha_generacion': datetime.now().isoformat(),
            'total_conceptos': len(self.conceptos_unificados),
            'validados_cie10_exactos': validados_cie10_exactos,
            'validados_cie10_parciales': validados_cie10_parciales,
            'validados_cemece': validados_cemece,
            'validados_ambas_fuentes': validados_ambas,
            'score_mexicanidad_promedio': f"{score_mx_promedio:.3f}",
            'terminos_bien_mexicanizados': terminos_bien_mexicanizados,
            'tipos_encontrados': sorted(list(tipos)),
            'archivos_fusionados': len(archivos_datos),
            'conceptos_con_fusion': conceptos_fusionados,
            'metodo_unificacion': 'LLM (GPT-OSS:20B)' if self.usar_llm else 'Análisis básico',
            'version': '4.0.0-unificado-llm',
            'notas_unificacion': f'Unificado de {len(archivos_datos)} archivos'
        })
    
    def guardar_resultado(self, resultado: Dict, ruta_salida: str) -> None:
        """Guarda el resultado unificado."""
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Archivo unificado guardado en: {ruta_salida}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Unifica JSONs médicos con análisis LLM'
    )
    parser.add_argument(
        'archivos', 
        nargs='+', 
        help='Archivos JSON a unificar'
    )
    parser.add_argument(
        '-o', '--output',
        default='unificado_llm.json',
        help='Archivo de salida (default: unificado_llm.json)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='No usar LLM, solo análisis básico'
    )
    parser.add_argument(
        '--umbral-confianza',
        type=float,
        default=0.7,
        help='Umbral de confianza LLM (0-1, default: 0.7)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNIFICADOR DE JSONs MÉDICOS CON LLM")
    print("=" * 60)
    
    # Crear unificador
    usar_llm = not args.no_llm
    unificador = ConceptoUnificadorLLM(
        usar_llm=usar_llm,
        umbral_confianza=args.umbral_confianza
    )
    
    # Cargar archivos
    print("\n1. CARGANDO ARCHIVOS...")
    archivos_datos = unificador.cargar_archivos_json(args.archivos)
    
    if not archivos_datos:
        print("✗ No se pudieron cargar archivos.")
        return
    
    # Unificar
    print("\n2. UNIFICANDO CONCEPTOS...")
    resultado = unificador.unificar_archivos(archivos_datos)
    
    # Guardar
    print("\n3. GUARDANDO RESULTADO...")
    unificador.guardar_resultado(resultado, args.output)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE UNIFICACIÓN")
    print("=" * 60)
    print(f"Método: {'LLM (GPT-OSS:20B)' if usar_llm else 'Análisis básico'}")
    print(f"Archivos procesados: {len(archivos_datos)}")
    print(f"Conceptos originales: {sum(len(a.get('conceptos', [])) for a in archivos_datos)}")
    print(f"Conceptos finales: {resultado['metadata']['total_conceptos']}")
    print(f"Conceptos fusionados: {resultado['metadata'].get('conceptos_con_fusion', 0)}")
    print(f"Score mexicanidad promedio: {resultado['metadata']['score_mexicanidad_promedio']}")
    print("=" * 60)


if __name__ == "__main__":
    main()