#!/usr/bin/env python3
"""
Unificador de JSONs médicos con detección de duplicados y fusión de sinónimos.
Versión sin LLM - usa similitud de texto y reglas para detectar duplicados.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any
from difflib import SequenceMatcher
from collections import defaultdict
import argparse


class ConceptoUnificador:
    """Clase para unificar conceptos médicos de múltiples JSONs."""
    
    def __init__(self, umbral_similitud: float = 0.85):
        """
        Inicializa el unificador.
        
        Args:
            umbral_similitud: Umbral para considerar dos conceptos como duplicados (0-1)
        """
        self.umbral_similitud = umbral_similitud
        self.conceptos_unificados = []
        self.metadata_global = None
        
    def cargar_archivos_json(self, rutas_archivos: List[str]) -> List[Dict]:
        """
        Carga múltiples archivos JSON.
        
        Args:
            rutas_archivos: Lista de rutas a archivos JSON
            
        Returns:
            Lista de diccionarios con los datos de cada archivo
        """
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
    
    def calcular_similitud(self, texto1: str, texto2: str) -> float:
        """
        Calcula la similitud entre dos textos.
        
        Args:
            texto1: Primer texto
            texto2: Segundo texto
            
        Returns:
            Valor de similitud entre 0 y 1
        """
        texto1_lower = texto1.lower().strip()
        texto2_lower = texto2.lower().strip()
        return SequenceMatcher(None, texto1_lower, texto2_lower).ratio()
    
    def son_conceptos_duplicados(self, concepto1: Dict, concepto2: Dict) -> bool:
        """
        Determina si dos conceptos son duplicados.
        
        Args:
            concepto1: Primer concepto
            concepto2: Segundo concepto
            
        Returns:
            True si son duplicados, False en caso contrario
        """
        # Verificar por concepto canónico
        if concepto1.get('canonico') == concepto2.get('canonico'):
            return True
        
        # Verificar por similitud de concepto
        sim_concepto = self.calcular_similitud(
            concepto1.get('concepto', ''),
            concepto2.get('concepto', '')
        )
        if sim_concepto >= self.umbral_similitud:
            return True
        
        # Verificar por concept_id (si existe y es igual)
        if (concepto1.get('concept_id') and 
            concepto2.get('concept_id') and 
            concepto1['concept_id'] == concepto2['concept_id']):
            return True
        
        # Verificar por término original
        if (concepto1.get('termino_original') and 
            concepto2.get('termino_original')):
            sim_original = self.calcular_similitud(
                concepto1['termino_original'],
                concepto2['termino_original']
            )
            if sim_original >= self.umbral_similitud:
                return True
        
        return False
    
    def fusionar_sinonimos_etiquetados(self, sinonimos1: List[Dict], 
                                      sinonimos2: List[Dict]) -> List[Dict]:
        """
        Fusiona listas de sinónimos etiquetados, evitando duplicados.
        
        Args:
            sinonimos1: Primera lista de sinónimos etiquetados
            sinonimos2: Segunda lista de sinónimos etiquetados
            
        Returns:
            Lista fusionada de sinónimos únicos
        """
        sinonimos_unificados = sinonimos1.copy()
        textos_existentes = {s['texto'].lower() for s in sinonimos1}
        
        for sinonimo in sinonimos2:
            texto_lower = sinonimo['texto'].lower()
            
            # Si el texto no existe, agregarlo
            if texto_lower not in textos_existentes:
                sinonimos_unificados.append(sinonimo)
                textos_existentes.add(texto_lower)
            else:
                # Si existe, actualizar con mejor score_mx o fusionar etiquetas
                for i, sin_existente in enumerate(sinonimos_unificados):
                    if sin_existente['texto'].lower() == texto_lower:
                        # Mantener el score_mx más alto
                        if sinonimo.get('score_mx', 0) > sin_existente.get('score_mx', 0):
                            sinonimos_unificados[i]['score_mx'] = sinonimo['score_mx']
                        
                        # Fusionar etiquetas únicas
                        etiquetas_existentes = set(sin_existente.get('etiquetas', []))
                        etiquetas_nuevas = set(sinonimo.get('etiquetas', []))
                        sinonimos_unificados[i]['etiquetas'] = list(
                            etiquetas_existentes.union(etiquetas_nuevas)
                        )
                        
                        # Fusionar notas si son diferentes
                        if sinonimo.get('notas') and sinonimo['notas'] != sin_existente.get('notas'):
                            nota_existente = sin_existente.get('notas', '')
                            nota_nueva = sinonimo['notas']
                            if nota_existente and nota_nueva not in nota_existente:
                                sinonimos_unificados[i]['notas'] = f"{nota_existente}; {nota_nueva}"
                            elif not nota_existente:
                                sinonimos_unificados[i]['notas'] = nota_nueva
                        break
        
        return sinonimos_unificados
    
    def fusionar_conceptos(self, concepto1: Dict, concepto2: Dict) -> Dict:
        """
        Fusiona dos conceptos duplicados en uno solo.
        
        Args:
            concepto1: Primer concepto
            concepto2: Segundo concepto
            
        Returns:
            Concepto fusionado
        """
        concepto_fusionado = concepto1.copy()
        
        # Fusionar sinónimos etiquetados
        sinonimos_etiq1 = concepto1.get('sinonimos_etiquetados', [])
        sinonimos_etiq2 = concepto2.get('sinonimos_etiquetados', [])
        concepto_fusionado['sinonimos_etiquetados'] = self.fusionar_sinonimos_etiquetados(
            sinonimos_etiq1, sinonimos_etiq2
        )
        
        # Actualizar lista de sinónimos simples
        sinonimos_set = set(concepto1.get('sinonimos', []))
        sinonimos_set.update(concepto2.get('sinonimos', []))
        concepto_fusionado['sinonimos'] = sorted(list(sinonimos_set))
        
        # Fusionar notas
        notas1 = concepto1.get('notas', '')
        notas2 = concepto2.get('notas', '')
        if notas1 and notas2 and notas1 != notas2:
            concepto_fusionado['notas'] = f"{notas1} | {notas2}"
        elif notas2 and not notas1:
            concepto_fusionado['notas'] = notas2
        
        # Actualizar métricas mexicanas
        self.actualizar_metricas_mx(concepto_fusionado)
        
        # Mantener las mejores validaciones
        if concepto2.get('validado_cie10', False):
            concepto_fusionado['validado_cie10'] = True
        if concepto2.get('validado_cemece', False):
            concepto_fusionado['validado_cemece'] = True
        
        return concepto_fusionado
    
    def actualizar_metricas_mx(self, concepto: Dict) -> None:
        """
        Actualiza las métricas de mexicanidad de un concepto.
        
        Args:
            concepto: Concepto a actualizar
        """
        sinonimos_etiq = concepto.get('sinonimos_etiquetados', [])
        if not sinonimos_etiq:
            return
        
        scores = [s.get('score_mx', 0) for s in sinonimos_etiq]
        total = len(scores)
        
        if total > 0:
            promedio = sum(scores) / total
            max_score = max(scores)
            count_alto = sum(1 for s in scores if s >= 0.15)
            
            concepto['metricas_mx'] = {
                'promedio_score': promedio,
                'max_score': max_score,
                'count_alto': count_alto,
                'total_sinonimos': total
            }
    
    def unificar_archivos(self, archivos_datos: List[Dict]) -> Dict:
        """
        Unifica múltiples archivos JSON en uno solo.
        
        Args:
            archivos_datos: Lista de diccionarios con datos de archivos JSON
            
        Returns:
            Diccionario unificado
        """
        todos_conceptos = []
        
        # Recopilar todos los conceptos
        for archivo in archivos_datos:
            conceptos = archivo.get('conceptos', [])
            todos_conceptos.extend(conceptos)
        
        print(f"\nTotal de conceptos encontrados: {len(todos_conceptos)}")
        
        # Agrupar conceptos duplicados
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
                    
                if self.son_conceptos_duplicados(concepto1, concepto2):
                    grupo_actual.append(concepto2)
                    conceptos_procesados.add(j)
                    print(f"  → Duplicado detectado: '{concepto1['concepto']}' ≈ '{concepto2['concepto']}'")
            
            grupos_duplicados.append(grupo_actual)
        
        # Fusionar grupos de duplicados
        for grupo in grupos_duplicados:
            if len(grupo) > 1:
                concepto_fusionado = grupo[0]
                for concepto in grupo[1:]:
                    concepto_fusionado = self.fusionar_conceptos(concepto_fusionado, concepto)
                self.conceptos_unificados.append(concepto_fusionado)
            else:
                self.conceptos_unificados.append(grupo[0])
        
        print(f"Conceptos después de unificación: {len(self.conceptos_unificados)}")
        print(f"Conceptos duplicados fusionados: {len(todos_conceptos) - len(self.conceptos_unificados)}")
        
        # Actualizar metadata
        self.actualizar_metadata_global(archivos_datos)
        
        return {
            'metadata': self.metadata_global,
            'conceptos': self.conceptos_unificados
        }
    
    def actualizar_metadata_global(self, archivos_datos: List[Dict]) -> None:
        """
        Actualiza los metadatos globales basándose en los conceptos unificados.
        
        Args:
            archivos_datos: Lista de archivos originales
        """
        # Tomar metadata base del primer archivo
        if archivos_datos:
            self.metadata_global = archivos_datos[0].get('metadata', {}).copy()
        else:
            self.metadata_global = {}
        
        # Actualizar con nuevos valores
        self.metadata_global['fecha_generacion'] = datetime.now().isoformat()
        self.metadata_global['total_conceptos'] = len(self.conceptos_unificados)
        
        # Recalcular estadísticas
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
        
        # Calcular score mexicanidad promedio
        scores_mx = []
        terminos_bien_mexicanizados = 0
        for concepto in self.conceptos_unificados:
            if 'metricas_mx' in concepto:
                score_promedio = concepto['metricas_mx'].get('promedio_score', 0)
                scores_mx.append(score_promedio)
                if score_promedio >= 0.15:
                    terminos_bien_mexicanizados += 1
        
        score_mx_promedio = sum(scores_mx) / len(scores_mx) if scores_mx else 0
        
        # Contar tipos
        tipos = set()
        for concepto in self.conceptos_unificados:
            if 'tipo' in concepto:
                tipos.add(concepto['tipo'])
        
        # Actualizar metadata
        self.metadata_global.update({
            'validados_cie10_exactos': validados_cie10_exactos,
            'validados_cie10_parciales': validados_cie10_parciales,
            'validados_cemece': validados_cemece,
            'validados_ambas_fuentes': validados_ambas,
            'score_mexicanidad_promedio': f"{score_mx_promedio:.3f}",
            'terminos_bien_mexicanizados': terminos_bien_mexicanizados,
            'tipos_encontrados': sorted(list(tipos)),
            'archivos_fusionados': len(archivos_datos),
            'version': '4.0.0-unificado',
            'notas_unificacion': f'Unificado de {len(archivos_datos)} archivos con umbral de similitud {self.umbral_similitud}'
        })
    
    def guardar_resultado(self, resultado: Dict, ruta_salida: str) -> None:
        """
        Guarda el resultado unificado en un archivo JSON.
        
        Args:
            resultado: Diccionario con el resultado unificado
            ruta_salida: Ruta del archivo de salida
        """
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Archivo unificado guardado en: {ruta_salida}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Unifica múltiples archivos JSON médicos fusionando conceptos duplicados'
    )
    parser.add_argument(
        'archivos', 
        nargs='+', 
        help='Archivos JSON a unificar'
    )
    parser.add_argument(
        '-o', '--output',
        default='unificado.json',
        help='Archivo de salida (default: unificado.json)'
    )
    parser.add_argument(
        '-u', '--umbral',
        type=float,
        default=0.85,
        help='Umbral de similitud para detectar duplicados (0-1, default: 0.85)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNIFICADOR DE JSONs MÉDICOS")
    print("=" * 60)
    
    # Crear unificador
    unificador = ConceptoUnificador(umbral_similitud=args.umbral)
    
    # Cargar archivos
    print("\n1. CARGANDO ARCHIVOS...")
    archivos_datos = unificador.cargar_archivos_json(args.archivos)
    
    if not archivos_datos:
        print("✗ No se pudieron cargar archivos. Terminando.")
        return
    
    # Unificar
    print("\n2. UNIFICANDO CONCEPTOS...")
    resultado = unificador.unificar_archivos(archivos_datos)
    
    # Guardar resultado
    print("\n3. GUARDANDO RESULTADO...")
    unificador.guardar_resultado(resultado, args.output)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE UNIFICACIÓN")
    print("=" * 60)
    print(f"Archivos procesados: {len(archivos_datos)}")
    print(f"Conceptos finales: {resultado['metadata']['total_conceptos']}")
    print(f"Score mexicanidad promedio: {resultado['metadata']['score_mexicanidad_promedio']}")
    print(f"Términos bien mexicanizados: {resultado['metadata']['terminos_bien_mexicanizados']}")
    print("=" * 60)


if __name__ == "__main__":
    main()