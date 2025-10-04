#!/usr/bin/env python3
"""
verificar_resultado.py

Script para verificar la correcta unificación de conceptos médicos.
Detecta duplicados potenciales, valida integridad y genera reporte detallado.

Uso:
  python verificar_resultado.py resultado.json
  python verificar_resultado.py resultado.json --jaccard 0.5 --verbose
  python verificar_resultado.py resultado.json --export-report reporte.txt
"""

import json
import re
import sys
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
import argparse
from collections import defaultdict, Counter
from datetime import datetime

# Importaciones opcionales
try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False
    def unidecode(s: str) -> str:
        return s

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc
        def __iter__(self):
            if self.desc:
                print(f"{self.desc}...")
            return iter(self.iterable)

# =============================================================================
# FUNCIONES DE NORMALIZACIÓN
# =============================================================================

def normalize_text(s: str) -> str:
    """Normaliza texto para comparación."""
    if not isinstance(s, str):
        s = str(s or "")
    
    result = s.strip().lower()
    if UNIDECODE_AVAILABLE:
        result = unidecode(result)
    result = re.sub(r"[^\w\s]+", " ", result, flags=re.UNICODE)
    result = re.sub(r"\s+", " ", result, flags=re.UNICODE).strip()
    
    return result

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calcula similitud de Jaccard entre dos conjuntos."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0

# =============================================================================
# CLASE VERIFICADOR
# =============================================================================

class VerificadorResultado:
    """Verificador completo de resultados de unificación."""
    
    def __init__(self, jaccard_threshold: float = 0.5, verbose: bool = False):
        self.jaccard_threshold = jaccard_threshold
        self.verbose = verbose
        self.problemas = []
        self.advertencias = []
        self.estadisticas = {}
        
    def log(self, msg: str, nivel: str = "INFO"):
        """Log con nivel."""
        if self.verbose or nivel in ["ERROR", "WARNING"]:
            print(f"[{nivel}] {msg}")
    
    def cargar_archivo(self, ruta: str) -> Optional[Dict]:
        """Carga archivo JSON."""
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            self.log(f"Error al cargar archivo: {e}", "ERROR")
            return None
    
    def verificar_estructura(self, data: Dict) -> bool:
        """Verifica la estructura básica del JSON."""
        self.log("Verificando estructura del archivo...")
        
        if not isinstance(data, dict):
            self.problemas.append("El archivo no es un diccionario JSON válido")
            return False
        
        if "conceptos" not in data:
            self.problemas.append("Falta campo 'conceptos' en el archivo")
            return False
        
        if "metadata" not in data:
            self.advertencias.append("Falta campo 'metadata' en el archivo")
        
        conceptos = data.get("conceptos", [])
        if not isinstance(conceptos, list):
            self.problemas.append("El campo 'conceptos' no es una lista")
            return False
        
        # Verificar campos básicos en cada concepto
        campos_requeridos = {"concepto", "canonico"}
        campos_opcionales = {"concept_id", "tipo", "sinonimos", "sinonimos_etiquetados", 
                            "validado_cie10", "validado_cemece", "notas"}
        
        for i, concepto in enumerate(conceptos[:10]):  # Verificar primeros 10
            if not isinstance(concepto, dict):
                self.problemas.append(f"Concepto {i} no es un diccionario")
                continue
            
            faltantes = campos_requeridos - set(concepto.keys())
            if faltantes:
                self.advertencias.append(f"Concepto {i} le faltan campos: {faltantes}")
        
        self.log(f"Estructura verificada: {len(conceptos)} conceptos")
        return True
    
    def detectar_duplicados_por_id(self, conceptos: List[Dict]) -> List[Tuple[str, List[int]]]:
        """Detecta conceptos con mismo concept_id."""
        self.log("Verificando duplicados por concept_id...")
        
        id_map = defaultdict(list)
        
        for i, concepto in enumerate(conceptos):
            concept_id = concepto.get("concept_id", "").strip()
            if concept_id:
                id_map[concept_id].append(i)
        
        duplicados = [(cid, indices) for cid, indices in id_map.items() if len(indices) > 1]
        
        if duplicados:
            self.log(f"¡Encontrados {len(duplicados)} IDs duplicados!", "WARNING")
            for concept_id, indices in duplicados[:5]:  # Mostrar primeros 5
                conceptos_dup = [conceptos[i].get("concepto", "") for i in indices]
                self.problemas.append(
                    f"concept_id '{concept_id}' duplicado en {len(indices)} conceptos: {conceptos_dup}"
                )
        
        return duplicados
    
    def detectar_duplicados_por_canonico(self, conceptos: List[Dict]) -> List[Tuple[str, List[int]]]:
        """Detecta conceptos con mismo canónico normalizado."""
        self.log("Verificando duplicados por término canónico...")
        
        canon_map = defaultdict(list)
        
        for i, concepto in enumerate(conceptos):
            canonico = normalize_text(concepto.get("canonico", ""))
            if canonico:
                canon_map[canonico].append(i)
        
        duplicados = [(canon, indices) for canon, indices in canon_map.items() if len(indices) > 1]
        
        if duplicados:
            self.log(f"¡Encontrados {len(duplicados)} canónicos duplicados!", "WARNING")
            for canonico_norm, indices in duplicados[:5]:  # Mostrar primeros 5
                conceptos_dup = [conceptos[i].get("concepto", "") for i in indices]
                self.problemas.append(
                    f"Canónico normalizado '{canonico_norm}' duplicado en {len(indices)} conceptos: {conceptos_dup}"
                )
        
        return duplicados
    
    def detectar_alta_similitud(self, conceptos: List[Dict], muestra: int = 100) -> List[Tuple[int, int, float]]:
        """
        Detecta pares de conceptos con alta similitud de Jaccard.
        Para eficiencia, solo verifica una muestra aleatoria.
        """
        self.log(f"Verificando alta similitud (muestra de {muestra} conceptos)...")
        
        import random
        
        # Tomar muestra aleatoria si hay muchos conceptos
        if len(conceptos) > muestra * 2:
            indices_muestra = random.sample(range(len(conceptos)), muestra)
        else:
            indices_muestra = range(len(conceptos))
        
        alta_similitud = []
        
        # Preparar términos normalizados
        conceptos_norm = []
        for i in indices_muestra:
            c = conceptos[i]
            canonico = normalize_text(c.get("canonico", ""))
            concepto = normalize_text(c.get("concepto", ""))
            sinonimos = c.get("sinonimos", [])
            sin_norm = {normalize_text(s) for s in sinonimos if s}
            todos_terminos = {canonico, concepto} | sin_norm
            conceptos_norm.append((i, todos_terminos))
        
        # Comparar pares
        for idx, (i, terminos_i) in enumerate(conceptos_norm):
            for j, terminos_j in conceptos_norm[idx+1:]:
                if not terminos_i or not terminos_j:
                    continue
                
                similitud = jaccard_similarity(terminos_i, terminos_j)
                
                if similitud >= self.jaccard_threshold:
                    alta_similitud.append((i, j, similitud))
        
        if alta_similitud:
            self.log(f"¡Encontrados {len(alta_similitud)} pares con alta similitud!", "WARNING")
            # Ordenar por similitud descendente
            alta_similitud.sort(key=lambda x: x[2], reverse=True)
            
            for i, j, sim in alta_similitud[:5]:  # Mostrar top 5
                concepto_i = conceptos[i].get("concepto", "")
                concepto_j = conceptos[j].get("concepto", "")
                self.problemas.append(
                    f"Alta similitud ({sim:.3f}) entre '{concepto_i}' y '{concepto_j}'"
                )
        
        return alta_similitud
    
    def detectar_canonico_en_sinonimos(self, conceptos: List[Dict]) -> List[Tuple[int, int, str]]:
        """Detecta casos donde el canónico de un concepto aparece en sinónimos de otro."""
        self.log("Verificando canónicos en sinónimos de otros conceptos...")
        
        cruces = []
        
        # Crear índice de sinónimos
        sinonimos_idx = {}
        for i, concepto in enumerate(conceptos):
            sinonimos = concepto.get("sinonimos", [])
            for sin in sinonimos:
                sin_norm = normalize_text(sin)
                if sin_norm:
                    if sin_norm not in sinonimos_idx:
                        sinonimos_idx[sin_norm] = []
                    sinonimos_idx[sin_norm].append(i)
        
        # Buscar canónicos en sinónimos
        for i, concepto in enumerate(conceptos):
            canonico = normalize_text(concepto.get("canonico", ""))
            if canonico and canonico in sinonimos_idx:
                for j in sinonimos_idx[canonico]:
                    if i != j:  # No el mismo concepto
                        cruces.append((i, j, canonico))
        
        if cruces:
            self.log(f"¡Encontrados {len(cruces)} cruces canónico-sinónimo!", "WARNING")
            for i, j, termino in cruces[:5]:  # Mostrar primeros 5
                concepto_i = conceptos[i].get("concepto", "")
                concepto_j = conceptos[j].get("concepto", "")
                self.problemas.append(
                    f"Canónico '{termino}' de '{concepto_i}' aparece en sinónimos de '{concepto_j}'"
                )
        
        return cruces
    
    def verificar_integridad_datos(self, conceptos: List[Dict]) -> Dict:
        """Verifica la integridad general de los datos."""
        self.log("Verificando integridad de datos...")
        
        stats = {
            "total": len(conceptos),
            "con_concept_id": 0,
            "con_tipo": 0,
            "con_sinonimos": 0,
            "con_sinonimos_etiquetados": 0,
            "validado_cie10": 0,
            "validado_cemece": 0,
            "con_metricas_mx": 0,
            "tipos_unicos": set(),
            "sin_canonico": 0,
            "sin_concepto": 0
        }
        
        for concepto in conceptos:
            if concepto.get("concept_id"):
                stats["con_concept_id"] += 1
            if concepto.get("tipo"):
                stats["con_tipo"] += 1
                stats["tipos_unicos"].add(concepto["tipo"])
            if concepto.get("sinonimos"):
                stats["con_sinonimos"] += 1
            if concepto.get("sinonimos_etiquetados"):
                stats["con_sinonimos_etiquetados"] += 1
            if concepto.get("validado_cie10"):
                stats["validado_cie10"] += 1
            if concepto.get("validado_cemece"):
                stats["validado_cemece"] += 1
            if concepto.get("metricas_mx"):
                stats["con_metricas_mx"] += 1
            if not concepto.get("canonico"):
                stats["sin_canonico"] += 1
            if not concepto.get("concepto"):
                stats["sin_concepto"] += 1
        
        stats["tipos_unicos"] = sorted(stats["tipos_unicos"])
        
        # Advertencias por datos faltantes
        if stats["sin_canonico"] > 0:
            self.advertencias.append(f"{stats['sin_canonico']} conceptos sin término canónico")
        if stats["sin_concepto"] > 0:
            self.advertencias.append(f"{stats['sin_concepto']} conceptos sin campo 'concepto'")
        
        return stats
    
    def verificar_metadata(self, data: Dict) -> Dict:
        """Verifica consistencia de metadata con los datos."""
        self.log("Verificando metadata...")
        
        metadata = data.get("metadata", {})
        conceptos = data.get("conceptos", [])
        
        verificaciones = {}
        
        # Verificar total_conceptos
        total_real = len(conceptos)
        total_metadata = metadata.get("total_conceptos", 0)
        
        if total_real != total_metadata:
            self.problemas.append(
                f"Inconsistencia en total_conceptos: metadata={total_metadata}, real={total_real}"
            )
        verificaciones["total_conceptos_correcto"] = (total_real == total_metadata)
        
        # Verificar validados
        val_cie10_real = sum(1 for c in conceptos if c.get("validado_cie10"))
        val_cemece_real = sum(1 for c in conceptos if c.get("validado_cemece"))
        
        val_cie10_meta = metadata.get("validados_cie10_exactos", 0) + metadata.get("validados_cie10_parciales", 0)
        val_cemece_meta = metadata.get("validados_cemece", 0)
        
        if abs(val_cie10_real - val_cie10_meta) > 1:  # Tolerancia de 1
            self.advertencias.append(
                f"Posible inconsistencia en validados CIE10: metadata={val_cie10_meta}, real={val_cie10_real}"
            )
        
        if val_cemece_real != val_cemece_meta:
            self.advertencias.append(
                f"Posible inconsistencia en validados CEMECE: metadata={val_cemece_meta}, real={val_cemece_real}"
            )
        
        return verificaciones
    
    def generar_reporte(self) -> str:
        """Genera reporte detallado de verificación."""
        reporte = []
        reporte.append("=" * 70)
        reporte.append("REPORTE DE VERIFICACIÓN DE UNIFICACIÓN")
        reporte.append("=" * 70)
        reporte.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        reporte.append("")
        
        # Resumen
        reporte.append("RESUMEN")
        reporte.append("-" * 70)
        
        if not self.problemas:
            reporte.append("✓ No se encontraron problemas críticos")
        else:
            reporte.append(f"✗ Se encontraron {len(self.problemas)} problemas")
        
        if self.advertencias:
            reporte.append(f"⚠ Se encontraron {len(self.advertencias)} advertencias")
        
        reporte.append("")
        
        # Estadísticas
        if self.estadisticas:
            reporte.append("ESTADÍSTICAS")
            reporte.append("-" * 70)
            for key, value in self.estadisticas.items():
                if isinstance(value, list) and len(value) > 10:
                    reporte.append(f"  {key}: {len(value)} elementos")
                else:
                    reporte.append(f"  {key}: {value}")
            reporte.append("")
        
        # Problemas
        if self.problemas:
            reporte.append("PROBLEMAS ENCONTRADOS")
            reporte.append("-" * 70)
            for i, problema in enumerate(self.problemas, 1):
                reporte.append(f"{i}. {problema}")
            reporte.append("")
        
        # Advertencias
        if self.advertencias:
            reporte.append("ADVERTENCIAS")
            reporte.append("-" * 70)
            for i, advertencia in enumerate(self.advertencias, 1):
                reporte.append(f"{i}. {advertencia}")
            reporte.append("")
        
        # Recomendaciones
        reporte.append("RECOMENDACIONES")
        reporte.append("-" * 70)
        
        if self.problemas:
            reporte.append("• Revisar y corregir los duplicados encontrados")
            reporte.append("• Ejecutar nuevamente el proceso de unificación")
            reporte.append("• Considerar ajustar el umbral de similitud si hay muchos falsos positivos")
        else:
            reporte.append("• El archivo parece estar correctamente unificado")
            reporte.append("• Se recomienda hacer verificaciones periódicas")
        
        reporte.append("")
        reporte.append("=" * 70)
        
        return "\n".join(reporte)
    
    def verificar(self, ruta_archivo: str) -> bool:
        """Ejecuta verificación completa."""
        print("\n" + "=" * 70)
        print("INICIANDO VERIFICACIÓN DE RESULTADO")
        print("=" * 70)
        
        # Cargar archivo
        data = self.cargar_archivo(ruta_archivo)
        if not data:
            return False
        
        # Verificar estructura
        if not self.verificar_estructura(data):
            self.log("Estructura inválida, abortando verificación", "ERROR")
            return False
        
        conceptos = data.get("conceptos", [])
        
        print(f"\nVerificando {len(conceptos)} conceptos...")
        print("-" * 70)
        
        # Ejecutar verificaciones
        duplicados_id = self.detectar_duplicados_por_id(conceptos)
        duplicados_canon = self.detectar_duplicados_por_canonico(conceptos)
        alta_similitud = self.detectar_alta_similitud(conceptos)
        cruces = self.detectar_canonico_en_sinonimos(conceptos)
        stats = self.verificar_integridad_datos(conceptos)
        meta_check = self.verificar_metadata(data)
        
        # Guardar estadísticas
        self.estadisticas = {
            "total_conceptos": len(conceptos),
            "duplicados_por_id": len(duplicados_id),
            "duplicados_por_canonico": len(duplicados_canon),
            "pares_alta_similitud": len(alta_similitud),
            "cruces_canonico_sinonimo": len(cruces),
            **stats
        }
        
        # Determinar resultado
        resultado = len(self.problemas) == 0
        
        return resultado


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Verificador de resultados de unificación de conceptos médicos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s resultado.json
  %(prog)s resultado.json --jaccard 0.6
  %(prog)s resultado.json --verbose
  %(prog)s resultado.json --export-report reporte.txt
  %(prog)s resultado.json --quick
        """
    )
    
    parser.add_argument(
        'archivo',
        help='Archivo JSON a verificar'
    )
    parser.add_argument(
        '--jaccard',
        type=float,
        default=0.5,
        help='Umbral de Jaccard para detectar similitud (default: 0.5)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar información detallada durante la verificación'
    )
    parser.add_argument(
        '--export-report',
        help='Exportar reporte a archivo de texto'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Verificación rápida (omite algunas validaciones pesadas)'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Desactivar colores en la salida'
    )
    
    args = parser.parse_args()
    
    # Verificar que el archivo existe
    if not Path(args.archivo).exists():
        print(f"Error: El archivo '{args.archivo}' no existe")
        return 1
    
    # Crear verificador
    verificador = VerificadorResultado(
        jaccard_threshold=args.jaccard,
        verbose=args.verbose
    )
    
    # Ejecutar verificación
    resultado = verificador.verificar(args.archivo)
    
    # Generar reporte
    reporte = verificador.generar_reporte()
    print(reporte)
    
    # Exportar si se solicitó
    if args.export_report:
        try:
            with open(args.export_report, 'w', encoding='utf-8') as f:
                f.write(reporte)
            print(f"\nReporte exportado a: {args.export_report}")
        except Exception as e:
            print(f"Error al exportar reporte: {e}")
    
    # Resultado final con colores (si están habilitados)
    if not args.no_color and sys.platform != 'win32':
        VERDE = '\033[92m'
        ROJO = '\033[91m'
        AMARILLO = '\033[93m'
        RESET = '\033[0m'
    else:
        VERDE = ROJO = AMARILLO = RESET = ''
    
    print("\n" + "=" * 70)
    if resultado:
        print(f"{VERDE}✓ VERIFICACIÓN EXITOSA{RESET}")
        print("El archivo está correctamente unificado")
    else:
        print(f"{ROJO}✗ VERIFICACIÓN FALLIDA{RESET}")
        print(f"Se encontraron {len(verificador.problemas)} problemas que requieren atención")
    
    if verificador.advertencias:
        print(f"{AMARILLO}⚠ Hay {len(verificador.advertencias)} advertencias a considerar{RESET}")
    
    print("=" * 70)
    
    # Código de salida
    return 0 if resultado else 1


if __name__ == "__main__":
    sys.exit(main())