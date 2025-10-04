#!/usr/bin/env python3
"""
unificar_optimizado_v2.py

Versión mejorada del unificador que corrige problemas de detección de duplicados.
Mejoras principales:
- Detección más agresiva de canónicos duplicados
- Fusión de conceptos con cruces canónico-sinónimo
- Mejor manejo de transitividad en las fusiones

Uso:
  python unificar_optimizado_v2.py *.json -o resultado.json
  python unificar_optimizado_v2.py *.json -o resultado.json --jaccard 0.4 --aggressive
"""

import json
import os
import re
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional, FrozenSet
from pathlib import Path
import argparse
from dataclasses import dataclass, field
from collections import defaultdict
import sys

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
        def __init__(self, iterable=None, total=None, desc=None, disable=False):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.disable = disable
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                if not self.disable and self.total:
                    pct = int(100 * self.n / self.total)
                    print(f"\r{self.desc or 'Progreso'}: {pct}%", end="", flush=True)
            if not self.disable:
                print()
        def update(self, n=1):
            self.n += n
        def close(self):
            pass

# =============================================================================
# CACHE GLOBAL MEJORADO
# =============================================================================

class CacheManager:
    """Gestor centralizado de cachés para optimizar rendimiento."""
    
    def __init__(self):
        self.jaccard_cache: Dict[Tuple[FrozenSet, FrozenSet], float] = {}
        self.normalize_cache: Dict[str, str] = {}
        self.canonico_index: Dict[str, List[int]] = {}  # Índice de canónicos
        self.sinonimo_index: Dict[str, List[int]] = {}  # Índice de sinónimos
        self.stats = {
            "jaccard_hits": 0,
            "jaccard_misses": 0,
            "normalize_hits": 0,
            "normalize_misses": 0,
            "canonico_matches": 0,
            "sinonimo_matches": 0
        }
    
    def get_jaccard(self, set1: Set[str], set2: Set[str]) -> float:
        """Jaccard con cache."""
        key = (frozenset(set1), frozenset(set2))
        if key in self.jaccard_cache:
            self.stats["jaccard_hits"] += 1
            return self.jaccard_cache[key]
        
        key_inv = (frozenset(set2), frozenset(set1))
        if key_inv in self.jaccard_cache:
            self.stats["jaccard_hits"] += 1
            return self.jaccard_cache[key_inv]
        
        self.stats["jaccard_misses"] += 1
        if not set1 and not set2:
            value = 1.0
        elif not set1 or not set2:
            value = 0.0
        else:
            inter = len(set1 & set2)
            union = len(set1 | set2)
            value = inter / union if union else 0.0
        
        self.jaccard_cache[key] = value
        return value
    
    def normalize_text(self, s: str) -> str:
        """Normalización con cache."""
        if s in self.normalize_cache:
            self.stats["normalize_hits"] += 1
            return self.normalize_cache[s]
        
        self.stats["normalize_misses"] += 1
        if not isinstance(s, str):
            s = str(s or "")
        
        result = s.strip().lower()
        if UNIDECODE_AVAILABLE:
            result = unidecode(result)
        result = re.sub(r"[^\w\s]+", " ", result, flags=re.UNICODE)
        result = re.sub(r"\s+", " ", result, flags=re.UNICODE).strip()
        
        self.normalize_cache[s] = result
        return result
    
    def build_indices(self, wrappers: List['ConceptoWrapper']):
        """Construye índices para búsqueda rápida."""
        self.canonico_index.clear()
        self.sinonimo_index.clear()
        
        for i, w in enumerate(wrappers):
            # Índice de canónicos
            canon_norm = self.normalize_text(w.canonico)
            if canon_norm:
                if canon_norm not in self.canonico_index:
                    self.canonico_index[canon_norm] = []
                self.canonico_index[canon_norm].append(i)
            
            # Índice de sinónimos
            for sin in w.synonyms_norm:
                if sin:
                    if sin not in self.sinonimo_index:
                        self.sinonimo_index[sin] = []
                    self.sinonimo_index[sin].append(i)
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de cache."""
        return {
            **self.stats,
            "jaccard_cache_size": len(self.jaccard_cache),
            "normalize_cache_size": len(self.normalize_cache),
            "canonico_index_size": len(self.canonico_index),
            "sinonimo_index_size": len(self.sinonimo_index)
        }

# Instancia global del cache
CACHE = CacheManager()

# =============================================================================
# NORMALIZACIÓN
# =============================================================================

def normalize_text(s: str) -> str:
    """Proxy para normalización con cache."""
    return CACHE.normalize_text(s)

def norm_set(items: List[str]) -> Set[str]:
    """Set normalizado usando cache."""
    return {normalize_text(x) for x in items if str(x).strip()}

# =============================================================================
# DSU MEJORADO
# =============================================================================

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
        self.grupos_info = {}  # Info adicional sobre cada grupo
    
    def find(self, x: int) -> int:
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while x != root:
            next_x = self.p[x]
            self.p[x] = root
            x = next_x
        return root
    
    def union(self, a: int, b: int, razon: str = ""):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        
        # Guardar razón de unión
        if ra not in self.grupos_info:
            self.grupos_info[ra] = []
        if rb not in self.grupos_info:
            self.grupos_info[rb] = []
        
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
            self.grupos_info[rb].extend(self.grupos_info.get(ra, []))
            self.grupos_info[rb].append(f"{a}-{b}: {razon}")
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
            self.grupos_info[ra].extend(self.grupos_info.get(rb, []))
            self.grupos_info[ra].append(f"{a}-{b}: {razon}")
        else:
            self.p[rb] = ra
            self.r[ra] += 1
            self.grupos_info[ra].extend(self.grupos_info.get(rb, []))
            self.grupos_info[ra].append(f"{a}-{b}: {razon}")

# =============================================================================
# WRAPPER MEJORADO
# =============================================================================

@dataclass
class ConceptoWrapper:
    """Wrapper optimizado con lazy loading de términos normalizados."""
    raw: Dict[str, Any]
    canonico: str
    concepto: str
    concept_id: str
    tipo: str
    tipo_norm: str = ""
    _synonyms_norm: Optional[Set[str]] = None
    _all_terms_norm: Optional[Set[str]] = None
    
    @property
    def synonyms_norm(self) -> Set[str]:
        """Lazy loading de sinónimos normalizados."""
        if self._synonyms_norm is None:
            sinonimos = self.raw.get("sinonimos") or []
            etiquetados = self.raw.get("sinonimos_etiquetados") or []
            sin_etq = [x.get("texto", "") for x in etiquetados if isinstance(x, dict)]
            sin_total = list({*sinonimos, *sin_etq})
            self._synonyms_norm = norm_set(sin_total)
        return self._synonyms_norm
    
    @property
    def all_terms_norm(self) -> Set[str]:
        """Lazy loading de todos los términos normalizados."""
        if self._all_terms_norm is None:
            canon_norm = normalize_text(self.canonico)
            conc_norm = normalize_text(self.concepto)
            self._all_terms_norm = {canon_norm, conc_norm, *self.synonyms_norm}
        return self._all_terms_norm
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConceptoWrapper":
        """Creación optimizada con normalización mínima."""
        concepto = d.get("concepto") or d.get("termino_original") or ""
        canonico = d.get("canonico") or concepto
        concept_id = d.get("concept_id") or ""
        tipo = d.get("tipo") or ""
        tipo_norm = normalize_text(tipo) if tipo else ""
        
        return cls(
            raw=d,
            canonico=canonico,
            concepto=concepto,
            concept_id=concept_id,
            tipo=tipo,
            tipo_norm=tipo_norm
        )

# =============================================================================
# PREFILTRADO POR TIPO MEJORADO
# =============================================================================

class PrefiltradoPorTipo:
    """Agrupa conceptos por tipo para reducir comparaciones innecesarias."""
    
    def __init__(self, verbose: bool = False, aggressive: bool = False):
        self.verbose = verbose
        self.aggressive = aggressive
        self.tipos_compatibles = {
            ("body_structure", "anatomia"): True,
            ("diagnostico", "hallazgo"): True,
            ("procedimiento", "intervencion"): True,
            ("diagnostico", "hallazgo sintoma"): True,
        }
    
    def agrupar_por_tipo(self, wrappers: List[ConceptoWrapper]) -> Dict[str, List[int]]:
        """Agrupa índices de wrappers por tipo normalizado."""
        grupos = defaultdict(list)
        
        for i, w in enumerate(wrappers):
            tipo_key = w.tipo_norm or "sin_tipo"
            grupos[tipo_key].append(i)
        
        if self.verbose:
            print(f"Grupos por tipo: {dict([(k, len(v)) for k, v in grupos.items()])}")
        
        return grupos
    
    def son_tipos_compatibles(self, tipo1: str, tipo2: str) -> bool:
        """Determina si dos tipos pueden potencialmente fusionarse."""
        if not tipo1 or not tipo2:
            return True
        
        if tipo1 == tipo2:
            return True
        
        # En modo agresivo, ser más permisivo
        if self.aggressive:
            # Permitir fusión entre tipos similares
            similares = [
                ("diagnostico", "hallazgo"),
                ("procedimiento", "intervencion"),
                ("body_structure", "anatomia"),
                ("farmaco", "medicamento"),
            ]
            for t1, t2 in similares:
                if (t1 in tipo1 and t2 in tipo2) or (t2 in tipo1 and t1 in tipo2):
                    return True
        
        # Verificar compatibilidades definidas
        for (t1, t2), compatible in self.tipos_compatibles.items():
            if (tipo1 in t1 and tipo2 in t2) or (tipo2 in t1 and tipo1 in t2):
                return compatible
        
        # Tipos claramente incompatibles
        incompatibles = [
            ("medicamento", "procedimiento"),
            ("medicamento", "body_structure"),
            ("farmaco", "anatomia"),
            ("dispositivo", "medicamento"),
            ("laboratorio", "procedimiento"),
        ]
        
        for t1, t2 in incompatibles:
            if (t1 in tipo1 and t2 in tipo2) or (t2 in tipo1 and t1 in tipo2):
                return False
        
        return False
    
    def generar_pares_candidatos(self, 
                                 wrappers: List[ConceptoWrapper],
                                 grupos_tipo: Dict[str, List[int]]) -> List[Tuple[int, int]]:
        """Genera solo los pares que necesitan ser comparados."""
        pares = []
        tipos = list(grupos_tipo.keys())
        
        # Comparar dentro del mismo tipo
        for tipo, indices in grupos_tipo.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pares.append((indices[i], indices[j]))
        
        # Comparar entre tipos compatibles
        for i in range(len(tipos)):
            for j in range(i + 1, len(tipos)):
                if self.son_tipos_compatibles(tipos[i], tipos[j]):
                    for idx1 in grupos_tipo[tipos[i]]:
                        for idx2 in grupos_tipo[tipos[j]]:
                            if idx1 < idx2:
                                pares.append((idx1, idx2))
        
        if self.verbose:
            total_posible = len(wrappers) * (len(wrappers) - 1) // 2
            print(f"Pares a comparar: {len(pares)} de {total_posible} posibles")
            print(f"Reducción: {100 * (1 - len(pares)/max(1, total_posible)):.1f}%")
        
        return pares

# =============================================================================
# FUNCIONES DE FUSIÓN
# =============================================================================

def merge_etiquetados(a: List[Dict], b: List[Dict]) -> List[Dict]:
    """Fusión de sinónimos etiquetados."""
    by_key: Dict[str, Dict] = {}
    
    def _add(item: Dict):
        if not isinstance(item, dict):
            return
        texto = item.get("texto", "")
        k = normalize_text(texto)
        if not k:
            return
            
        if k not in by_key:
            by_key[k] = {
                "texto": texto,
                "etiquetas": list({*(item.get("etiquetas") or [])}),
                "fuente": item.get("fuente", ""),
                "confianza": float(item.get("confianza", 0) or 0),
                "score_mx": float(item.get("score_mx", 0) or 0),
                "notas": item.get("notas", ""),
                "region": item.get("region", ""),
            }
        else:
            existing = by_key[k]
            etiquetas = set(existing.get("etiquetas") or [])
            etiquetas.update(item.get("etiquetas") or [])
            existing["etiquetas"] = sorted(etiquetas)
            existing["confianza"] = max(
                float(existing.get("confianza", 0)),
                float(item.get("confianza", 0) or 0)
            )
            existing["score_mx"] = max(
                float(existing.get("score_mx", 0)),
                float(item.get("score_mx", 0) or 0)
            )
            if not existing.get("fuente") and item.get("fuente"):
                existing["fuente"] = item.get("fuente")
            notas = set(filter(None, [existing.get("notas", ""), item.get("notas", "")]))
            existing["notas"] = " | ".join(sorted(notas)) if notas else ""
            if item.get("region") and item["region"] != existing.get("region"):
                if existing.get("region"):
                    extra = f"región alternativa: {item['region']}"
                    existing["notas"] = (existing["notas"] + " | " + extra).strip(" |")
                else:
                    existing["region"] = item["region"]
    
    for lst in (a or []), (b or []):
        for it in lst:
            _add(it)
    
    return list(by_key.values())

def choose_representative(a: Dict, b: Dict) -> Dict:
    """Elige el concepto más completo."""
    def score(d: Dict) -> Tuple[int, int, int]:
        val_score = 0
        if d.get("validado_cie10"):
            val_score += 1
        if d.get("validado_cemece"):
            val_score += 1
        n_syn = len(d.get("sinonimos_etiquetados") or [])
        n_notas = len(d.get("notas") or "")
        return (val_score, n_syn, n_notas)
    
    return a if score(a) >= score(b) else b

def recompute_metricas_mx(concept: Dict, umbral_alto: float = 0.5) -> Dict:
    """Recalcula métricas MX."""
    etq = concept.get("sinonimos_etiquetados") or []
    scores = [float(x.get("score_mx", 0) or 0) for x in etq]
    total = len(scores)
    
    if total == 0:
        return {
            "promedio_score": 0.0,
            "max_score": 0.0,
            "count_alto": 0,
            "total_sinonimos": 0
        }
    
    return {
        "promedio_score": sum(scores) / total,
        "max_score": max(scores),
        "count_alto": sum(1 for s in scores if s >= umbral_alto),
        "total_sinonimos": total
    }

def recompute_metadata(concepts: List[Dict], base_meta: Dict) -> Dict:
    """Recálculo completo de metadata."""
    m = {}
    m["fecha_generacion"] = datetime.utcnow().isoformat()
    m["total_conceptos"] = len(concepts)
    
    val_cie10_exactos = 0
    val_cie10_parciales = 0
    val_cemece = 0
    val_ambas = 0
    prom_scores = []
    tipos = set(base_meta.get("tipos_encontrados") or [])
    
    umbral_mex = float(base_meta.get("umbral_mexicanidad", 0.15))
    bien_mex = 0
    enriquec_pend = 0
    total_meds = 0
    meds_con_marcas = 0
    
    for c in concepts:
        if c.get("validado_cie10"):
            cal = (c.get("cie10_match_calidad") or "").lower()
            if cal == "exacto":
                val_cie10_exactos += 1
            else:
                val_cie10_parciales += 1
        
        if c.get("validado_cemece"):
            val_cemece += 1
        
        if c.get("validado_cie10") and c.get("validado_cemece"):
            val_ambas += 1
        
        metric = c.get("metricas_mx") or {}
        if isinstance(metric, dict) and "promedio_score" in metric:
            try:
                score_val = float(metric["promedio_score"])
                prom_scores.append(score_val)
                if score_val >= umbral_mex:
                    bien_mex += 1
            except:
                pass
        
        if c.get("tipo"):
            tipos.add(c.get("tipo"))
        
        if c.get("enriquecimiento_pendiente"):
            enriquec_pend += 1
        
        tipo_norm = normalize_text(str(c.get("tipo", "")))
        if "medicamento" in tipo_norm or "farmaco" in tipo_norm:
            total_meds += 1
            marcas = c.get("marcas") or []
            if isinstance(marcas, list) and len(marcas) > 0:
                meds_con_marcas += 1
    
    m["validados_cie10_exactos"] = val_cie10_exactos
    m["validados_cie10_parciales"] = val_cie10_parciales
    m["validados_cemece"] = val_cemece
    m["validados_ambas_fuentes"] = val_ambas
    
    if prom_scores:
        m["score_mexicanidad_promedio"] = f"{(sum(prom_scores)/len(prom_scores)):.3f}"
    else:
        m["score_mexicanidad_promedio"] = "0.000"
    
    m["terminos_bien_mexicanizados"] = bien_mex
    m["terminos_enriquecimiento_pendiente"] = enriquec_pend
    m["medicamentos_con_marcas"] = meds_con_marcas
    m["total_medicamentos"] = total_meds
    
    m["version"] = base_meta.get("version", "4.0.1-mejorado")
    m["fuentes_validacion"] = sorted(set(
        base_meta.get("fuentes_validacion", ["CIE-10", "CEMECE"])
    ))
    m["lexico_mexicano_etiquetado"] = True
    m["marcas_desde_lista_blanca"] = base_meta.get("marcas_desde_lista_blanca", True)
    m["umbral_mexicanidad"] = umbral_mex
    m["tipos_encontrados"] = sorted(tipos) if tipos else []
    
    return m

# =============================================================================
# UNIFICADOR MEJORADO
# =============================================================================

class UnificadorMejorado:
    """
    Unificador con detección mejorada de duplicados.
    """
    
    def __init__(self, 
                 jaccard_threshold: float = 0.5,
                 aggressive: bool = False,
                 verbose: bool = False,
                 mostrar_progreso: bool = True):
        self.jaccard_threshold = jaccard_threshold
        self.aggressive = aggressive
        self.verbose = verbose
        self.mostrar_progreso = mostrar_progreso and TQDM_AVAILABLE
        
        self.prefiltrador = PrefiltradoPorTipo(verbose, aggressive)
        self.conceptos_unificados = []
        self.metadata_global = {}
    
    def log(self, msg: str):
        if self.verbose:
            print(f"[INFO] {msg}")
    
    def cargar_archivos(self, rutas: List[str]) -> Tuple[List[Dict], Dict]:
        """Carga archivos."""
        conceptos_totales = []
        metadata_combinada = {}
        
        for ruta in rutas:
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        conceptos = data.get('conceptos', [])
                        metadata = data.get('metadata', {})
                        conceptos_totales.extend(conceptos)
                        if metadata:
                            metadata_combinada.update(metadata)
            except Exception as e:
                print(f"Error al cargar {ruta}: {e}")
                continue
        
        return conceptos_totales, metadata_combinada
    
    def detectar_duplicados_mejorado(self, wrappers: List[ConceptoWrapper]) -> List[Tuple[int, int, str]]:
        """
        Detección mejorada de duplicados con múltiples estrategias.
        """
        # Construir índices para búsqueda rápida
        CACHE.build_indices(wrappers)
        
        pares_duplicados = set()  # Usar set para evitar duplicados
        
        # Estrategia 1: Detectar canónicos idénticos usando índice
        self.log("Detectando canónicos idénticos...")
        for canon_norm, indices in CACHE.canonico_index.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pares_duplicados.add((min(indices[i], indices[j]), 
                                            max(indices[i], indices[j]), 
                                            f"canónico idéntico: {canon_norm}"))
                        CACHE.stats["canonico_matches"] += 1
        
        # Estrategia 2: Detectar cruces canónico-sinónimo
        self.log("Detectando cruces canónico-sinónimo...")
        for i, w in enumerate(wrappers):
            canon_norm = normalize_text(w.canonico)
            
            # Buscar si este canónico aparece en sinónimos de otros
            if canon_norm in CACHE.sinonimo_index:
                for j in CACHE.sinonimo_index[canon_norm]:
                    if i != j:
                        pares_duplicados.add((min(i, j), max(i, j), 
                                            f"canónico '{w.canonico}' en sinónimos"))
                        CACHE.stats["sinonimo_matches"] += 1
        
        # Estrategia 3: Detectar por concept_id
        self.log("Detectando por concept_id...")
        id_map = defaultdict(list)
        for i, w in enumerate(wrappers):
            if w.concept_id:
                id_map[w.concept_id].append(i)
        
        for concept_id, indices in id_map.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pares_duplicados.add((min(indices[i], indices[j]), 
                                            max(indices[i], indices[j]), 
                                            f"concept_id={concept_id}"))
        
        # Estrategia 4: Agrupar por tipo y comparar con Jaccard
        self.log("Detectando por similitud de Jaccard...")
        grupos_tipo = self.prefiltrador.agrupar_por_tipo(wrappers)
        pares_candidatos = self.prefiltrador.generar_pares_candidatos(wrappers, grupos_tipo)
        
        if self.mostrar_progreso:
            pbar = tqdm(pares_candidatos, desc="Comparando por Jaccard")
        else:
            pbar = pares_candidatos
        
        for i, j in pbar:
            # Skip si ya está marcado como duplicado
            if (i, j) in pares_duplicados or (j, i) in pares_duplicados:
                continue
            
            wi, wj = wrappers[i], wrappers[j]
            
            # Calcular Jaccard
            jac = CACHE.get_jaccard(wi.all_terms_norm, wj.all_terms_norm)
            
            # Umbral más bajo si es modo agresivo
            threshold = self.jaccard_threshold * 0.8 if self.aggressive else self.jaccard_threshold
            
            if jac >= threshold:
                pares_duplicados.add((min(i, j), max(i, j), f"Jaccard={jac:.3f}"))
        
        if self.mostrar_progreso and hasattr(pbar, 'close'):
            pbar.close()
        
        # Convertir a lista
        pares_lista = list(pares_duplicados)
        
        self.log(f"Total de pares duplicados detectados: {len(pares_lista)}")
        
        return pares_lista
    
    def agrupar_con_dsu(self, n: int, pares: List[Tuple[int, int, str]]) -> Dict[int, List[int]]:
        """Agrupa con Union-Find mejorado."""
        dsu = DSU(n)
        
        for i, j, razon in pares:
            dsu.union(i, j, razon)
            if self.verbose and len(pares) < 30:
                self.log(f"  → Union ({i},{j}): {razon}")
        
        grupos = {}
        for idx in range(n):
            root = dsu.find(idx)
            grupos.setdefault(root, []).append(idx)
        
        # Mostrar info de grupos grandes
        if self.verbose:
            grandes = [(root, len(indices)) for root, indices in grupos.items() if len(indices) > 2]
            if grandes:
                self.log(f"Grupos grandes encontrados: {grandes[:5]}")
        
        return grupos
    
    def fusionar_grupo(self, wrappers: List[ConceptoWrapper], indices: List[int]) -> Dict:
        """Fusiona conceptos de un grupo."""
        if len(indices) == 1:
            concepto = wrappers[indices[0]].raw
            concepto["metricas_mx"] = recompute_metricas_mx(concepto)
            return concepto
        
        # Elegir el mejor representante
        mejor_idx = indices[0]
        mejor_score = 0
        
        for idx in indices:
            score = 0
            c = wrappers[idx].raw
            if c.get("validado_cie10"): score += 2
            if c.get("validado_cemece"): score += 2
            score += len(c.get("sinonimos", []))
            score += len(c.get("sinonimos_etiquetados", []))
            
            if score > mejor_score:
                mejor_score = score
                mejor_idx = idx
        
        # Usar el mejor como base
        base = wrappers[mejor_idx].raw.copy()
        nombres = []
        todos_sinonimos = set(base.get("sinonimos", []))
        
        # Fusionar todos
        for idx in indices:
            concepto = wrappers[idx].raw
            nombre = concepto.get("concepto", "")
            if nombre:
                nombres.append(nombre)
            
            # Agregar todos los términos como sinónimos
            todos_sinonimos.add(concepto.get("canonico", ""))
            todos_sinonimos.add(concepto.get("concepto", ""))
            todos_sinonimos.update(concepto.get("sinonimos", []))
            
            # Fusionar sinónimos etiquetados
            base["sinonimos_etiquetados"] = merge_etiquetados(
                base.get("sinonimos_etiquetados", []),
                concepto.get("sinonimos_etiquetados", [])
            )
            
            # OR de validaciones
            base["validado_cie10"] = base.get("validado_cie10", False) or concepto.get("validado_cie10", False)
            base["validado_cemece"] = base.get("validado_cemece", False) or concepto.get("validado_cemece", False)
            
            # Mejorar calidad si es exacto
            if concepto.get("cie10_match_calidad", "").lower() == "exacto":
                base["cie10_match_calidad"] = "exacto"
        
        # Limpiar y actualizar sinónimos
        # Remover el canónico y concepto actual de los sinónimos
        canonico_base = base.get("canonico", "")
        concepto_base = base.get("concepto", "")
        todos_sinonimos.discard(canonico_base)
        todos_sinonimos.discard(concepto_base)
        todos_sinonimos.discard("")
        
        base["sinonimos"] = sorted(list(todos_sinonimos))
        
        # Info de fusión
        base["_fusion_info"] = {
            "conceptos_fusionados": nombres,
            "total_fusionados": len(nombres),
            "fecha_fusion": datetime.utcnow().isoformat(),
            "metodo": "Mejorado-Agresivo" if self.aggressive else "Mejorado"
        }
        
        # Recalcular métricas
        base["metricas_mx"] = recompute_metricas_mx(base)
        
        return base
    
    def unificar(self, rutas_archivos: List[str]) -> Dict:
        """Pipeline completo mejorado."""
        inicio = time.time()
        
        # 1. Cargar archivos
        self.log(f"Cargando {len(rutas_archivos)} archivos...")
        conceptos, metadata_base = self.cargar_archivos(rutas_archivos)
        self.log(f"Total conceptos cargados: {len(conceptos)}")
        
        if not conceptos:
            return {"metadata": metadata_base, "conceptos": []}
        
        # 2. Crear wrappers
        self.log("Creando wrappers normalizados...")
        wrappers = [ConceptoWrapper.from_dict(c) for c in conceptos]
        
        # 3. Detectar duplicados con estrategias mejoradas
        self.log("Detectando duplicados (mejorado)...")
        pares = self.detectar_duplicados_mejorado(wrappers)
        self.log(f"Pares de duplicados detectados: {len(pares)}")
        
        # 4. Agrupar con DSU
        self.log("Agrupando con Union-Find...")
        grupos = self.agrupar_con_dsu(len(wrappers), pares)
        self.log(f"Grupos formados: {len(grupos)}")
        
        # Estadísticas de agrupamiento
        tamaños = [len(indices) for indices in grupos.values()]
        grupos_grandes = sum(1 for t in tamaños if t > 1)
        self.log(f"Grupos con fusión: {grupos_grandes}")
        self.log(f"Grupo más grande: {max(tamaños)} conceptos")
        
        # 5. Fusionar grupos
        self.log("Fusionando grupos...")
        self.conceptos_unificados = []
        
        if self.mostrar_progreso:
            grupos_items = tqdm(grupos.items(), desc="Fusionando grupos")
        else:
            grupos_items = grupos.items()
        
        for root, indices in grupos_items:
            concepto_fusionado = self.fusionar_grupo(wrappers, indices)
            self.conceptos_unificados.append(concepto_fusionado)
        
        # 6. Recalcular metadata
        self.log("Recalculando metadata...")
        self.metadata_global = recompute_metadata(self.conceptos_unificados, metadata_base)
        
        # 7. Agregar estadísticas
        tiempo_total = time.time() - inicio
        reduccion = len(conceptos) - len(self.conceptos_unificados)
        
        self.metadata_global["_procesamiento"] = {
            "tiempo_segundos": round(tiempo_total, 2),
            "conceptos_originales": len(conceptos),
            "conceptos_finales": len(self.conceptos_unificados),
            "conceptos_fusionados": reduccion,
            "porcentaje_reduccion": round(100 * reduccion / len(conceptos), 2),
            "archivos_procesados": len(rutas_archivos),
            "cache_stats": CACHE.get_stats(),
            "modo": "agresivo" if self.aggressive else "normal"
        }
        
        self.log(f"Completado en {tiempo_total:.1f} segundos")
        self.log(f"Reducción: {reduccion} conceptos ({self.metadata_global['_procesamiento']['porcentaje_reduccion']}%)")
        
        return {
            "metadata": self.metadata_global,
            "conceptos": self.conceptos_unificados
        }
    
    def guardar(self, resultado: Dict, ruta_salida: str):
        """Guarda resultado."""
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Guardado en: {ruta_salida}")
        print(f"  Conceptos: {resultado['metadata']['total_conceptos']}")
        print(f"  Score MX: {resultado['metadata']['score_mexicanidad_promedio']}")
        print(f"  Tiempo: {resultado['metadata']['_procesamiento']['tiempo_segundos']}s")
        print(f"  Reducción: {resultado['metadata']['_procesamiento']['porcentaje_reduccion']}%")
        
        # Mostrar stats mejoradas
        if self.verbose:
            stats = resultado['metadata']['_procesamiento']['cache_stats']
            print(f"\nEstadísticas de detección:")
            print(f"  Matches por canónico: {stats.get('canonico_matches', 0)}")
            print(f"  Matches por sinónimo: {stats.get('sinonimo_matches', 0)}")
            print(f"  Cache Jaccard: {stats['jaccard_hits']} hits, {stats['jaccard_misses']} misses")


def main():
    parser = argparse.ArgumentParser(
        description='Unificador mejorado con detección agresiva de duplicados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s *.json -o resultado.json
  %(prog)s *.json -o resultado.json --aggressive
  %(prog)s *.json -o resultado.json --jaccard 0.4 --verbose
  %(prog)s data/*.json -o resultado.json --no-progress
        """
    )
    
    parser.add_argument(
        'archivos',
        nargs='+',
        help='Archivos JSON a unificar (acepta wildcards)'
    )
    parser.add_argument(
        '-o', '--output',
        default='unificado_mejorado.json',
        help='Archivo de salida (default: unificado_mejorado.json)'
    )
    parser.add_argument(
        '--jaccard',
        type=float,
        default=0.5,
        help='Umbral Jaccard para fusión (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Modo agresivo: detecta más duplicados potenciales'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar información detallada'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Desactivar barra de progreso'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='No guardar archivo, solo mostrar estadísticas'
    )
    
    args = parser.parse_args()
    
    # Expandir wildcards
    import glob
    archivos_expandidos = []
    for patron in args.archivos:
        if '*' in patron or '?' in patron:
            archivos_expandidos.extend(glob.glob(patron))
        else:
            archivos_expandidos.append(patron)
    
    if not archivos_expandidos:
        print("Error: No se encontraron archivos")
        return 1
    
    print("=" * 70)
    print("UNIFICADOR MEJORADO DE JSONs MÉDICOS v2")
    print("=" * 70)
    print(f"Archivos a procesar: {len(archivos_expandidos)}")
    print(f"Umbral Jaccard: {args.jaccard}")
    print(f"Modo: {'AGRESIVO' if args.aggressive else 'Normal'}")
    print(f"Mejoras activas:")
    print(f"  ✓ Detección de canónicos duplicados")
    print(f"  ✓ Detección de cruces canónico-sinónimo")
    print(f"  ✓ Índices para búsqueda rápida")
    print(f"  ✓ Fusión mejorada con preservación de información")
    if args.aggressive:
        print(f"  ✓ Modo agresivo activado")
    print()
    
    # Crear unificador
    unificador = UnificadorMejorado(
        jaccard_threshold=args.jaccard,
        aggressive=args.aggressive,
        verbose=args.verbose,
        mostrar_progreso=not args.no_progress
    )
    
    # Ejecutar
    try:
        resultado = unificador.unificar(archivos_expandidos)
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\nError durante la unificación: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Guardar o mostrar
    if not args.dry_run:
        unificador.guardar(resultado, args.output)
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    proc = resultado['metadata'].get('_procesamiento', {})
    print(f"Tiempo total: {proc.get('tiempo_segundos', 0):.1f} segundos")
    print(f"Conceptos originales: {proc.get('conceptos_originales', 0)}")
    print(f"Conceptos finales: {proc.get('conceptos_finales', 0)}")
    print(f"Reducción: {proc.get('conceptos_fusionados', 0)} conceptos")
    print(f"Porcentaje de reducción: {proc.get('porcentaje_reduccion', 0)}%")
    
    if args.verbose and proc.get('cache_stats'):
        print(f"\nEstadísticas de detección:")
        stats = proc['cache_stats']
        print(f"  Matches por canónico idéntico: {stats.get('canonico_matches', 0)}")
        print(f"  Matches por cruce canónico-sinónimo: {stats.get('sinonimo_matches', 0)}")
        total_j = stats['jaccard_hits'] + stats['jaccard_misses']
        if total_j > 0:
            print(f"  Cache Jaccard hit rate: {100*stats['jaccard_hits']/total_j:.1f}%")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)