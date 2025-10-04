#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_conceptos.py

Une múltiples archivos JSON con el formato indicado (metadata + "conceptos": [ ... ]),
deduplica conceptos cercanos (e.g., "Abdomen" vs "Abdominal") y unifica sinónimos,
sin depender necesariamente de un LLM. Opcionalmente, puede usar un hook para
consultar un LLM (por ejemplo, gpt-oss:20b) cuando existan dudas de duplicidad.

Uso:
  python merge_conceptos.py -o salida.json a.json b.json c.json
  python merge_conceptos.py -o salida.json --glob "data/*.json"

Argumentos clave:
  --jaccard  : umbral de similitud (0.0–1.0) para unión por superposición de sinónimos (default: 0.5)
  --use-llm  : activa el hook para preguntar a un LLM si dos conceptos deben fusionarse (default: False)
  --dry-run  : no escribe salida, solo reporta resumen en consola
  --verbose  : imprime detalles de las fusiones
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from unidecode import unidecode
except Exception:
    def unidecode(s: str) -> str:
        return s

# ---------------------------
# Utilidades de normalización
# ---------------------------

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

def normalize_text(s: str) -> str:
    """Normaliza texto para comparaciones robustas."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = unidecode(s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def norm_set(items: List[str]) -> Set[str]:
    return {normalize_text(x) for x in items if str(x).strip()}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ---------------------------
# LLM Hook (opcional)
# ---------------------------

def llm_are_synonyms(c1: Dict[str, Any], c2: Dict[str, Any]) -> Optional[bool]:
    """
    Hook opcional para consultar un LLM (e.g., gpt-oss:20b) y decidir si c1 y c2
    representan el mismo concepto. Devuelve:
      - True  : debe fusionarse
      - False : no debe fusionarse
      - None  : indecisión (deja la decisión al algoritmo por Jaccard)
    Por defecto, no llama a ningún servicio. Personaliza este método para integrar tu LLM.
    """
    # EJEMPLO (pseudo-código):
    # prompt = f"¿'${c1['canonico']}' y '{c2['canonico']}' son el mismo concepto médico? ..."
    # resp = tu_cliente_gpt_oss_20b.complete(prompt)
    # return parse_boolean(resp)
    return None

# ---------------------------
# Estructuras y fusión
# ---------------------------

@dataclass
class Concepto:
    raw: Dict[str, Any]
    canonico: str
    concepto: str
    synonyms_norm: Set[str] = field(default_factory=set)
    all_terms_norm: Set[str] = field(default_factory=set)  # incluye canonico + concepto + sinonimos

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Concepto":
        concepto = d.get("concepto") or d.get("termino_original") or ""
        canonico = d.get("canonico") or concepto
        # Columna sinonimos (lista simple)
        sinonimos = d.get("sinonimos") or []
        # Extrae también de sinonimos_etiquetados
        etiquetados = d.get("sinonimos_etiquetados") or []
        sin_etq = [x.get("texto", "") for x in etiquetados if isinstance(x, dict)]
        sin_total = list({*sinonimos, *sin_etq})

        syn_norm = norm_set(sin_total)
        canon_norm = normalize_text(canonico)
        conc_norm = normalize_text(concepto)
        all_terms = {canon_norm, conc_norm, *syn_norm}
        return cls(raw=d, canonico=canonico, concepto=concepto, synonyms_norm=syn_norm, all_terms_norm=all_terms)

def merge_etiquetados(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Une listas de sinonimos_etiquetados, deduplicando por 'texto' normalizado.
    Fusiona etiquetas (union), toma max(confianza), max(score_mx) y concadena notas únicas.
    """
    by_key: Dict[str, Dict[str, Any]] = {}

    def _add(item: Dict[str, Any]):
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
            # Unir etiquetas únicas
            etiquetas = set(existing.get("etiquetas") or [])
            etiquetas.update(item.get("etiquetas") or [])
            existing["etiquetas"] = sorted(etiquetas)
            # Max de confianza y score_mx
            existing["confianza"] = max(float(existing.get("confianza") or 0), float(item.get("confianza") or 0))
            existing["score_mx"] = max(float(existing.get("score_mx") or 0), float(item.get("score_mx") or 0))
            # Prefiere la fuente no vacía
            if not existing.get("fuente") and item.get("fuente"):
                existing["fuente"] = item.get("fuente")
            # Combinar notas (evitar duplicadas)
            notas = set(filter(None, [existing.get("notas", ""), item.get("notas", "")]))
            existing["notas"] = " | ".join(sorted(notas)) if notas else ""
            # Region: si son distintas y no vacías, mantener la primera y anotar la segunda en notas
            if item.get("region") and item.get("region") != existing.get("region"):
                extra = f"región alternativa: {item['region']}"
                existing["notas"] = (existing["notas"] + " | " + extra).strip(" |")
            elif item.get("region") and not existing.get("region"):
                existing["region"] = item.get("region")

    for lst in (a or []), (b or []):
        for it in lst:
            if isinstance(it, dict):
                _add(it)

    return list(by_key.values())

def recompute_metricas_mx(concept: Dict[str, Any], umbral_alto: float = 0.5) -> Dict[str, Any]:
    etq = concept.get("sinonimos_etiquetados") or []
    scores = [float(x.get("score_mx", 0) or 0) for x in etq]
    total = len(scores)
    promedio = (sum(scores) / total) if total else 0.0
    max_score = max(scores) if scores else 0.0
    count_alto = sum(1 for s in scores if s >= umbral_alto)
    return {
        "promedio_score": promedio,
        "max_score": max_score,
        "count_alto": count_alto,
        "total_sinonimos": total,
    }

def choose_representative(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide cuál ficha conservar como base al fusionar. Heurísticas:
    - Prefiere la que tenga validado_cie10 y validado_cemece verdaderos.
    - Si empatan, prefiere la que tenga más sinonimos_etiquetados.
    - Si empatan, prefiere la que tenga notas más largas.
    """
    def score(d: Dict[str, Any]) -> Tuple[int, int, int]:
        c1 = 1 if d.get("validado_cie10") else 0
        c2 = 1 if d.get("validado_cemece") else 0
        n_syn = len(d.get("sinonimos_etiquetados") or [])
        n_notas = len((d.get("notas") or ""))
        return (c1 + c2, n_syn, n_notas)
    return a if score(a) >= score(b) else b

def merge_concepts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona los dicts de dos conceptos. Mantiene un solo concept_id (del representativo).
    Unifica sinonimos y sinonimos_etiquetados; recalcula métricas.
    """
    rep = choose_representative(a, b)
    other = b if rep is a else a

    out = {**rep}  # copia
    # concept_id: conservar del representativo
    # concepto/canonico: mantener del representativo
    # tipo: mantener del representativo

    # Unir sinónimos simples
    syn_a = set(a.get("sinonimos") or [])
    syn_b = set(b.get("sinonimos") or [])
    out["sinonimos"] = sorted({*syn_a, *syn_b})

    # Unir etiquetados
    out["sinonimos_etiquetados"] = merge_etiquetados(a.get("sinonimos_etiquetados") or [],
                                                     b.get("sinonimos_etiquetados") or [])

    # Unir notas (sin duplicar)
    notas_set = set(filter(None, [(a.get("notas") or "").strip(), (b.get("notas") or "").strip()]))
    if notas_set:
        out["notas"] = " | ".join(sorted(notas_set))

    # Unir banderas de validación (OR lógico)
    out["validado_cie10"] = bool(a.get("validado_cie10") or b.get("validado_cie10"))
    out["validado_cemece"] = bool(a.get("validado_cemece") or b.get("validado_cemece"))

    # cie10_match_calidad: preferir 'exacto' si alguna lo es; sino, conservar del representativo
    cal_a = (a.get("cie10_match_calidad") or "").lower()
    cal_b = (b.get("cie10_match_calidad") or "").lower()
    if "exacto" in (cal_a, cal_b):
        out["cie10_match_calidad"] = "exacto"

    # termino_original: preferir el del representativo; si difiere, anexar a notas
    if a.get("termino_original") and b.get("termino_original") and a.get("termino_original") != b.get("termino_original"):
        extra = f"también referido como '{b.get('termino_original')}'"
        out["notas"] = ((out.get("notas") or "") + " | " + extra).strip(" |")

    # enriq pendiente: OR
    out["enriquecimiento_pendiente"] = bool(a.get("enriquecimiento_pendiente") or b.get("enriquecimiento_pendiente"))

    # Recalcular métricas por concepto
    out["metricas_mx"] = recompute_metricas_mx(out)

    return out

# ---------------------------
# Unión por componentes (Union-Find)
# ---------------------------

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

# ---------------------------
# Pipeline principal
# ---------------------------

def load_inputs(paths: List[str]) -> List[Dict[str, Any]]:
    docs = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        docs.append(data)
    return docs

def collect_concepts(docs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_concepts: List[Dict[str, Any]] = []
    combined_meta: Dict[str, Any] = {}
    metas = [d.get("metadata") or {} for d in docs if isinstance(d, dict)]
    # Combinar metadatos suavemente (mantendremos solo parámetros útiles/constantes).
    # El recomputo final sobrescribirá los campos pedestres (totales, promedios, etc.).
    # Escoge umbral_mexicanidad si existe en alguna metadata, default 0.15
    umbral = None
    tipos: Set[str] = set()
    fuentes: Set[str] = set()
    version = None
    for m in metas:
        if umbral is None and "umbral_mexicanidad" in m:
            umbral = m.get("umbral_mexicanidad")
        if "tipos_encontrados" in m and isinstance(m["tipos_encontrados"], list):
            tipos.update(m["tipos_encontrados"])
        if "fuentes_validacion" in m and isinstance(m["fuentes_validacion"], list):
            fuentes.update(m["fuentes_validacion"])
        if version is None and m.get("version"):
            version = m.get("version")

    combined_meta["umbral_mexicanidad"] = umbral if umbral is not None else 0.15
    combined_meta["tipos_encontrados"] = sorted(tipos) if tipos else []
    combined_meta["fuentes_validacion"] = sorted(fuentes) if fuentes else ["CIE-10", "CEMECE"]
    combined_meta["version"] = version or "3.5.4"  # conserva o usa la que llegó como ejemplo

    for d in docs:
        conceptos = d.get("conceptos") or []
        for c in conceptos:
            if isinstance(c, dict):
                all_concepts.append(c)
    return all_concepts, combined_meta

def initial_bucket_by_canonical(concepts: List[Dict[str, Any]]) -> List[List[Concepto]]:
    buckets: Dict[str, List[Concepto]] = {}
    for c in concepts:
        obj = Concepto.from_dict(c)
        key = normalize_text(obj.canonico or obj.concepto)
        buckets.setdefault(key, []).append(obj)
    return list(buckets.values())

def fuse_bucket(objs: List[Concepto]) -> Dict[str, Any]:
    """Fusiona todos los Concepto de un bucket (mismo canonico normalizado)."""
    base = objs[0].raw
    for o in objs[1:]:
        base = merge_concepts(base, o.raw)
    return base

def cross_bucket_merge(concepts: List[Dict[str, Any]], jaccard_thr: float, use_llm: bool, verbose: bool) -> List[Dict[str, Any]]:
    """
    Segunda pasada: detecta duplicados entre buckets diferentes usando:
    - Jaccard de conjuntos de términos (sinónimos + canon + concepto).
    - Si un canonico aparece como sinónimo del otro (y viceversa).
    - (Opcional) LLM hook para decidir en casos ambigüos.
    """
    wrappers = [Concepto.from_dict(c) for c in concepts]
    n = len(wrappers)
    dsu = DSU(n)

    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = wrappers[i], wrappers[j]
            # Evita unir tipos radicalmente distintos si ambos tienen 'tipo' y difieren
            ti, tj = (wi.raw.get("tipo"), wj.raw.get("tipo"))
            if ti and tj and normalize_text(str(ti)) != normalize_text(str(tj)):
                continue

            jac = jaccard(wi.all_terms_norm, wj.all_terms_norm)

            should_merge = False
            reason = ""

            if jac >= jaccard_thr:
                should_merge = True
                reason = f"Jaccard={jac:.2f}"

            # Si el canónico de uno aparece entre los sinónimos del otro
            if not should_merge:
                ci = normalize_text(wi.canonico)
                cj = normalize_text(wj.canonico)
                if (ci in wj.all_terms_norm) or (cj in wi.all_terms_norm):
                    should_merge = True
                    reason = f"canonico en sinónimos ({wi.canonico} ~ {wj.canonico})"

            # Hook LLM (si está activado)
            if not should_merge and use_llm:
                verdict = llm_are_synonyms(wi.raw, wj.raw)
                if verdict is True:
                    should_merge = True
                    reason = "LLM=True"
                elif verdict is False:
                    should_merge = False
                # None -> sin decisión, se mantiene criterio actual

            if should_merge:
                dsu.union(i, j)
                if verbose:
                    print(f"[merge] '{wi.canonico}' + '{wj.canonico}' -> {reason}")

    # Reunir por representante y fusionar
    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = dsu.find(idx)
        groups.setdefault(root, []).append(idx)

    merged: List[Dict[str, Any]] = []
    for _, idxs in groups.items():
        base = wrappers[idxs[0]].raw
        for k in idxs[1:]:
            base = merge_concepts(base, wrappers[k].raw)
        merged.append(base)

    return merged

def recompute_metadata(concepts: List[Dict[str, Any]], base_meta: Dict[str, Any]) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    now = datetime.utcnow().isoformat()
    m["fecha_generacion"] = now

    total = len(concepts)
    m["total_conceptos"] = total

    val_cie10_exactos = 0
    val_cie10_parciales = 0
    val_cemece = 0
    val_ambas = 0

    prom_scores = []
    tipos: Set[str] = set(base_meta.get("tipos_encontrados") or [])

    umbral_mex = base_meta.get("umbral_mexicanidad", 0.15)
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
                prom_scores.append(float(metric["promedio_score"]))
            except Exception:
                pass

        if c.get("tipo"):
            tipos.add(c.get("tipo"))

        if c.get("enriquecimiento_pendiente"):
            enriquec_pend += 1

        # Bien mexicanizados (promedio por concepto >= umbral_mexicanidad)
        try:
            if float(metric.get("promedio_score", 0)) >= float(umbral_mex):
                bien_mex += 1
        except Exception:
            pass

        # Medicamentos (heurística mínima)
        if normalize_text(str(c.get("tipo", ""))) == "medicamento":
            total_meds += 1
            marcas = c.get("marcas") or []
            if isinstance(marcas, list) and len(marcas) > 0:
                meds_con_marcas += 1

    m["validados_cie10_exactos"] = val_cie10_exactos
    m["validados_cie10_parciales"] = val_cie10_parciales
    m["validados_cemece"] = val_cemece
    m["validados_ambas_fuentes"] = val_ambas

    m["score_mexicanidad_promedio"] = f"{(sum(prom_scores)/len(prom_scores)):.3f}" if prom_scores else "0.000"
    m["terminos_bien_mexicanizados"] = bien_mex
    m["terminos_enriquecimiento_pendiente"] = enriquec_pend
    m["medicamentos_con_marcas"] = meds_con_marcas
    m["total_medicamentos"] = total_meds

    # Conserva/combina banderas y listas
    m["version"] = base_meta.get("version", "3.5.4")
    m["fuentes_validacion"] = base_meta.get("fuentes_validacion") or ["CIE-10", "CEMECE", "Lista blanca marcas MX"]
    m["lexico_mexicano_etiquetado"] = True  # asumimos que el corpus trae etiquetas
    m["marcas_desde_lista_blanca"] = True   # mantener True si se alimentó de lista blanca
    m["umbral_mexicanidad"] = umbral_mex
    m["tipos_encontrados"] = sorted(tipos) if tipos else []

    return m

def main():
    ap = argparse.ArgumentParser(description="Unifica y deduplica conceptos médicos en JSON.")
    ap.add_argument("inputs", nargs="*", help="Archivos JSON de entrada")
    ap.add_argument("--glob", help="Patrón glob para archivos (por ejemplo, 'data/*.json')")
    ap.add_argument("-o", "--output", required=False, help="Archivo JSON de salida")
    ap.add_argument("--jaccard", type=float, default=0.5, help="Umbral Jaccard de unión (default: 0.5)")
    ap.add_argument("--use-llm", action="store_true", help="Usar hook de LLM (gpt-oss:20b) para desambiguación")
    ap.add_argument("--dry-run", action="store_true", help="No escribe salida, solo imprime resumen")
    ap.add_argument("--verbose", action="store_true", help="Imprime detalles de fusiones")
    args = ap.parse_args()

    paths: List[str] = []
    if args.glob:
        import glob
        paths.extend(sorted(glob.glob(args.glob)))
    if args.inputs:
        paths.extend(args.inputs)

    if not paths:
        raise SystemExit("No se proporcionaron archivos de entrada. Usa archivos o --glob.")

    docs = load_inputs(paths)
    raw_concepts, base_meta = collect_concepts(docs)

    # PASO 1: agrupar por canónico normalizado y fusionar por bucket
    buckets = initial_bucket_by_canonical(raw_concepts)
    fused_by_canon: List[Dict[str, Any]] = [fuse_bucket(b) for b in buckets]

    # PASO 2: fusión entre buckets por similitud de términos + (opcional) LLM
    merged = cross_bucket_merge(fused_by_canon, jaccard_thr=args.jaccard, use_llm=args.use_llm, verbose=args.verbose)

    # Recalcular métricas por concepto (por si algo quedó sin metricas_mx)
    for c in merged:
        c["metricas_mx"] = recompute_metricas_mx(c)

    # Metadatos actualizados
    metadata = recompute_metadata(merged, base_meta)

    result = {
        "metadata": metadata,
        "conceptos": merged
    }

    if args.dry_run or not args.output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Escrito: {args.output}  (conceptos={len(merged)})")

if __name__ == "__main__":
    main()
