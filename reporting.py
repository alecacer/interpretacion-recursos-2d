"""
reporting.py
------------
Reportes de recursos:
  - por categoría dentro de un escenario;
  - totales por escenario;
  - estadísticas entre escenarios (P5 / P50 / P95 / I90);
  - comparación contra la realidad (IoU, falsos positivos/negativos);
  - comentarios automáticos simples para guiar la discusión.
"""

import numpy as np
import pandas as pd

from classification import CATEGORIES


def block_tonnage(block_size, espesor, densidad):
    """Toneladas de un bloque: área * espesor * densidad.
    Con los valores por defecto: 20*20*20*2.6 = 20 800 t."""
    return block_size * block_size * espesor * densidad


def report_resources_by_category(scenario_name, mask, categories, est,
                                 block_size, espesor, densidad,
                                 cutoff_grade=0.0):
    """Tabla de recursos por categoría para un escenario.

    mask         : bloques dentro de la interpretación mineralizada
    categories   : categoría por bloque (todos los bloques)
    est          : ley estimada por bloque (Cu%)
    cutoff_grade : ley de corte (%CuT) — sólo se reportan bloques con
                   ley estimada >= corte
    """
    ton_b = block_tonnage(block_size, espesor, densidad)
    area_b = block_size * block_size
    # Aplicar la ley de corte sobre la estimación
    mask = mask & np.isfinite(est) & (est >= cutoff_grade)
    rows = []
    for cat in CATEGORIES + ["Total recurso"]:
        if cat == "Total recurso":
            sel = mask
        else:
            sel = mask & (categories == cat)
        n = int(sel.sum())
        if n == 0:
            rows.append({"Escenario": scenario_name, "Categoría": cat,
                         "N° bloques": 0, "Área m²": 0.0, "Toneladas": 0.0,
                         "Ley media Cu%": np.nan, "Metal Cu (t)": 0.0,
                         "Ley mín": np.nan, "Ley máx": np.nan})
            continue
        # Las leyes pueden tener NaN si el dominio quedó sin muestras
        # (p.ej. un polígono que no encierra ningún sondaje)
        leyes = est[sel]
        finitas = np.isfinite(leyes)
        ton = n * ton_b
        # metal contenido: toneladas * ley(%) / 100, sumado por bloque
        metal = float(np.nansum(ton_b * leyes / 100.0))
        rows.append({
            "Escenario": scenario_name, "Categoría": cat,
            "N° bloques": n,
            "Área m²": n * area_b,
            "Toneladas": ton,
            "Ley media Cu%": (float(np.nanmean(leyes)) if finitas.any()
                              else np.nan),
            "Metal Cu (t)": metal,
            "Ley mín": (float(np.nanmin(leyes)) if finitas.any()
                        else np.nan),
            "Ley máx": (float(np.nanmax(leyes)) if finitas.any()
                        else np.nan),
        })
    return pd.DataFrame(rows)


def report_total_by_scenario(scenarios, block_size, espesor, densidad,
                             cutoff_grade=0.0):
    """Tabla resumen con una fila por escenario guardado.

    Cada escenario usa SU PROPIA estimación (sc["est_result"]): la
    estimación cambia entre escenarios porque cambia el dominio.
    Los escenarios sin estimación se omiten. Sólo se contabilizan los
    bloques con ley estimada >= cutoff_grade (%CuT).

    Unidades: tonelaje en Mt (millones de t), metal en kt (miles de t).
    """
    ton_b = block_tonnage(block_size, espesor, densidad)
    rows = []
    for sc in scenarios:
        if sc.get("est_result") is None:
            continue   # aún no estimado
        est = sc["est_result"]["est"]
        # Interpretación + ley de corte sobre la estimación
        mask = sc["mask"] & np.isfinite(est) & (est >= cutoff_grade)
        n = int(mask.sum())
        if n == 0:
            rows.append({"Escenario": sc["name"], "Ton (Mt)": 0.0,
                         "Ley media Cu%": np.nan, "Metal Cu (kt)": 0.0})
            continue
        leyes = est[mask]
        metal = float(np.nansum(ton_b * leyes / 100.0))
        rows.append({
            "Escenario": sc["name"],
            "Ton (Mt)": n * ton_b / 1e6,
            "Ley media Cu%": float(np.nanmean(leyes)),
            "Metal Cu (kt)": metal / 1e3,
        })
    return pd.DataFrame(rows)


def scenario_percentiles(totals):
    """Estadísticas entre escenarios para tonelaje (Mt), ley y metal (kt).

    I90_abs = P95 - P5
    I90_rel = 50 * (P95 - P5) / P50   [%]
    """
    rows = []
    variables = [("Ton (Mt)", "Ton (Mt)"),
                 ("Ley Cu %", "Ley media Cu%"),
                 ("Metal Cu (kt)", "Metal Cu (kt)")]
    for label, col in variables:
        vals = totals[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        i90 = p95 - p5
        i90rel = 50.0 * i90 / p50 if p50 > 0 else np.nan
        rows.append({"Variable": label, "P5": p5, "P50": p50, "P95": p95,
                     "I90 abs": i90, "I90 rel %": i90rel})
    return pd.DataFrame(rows)


def compare_tonnage_with_truth(scenarios, cu_true, block_size, espesor,
                               densidad, cutoff_grade=0.2):
    """Tabla escenario vs realidad sobre una ley de corte (%CuT):

    Por escenario: Ton / Ley / Metal estimados (bloques dentro de la
    interpretación con ley estimada >= corte) versus Ton / Ley / Metal
    REALES (bloques con ley verdadera >= corte), y la diferencia
    porcentual de cada variable: 100*(estimado - real)/real.
    """
    ton_b = block_tonnage(block_size, espesor, densidad)

    # Recurso REAL sobre la ley de corte (igual para todos los escenarios)
    sel_r = cu_true >= cutoff_grade
    ton_r = sel_r.sum() * ton_b
    ley_r = float(cu_true[sel_r].mean()) if sel_r.any() else np.nan
    met_r = float(np.sum(ton_b * cu_true[sel_r] / 100.0))

    rows = []
    for sc in scenarios:
        if sc.get("est_result") is None:
            continue
        est = sc["est_result"]["est"]
        sel = sc["mask"] & np.isfinite(est) & (est >= cutoff_grade)
        ton = sel.sum() * ton_b
        ley = float(est[sel].mean()) if sel.any() else np.nan
        met = float(np.sum(ton_b * est[sel] / 100.0))
        rows.append({
            "Escenario": sc["name"],
            "Ton (Mt)": ton / 1e6,
            "Ley Cu%": ley,
            "Metal Cu (t)": met,
            "Ton real (Mt)": ton_r / 1e6,
            "Ley real Cu%": ley_r,
            "Metal real (t)": met_r,
            "Δ% Ton": 100.0 * (ton - ton_r) / ton_r if ton_r else np.nan,
            "Δ% Ley": 100.0 * (ley - ley_r) / ley_r if ley_r else np.nan,
            "Δ% Metal": 100.0 * (met - met_r) / met_r if met_r else np.nan,
        })
    return pd.DataFrame(rows)


def compare_with_truth(scenarios, truth_mask, block_size):
    """Métricas de acierto interpretativo de cada escenario vs la realidad,
    calculadas sobre la grilla de bloques.

    IoU = área intersección / área unión (1 = interpretación perfecta).
    """
    area_b = block_size * block_size
    rows = []
    for sc in scenarios:
        m = sc["mask"]
        inter = (m & truth_mask).sum()
        union = (m | truth_mask).sum()
        fp = (m & ~truth_mask).sum()   # interpretado pero estéril real
        fn = (~m & truth_mask).sum()   # mineral real no interpretado
        rows.append({
            "Escenario": sc["name"],
            "Área interpretada m²": m.sum() * area_b,
            "Área verdadera m²": truth_mask.sum() * area_b,
            "Área intersección m²": inter * area_b,
            "Área unión m²": union * area_b,
            "IoU": inter / union if union > 0 else 0.0,
            "Falso positivo m²": fp * area_b,
            "Falso negativo m²": fn * area_b,
        })
    return pd.DataFrame(rows)
