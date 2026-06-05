"""
smoke_test.py
-------------
Prueba rápida (sin interfaz) de todo el pipeline computacional:
caso sintético -> escenarios simulados -> estimación POR DOMINIOS
(dentro/fuera de cada interpretación) -> categorización -> reportes ->
percentiles -> p(mineral) -> comparación con la realidad.

Ejecución:  python smoke_test.py
"""

import sys

import numpy as np

# La consola Windows usa cp1252: forzar UTF-8 para imprimir símbolos
sys.stdout.reconfigure(encoding="utf-8")

import geometry
import reporting
import uncertainty
from classification import classify_resources
from estimation import estimate_scenario, get_scenario_weights_for_block
from synthetic_case import make_case

# 1. Caso sintético (fijo)
case = make_case(seed=12345, spacing=85.0, jitter=8.0)
blocks, samples = case["blocks"], case["samples"]
assert len(blocks) == 2500, "deben ser 2500 bloques"
print(f"[1] Caso: {len(blocks)} bloques, {len(samples)} sondajes, "
      f"Cu muestras {samples['Cu_pct'].min():.3f}-"
      f"{samples['Cu_pct'].max():.3f}%")
n_min = int(case["truth_mask"].sum())
print(f"    Geología verdadera: {n_min} bloques mineralizados "
      f"({100*n_min/2500:.1f}% del dominio)")
assert 100 < n_min < 1200, "proporción mineralizada fuera de rango razonable"
cu_in = case["cu_true"][case["truth_mask"]]
cu_out = case["cu_true"][~case["truth_mask"]]
print(f"    Cu dentro: {cu_in.min():.2f}-{cu_in.max():.2f} "
      f"(media {cu_in.mean():.2f}) | fuera: {cu_out.min():.3f}-"
      f"{cu_out.max():.3f} (media {cu_out.mean():.3f})")

# 2. Escenarios simulados: franjas inclinadas distintas
scenarios = []
for k in range(3):
    w = 120 + 40 * k
    coords = [(150, 100 + 30 * k), (900, 700 + 30 * k),
              (900 - w, 700 + w + 30 * k), (150, 100 + w + 30 * k)]
    polys = [{"tipo": "mineral", "coords": coords}]
    mask, area = geometry.scenario_mask(polys, blocks)
    smask = geometry.samples_inside(polys, samples)
    assert mask.sum() > 0 and smask.sum() > 0
    scenarios.append({"scenario_id": k + 1, "name": f"Esc {k+1}",
                      "polygons": polys, "mask": mask,
                      "sample_mask": smask})
print(f"[2] Escenarios: bloques = "
      f"{[int(s['mask'].sum()) for s in scenarios]}, sondajes dentro = "
      f"{[int(s['sample_mask'].sum()) for s in scenarios]}")

# 3. Estimación POR DOMINIOS de cada escenario (kriging ordinario)
params = {"azimuth": 37.0, "r_major": 90.0, "r_minor": 45.0,
          "nugget": 0.0, "sill": 1.0, "model": "esferico"}
for sc in scenarios:
    sc["est_result"] = estimate_scenario("Kriging ordinario", samples,
                                         blocks, sc["mask"],
                                         sc["sample_mask"], params)
    r = sc["est_result"]
    assert r["est"].shape == (2500,)
    # los pesos de cada dominio deben sumar 1 (kriging ordinario)
    for name, dom in r["domains"].items():
        if dom["weights"] is not None:
            assert np.allclose(dom["weights"].sum(axis=0), 1.0, atol=1e-6), \
                f"pesos no suman 1 en dominio {name}"
    ley_in = np.nanmean(r["est"][sc["mask"]])
    ley_out = np.nanmean(r["est"][~sc["mask"]])
    print(f"[3] {sc['name']}: ley dentro {ley_in:.3f}% "
          f"(con {r['domains']['dentro']['n_samples']} muestras) | "
          f"fuera {ley_out:.3f}% "
          f"(con {r['domains']['fuera']['n_samples']} muestras)")

# La estimación DEBE cambiar entre escenarios (cambia la unidad)
e1 = scenarios[0]["est_result"]["est"]
e2 = scenarios[1]["est_result"]["est"]
dif = np.nanmax(np.abs(e1 - e2))
assert dif > 0.01, "la estimación debería diferir entre escenarios"
print(f"    Diferencia máxima de ley entre Esc 1 y Esc 2: {dif:.3f}% "
      f"(la estimación cambia con la interpretación) ✔")

# La ley media dentro debe ser mayor que fuera (dominio mineralizado)
assert np.nanmean(e1[scenarios[0]['mask']]) > \
       np.nanmean(e1[~scenarios[0]['mask']])

# 3b. Inspector de pesos: bloque interior usa sólo muestras interiores
bid = int(np.where(scenarios[0]["mask"])[0][len(np.where(scenarios[0]["mask"])[0]) // 2])
info = get_scenario_weights_for_block(bid, samples, blocks,
                                      scenarios[0]["est_result"])
assert info["domain"] == "dentro"
assert len(info["samples_used"]) == scenarios[0]["sample_mask"].sum()
print(f"    Pesos bloque interior ({info['x']:.0f},{info['y']:.0f}): "
      f"dominio '{info['domain']}', {len(info['samples_used'])} muestras, "
      f"Cu*={info['estimated_grade']:.3f}")

# escenario sin sondajes dentro -> dominio interior NaN
polys_vacio = [{"tipo": "mineral",
                "coords": [(20, 20), (60, 20), (60, 60), (20, 60)]}]
m_v, _ = geometry.scenario_mask(polys_vacio, blocks)
s_v = geometry.samples_inside(polys_vacio, samples)
if s_v.sum() == 0 and m_v.sum() > 0:
    r_v = estimate_scenario("Kriging ordinario", samples, blocks, m_v,
                            s_v, params)
    assert np.isnan(r_v["est"][m_v]).all()
    print("    Polígono sin sondajes: dominio interior queda NaN ✔")

# 4. Categorización
cat = classify_resources(case["dist"], scenarios[0]["mask"], 50, 100, 150)
vals, cnts = np.unique(cat, return_counts=True)
print(f"[4] Categorías: {dict(zip(vals, cnts))}")

# 5. Reportes (cada escenario con su propia estimación)
rep = reporting.report_resources_by_category(
    "Esc 1", scenarios[0]["mask"], cat, e1, 20.0, 20.0, 2.6)
tot = rep[rep["Categoría"] == "Total recurso"].iloc[0]
print(f"[5] Reporte Esc 1: {tot['Toneladas']:,.0f} t @ "
      f"{tot['Ley media Cu%']:.3f}% Cu = {tot['Metal Cu (t)']:,.0f} t Cu")

totals = reporting.report_total_by_scenario(scenarios, 20.0, 20.0, 2.6,
                                            cutoff_grade=0.2)
assert len(totals) == 3
assert list(totals.columns) == ["Escenario", "Ton (Mt)",
                                "Ley media Cu%", "Metal Cu (kt)"]
perc = reporting.scenario_percentiles(totals)
print(f"[6] Percentiles entre escenarios (corte 0.2% CuT):\n"
      f"{perc.round(2).to_string(index=False)}")

# 6. Incertidumbre
p = uncertainty.compute_p_mineral(scenarios, 2500)
p1p = uncertainty.compute_p_one_minus_p(p)
assert p.min() >= 0 and p.max() <= 1 and p1p.max() <= 0.25 + 1e-9
print(f"[7] p(mineral): max {p.max():.2f} | p(1-p): max {p1p.max():.3f}")

# 7. Comparación con la realidad
comp = reporting.compare_with_truth(scenarios, case["truth_mask"], 20.0)
print(f"[8] IoU vs realidad: {comp['IoU'].round(3).tolist()}")

# 8. Conversión canvas <-> mundo (ida y vuelta exacta)
x, y = geometry.canvas_to_world(320, 160, 640, 1000)
px, py = geometry.world_to_canvas(x, y, 640, 1000)
assert abs(px - 320) < 1e-9 and abs(py - 160) < 1e-9
print(f"[9] Canvas<->mundo: (320,160)px -> ({x:.0f},{y:.0f})m -> OK")

# 9. Parseo de objetos fabric.js tipo 'path' (como los del canvas)
fake_json = {"objects": [{
    "type": "path", "stroke": "#FF0000",
    "path": [["M", 100, 100], ["L", 300, 100], ["L", 300, 300], ["z"]]}]}
polys = geometry.parse_canvas_polygons(fake_json, 640, 1000)
assert len(polys) == 1 and polys[0]["tipo"] == "mineral"
assert geometry.validate_polygon(polys[0]["coords"]) is not None
print(f"[10] Parseo canvas: 1 polígono mineral válido")

print("\n*** SMOKE TEST COMPLETO: todo OK ***")
