"""
estimation.py
-------------
Estimación POR ESCENARIO con dominios controlados por la interpretación
(frontera dura / hard boundary):

  - Dominio "dentro":  los bloques dentro del polígono interpretado se
    estiman SOLO con las muestras que caen dentro de la interpretación.
  - Dominio "fuera":   los bloques fuera del polígono se estiman SOLO con
    el resto de las muestras.

Por eso la estimación cambia entre escenarios aunque los datos sean los
mismos: cambia la unidad (el dominio de estimación). Éste es el punto
pedagógico central: la interpretación geológica controla qué muestras se
mezclan con qué bloques.

Métodos disponibles: kriging ordinario (por defecto) e IDW (respaldo).
"""

import numpy as np
import pandas as pd

import kriging


def idw_estimate(samples, blocks, power=2.0):
    """Inverso de la distancia (IDW) como método de respaldo.

    Devuelve el mismo dict que el kriging (con var=None) para que el
    inspector de pesos también funcione con IDW.
    """
    sx = samples["X"].to_numpy()
    sy = samples["Y"].to_numpy()
    z = samples["Cu_pct"].to_numpy()
    bx = blocks["x"].to_numpy()
    by = blocks["y"].to_numpy()

    dx = sx[:, None] - bx[None, :]
    dy = sy[:, None] - by[None, :]
    d = np.sqrt(dx ** 2 + dy ** 2)
    d = np.maximum(d, 1e-6)            # evitar división por cero

    w = 1.0 / d ** power
    w = w / w.sum(axis=0, keepdims=True)   # pesos normalizados (suman 1)
    est = w.T @ z

    return {"est": est, "var": None, "weights": w, "dists": d,
            "method": "IDW", "params": {"power": power}}


def run_estimation(method, samples, blocks, params):
    """Estimación en un único dominio (subconjunto de muestras y bloques)."""
    if method == "IDW":
        return idw_estimate(samples, blocks)
    return kriging.estimate_blocks_ok(samples, blocks, params)


def estimate_scenario(method, samples, blocks, block_mask, sample_mask,
                      params):
    """Estimación completa de UN escenario, dominio por dominio.

    block_mask  : bloques dentro de la interpretación mineralizada
    sample_mask : muestras dentro de la interpretación mineralizada

    Devuelve dict:
        est     : ley estimada por bloque (NaN si el dominio del bloque
                  no tiene muestras)
        var     : varianza de kriging por bloque (NaN donde no aplica)
        domains : por dominio ('dentro' / 'fuera'):
                    sample_idx, block_idx, weights, dists, n_samples
        method, params
    """
    n_blocks = len(blocks)
    est = np.full(n_blocks, np.nan)
    var = np.full(n_blocks, np.nan)
    domains = {}

    for name, smask, bmask in [("dentro", sample_mask, block_mask),
                               ("fuera", ~sample_mask, ~block_mask)]:
        sidx = np.where(smask)[0]
        bidx = np.where(bmask)[0]
        if len(bidx) == 0:
            continue   # el dominio no tiene bloques (p.ej. polígono vacío)
        if len(sidx) == 0:
            # Dominio sin muestras: no se puede estimar -> queda NaN.
            # (Ocurre si el polígono no encierra ningún sondaje.)
            domains[name] = {"sample_idx": sidx, "block_idx": bidx,
                             "weights": None, "dists": None, "n_samples": 0}
            continue

        sub_s = samples.iloc[sidx].reset_index(drop=True)
        sub_b = blocks.iloc[bidx].reset_index(drop=True)
        r = run_estimation(method, sub_s, sub_b, params)
        est[bidx] = r["est"]
        if r["var"] is not None:
            var[bidx] = r["var"]
        domains[name] = {"sample_idx": sidx, "block_idx": bidx,
                         "weights": r["weights"], "dists": r["dists"],
                         "n_samples": len(sidx)}

    return {"est": est, "var": var, "domains": domains,
            "method": method, "params": dict(params)}


def get_scenario_weights_for_block(block_id, samples, blocks, scen_result):
    """Detalle de la estimación de un bloque dentro de un escenario:
    dominio al que pertenece, muestras DE ESE DOMINIO, distancias y pesos
    (ordenados por |peso| descendente).

    Devuelve None si el bloque no pertenece a ningún dominio estimado.
    """
    for name, dom in scen_result["domains"].items():
        pos = np.where(dom["block_idx"] == block_id)[0]
        if pos.size == 0:
            continue
        base = {
            "block_id": int(block_id),
            "domain": name,
            "x": float(blocks.loc[block_id, "x"]),
            "y": float(blocks.loc[block_id, "y"]),
            "estimated_grade": float(scen_result["est"][block_id]),
        }
        if dom["weights"] is None:
            # Dominio sin muestras: no hay pesos que mostrar
            base["samples_used"] = pd.DataFrame()
            base["weights"] = np.array([])
            base["distances"] = np.array([])
            return base

        j = pos[0]
        w = dom["weights"][:, j]
        d = dom["dists"][:, j]
        sub = samples.iloc[dom["sample_idx"]].reset_index(drop=True)
        tabla = pd.DataFrame({
            "ID": sub["ID"].to_numpy(),
            "X": sub["X"].round(1).to_numpy(),
            "Y": sub["Y"].round(1).to_numpy(),
            "Cu_pct": sub["Cu_pct"].round(3).to_numpy(),
            "distancia_m": d.round(1),
            "peso": w.round(4),
        }).sort_values("peso", key=np.abs,
                       ascending=False).reset_index(drop=True)
        base["samples_used"] = tabla
        base["weights"] = w
        base["distances"] = d
        return base
    return None
