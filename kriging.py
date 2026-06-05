"""
kriging.py
----------
Implementación propia y simple de kriging ordinario 2D (Opción B de la
especificación). Se eligió implementación propia porque permite guardar
y mostrar fácilmente los pesos de kriging — objetivo docente clave.

La anisotropía se maneja con transformación de coordenadas:
  1. rotar según el azimut (medido desde el norte, horario);
  2. escalar la coordenada perpendicular por (rango_mayor / rango_menor);
  3. usar un variograma isotrópico con alcance = rango_mayor.
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Distancia anisotrópica y modelos de variograma
# ------------------------------------------------------------------
def rotate_and_scale_coordinates(dx, dy, azimuth_deg, r_major, r_minor):
    """Distancia efectiva anisotrópica entre pares de puntos.

    dx, dy : diferencias de coordenadas (arrays compatibles)
    Devuelve la distancia equivalente isotrópica respecto del rango mayor.
    """
    az = np.radians(azimuth_deg)
    # u: componente a lo largo del azimut (eje mayor); v: perpendicular
    u = dx * np.sin(az) + dy * np.cos(az)
    v = dx * np.cos(az) - dy * np.sin(az)
    return np.sqrt(u ** 2 + (v * (r_major / r_minor)) ** 2)


def variogram_model(h, model, nugget, sill, a):
    """Evalúa el semivariograma γ(h).

    model  : 'esferico' o 'exponencial'
    nugget : efecto pepita
    sill   : meseta total (nugget + contribución estructurada)
    a      : alcance (rango mayor, en el espacio transformado)
    """
    c = max(sill - nugget, 1e-12)   # contribución de la estructura
    h = np.asarray(h, dtype=float)
    if model == "exponencial":
        g = nugget + c * (1.0 - np.exp(-3.0 * h / a))
    else:  # esférico (por defecto)
        hr = np.clip(h / a, 0.0, 1.0)
        g = nugget + c * (1.5 * hr - 0.5 * hr ** 3)
        g = np.where(h >= a, sill, g)
    # Por definición γ(0) = 0 (el nugget actúa sólo para h > 0)
    return np.where(h <= 1e-9, 0.0, g)


# ------------------------------------------------------------------
# Sistema de kriging ordinario
# ------------------------------------------------------------------
def build_kriging_matrix(sx, sy, params):
    """Construye la matriz del sistema de kriging ordinario (n+1, n+1):

        [ γ(xi,xj)  1 ] [ λ ]   [ γ(xi,x0) ]
        [   1ᵀ      0 ] [ μ ] = [    1     ]
    """
    n = len(sx)
    dx = sx[:, None] - sx[None, :]
    dy = sy[:, None] - sy[None, :]
    h = rotate_and_scale_coordinates(dx, dy, params["azimuth"],
                                     params["r_major"], params["r_minor"])
    G = variogram_model(h, params["model"], params["nugget"],
                        params["sill"], params["r_major"])
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[n, :n] = 1.0   # condición de insesgo: suma de pesos = 1
    A[:n, n] = 1.0
    return A


def solve_ordinary_kriging(A, B):
    """Resuelve el sistema de kriging para todos los bloques a la vez.

    A : (n+1, n+1) matriz del sistema
    B : (n+1, m)   lados derechos (uno por bloque)
    Devuelve lambdas (n, m) y multiplicadores de Lagrange (m,).
    """
    try:
        sol = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        # Respaldo numérico si la matriz resulta singular
        sol = np.linalg.lstsq(A, B, rcond=None)[0]
    return sol[:-1, :], sol[-1, :]


def estimate_blocks_ok(samples, blocks, params):
    """Kriging ordinario en los centroides de todos los bloques.

    samples : DataFrame con X, Y, Cu_pct
    blocks  : DataFrame con x, y
    params  : dict con azimuth, r_major, r_minor, nugget, sill, model

    Devuelve dict:
        est      : ley estimada por bloque (m,)
        var      : varianza de kriging por bloque (m,)
        weights  : matriz de pesos (n_muestras, m_bloques)
        dists    : distancias euclidianas muestra-bloque (n, m)
    """
    sx = samples["X"].to_numpy()
    sy = samples["Y"].to_numpy()
    z = samples["Cu_pct"].to_numpy()
    bx = blocks["x"].to_numpy()
    by = blocks["y"].to_numpy()
    n, m = len(sx), len(bx)

    A = build_kriging_matrix(sx, sy, params)

    # Lado derecho: γ entre cada muestra y cada bloque
    dx = sx[:, None] - bx[None, :]
    dy = sy[:, None] - by[None, :]
    h0 = rotate_and_scale_coordinates(dx, dy, params["azimuth"],
                                      params["r_major"], params["r_minor"])
    B = np.ones((n + 1, m))
    B[:n, :] = variogram_model(h0, params["model"], params["nugget"],
                               params["sill"], params["r_major"])

    lam, mu = solve_ordinary_kriging(A, B)

    est = lam.T @ z                                   # ley estimada
    var = np.einsum("ij,ij->j", lam, B[:n, :]) + mu   # varianza de kriging
    var = np.maximum(var, 0.0)

    # Distancias euclidianas reales (para el inspector de pesos)
    dists = np.sqrt(dx ** 2 + dy ** 2)

    return {"est": est, "var": var, "weights": lam, "dists": dists,
            "method": "Kriging ordinario", "params": dict(params)}


def get_kriging_weights_for_block(block_id, samples, blocks, result):
    """Detalle de la estimación de un bloque: muestras usadas, distancias
    y pesos de kriging, ordenados por |peso| descendente."""
    w = result["weights"][:, block_id]
    d = result["dists"][:, block_id]
    tabla = pd.DataFrame({
        "ID": samples["ID"].to_numpy(),
        "X": samples["X"].round(1).to_numpy(),
        "Y": samples["Y"].round(1).to_numpy(),
        "Cu_pct": samples["Cu_pct"].round(3).to_numpy(),
        "distancia_m": d.round(1),
        "peso": w.round(4),
    }).sort_values("peso", key=np.abs, ascending=False).reset_index(drop=True)

    return {
        "block_id": int(block_id),
        "x": float(blocks.loc[block_id, "x"]),
        "y": float(blocks.loc[block_id, "y"]),
        "estimated_grade": float(result["est"][block_id]),
        "samples_used": tabla,
        "weights": w,
        "distances": d,
    }
