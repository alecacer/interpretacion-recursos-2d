"""
professor.py
------------
Modo profesor: utilidades para PROBAR la aplicación sin dibujar a mano.

Se activa agregando `?profesor=1` a la URL de la app, p. ej.:
    http://localhost:8501/?profesor=1

Genera 5 interpretaciones plausibles y distintas a partir de las muestras
mineralizadas visibles (no usa la verdad oculta), imitando lo que harían
estudiantes con criterios diferentes:

    1. Envolvente ajustada (conservador)
    2. Envolvente amplia (optimista)
    3. Tres cuerpos separados (terciles a lo largo del rumbo 37°)
    4. Veta continua angosta (banda que sigue los puntos)
    5. Dos cuerpos (mitades a lo largo del rumbo)
"""

import numpy as np
from shapely.geometry import LineString, MultiPoint, box
from shapely.ops import unary_union

DOMAIN_BOX = box(0.0, 0.0, 1000.0, 1000.0)


def _to_poly_dicts(geom, min_area=2000.0):
    """Convierte una geometría Shapely (Polygon o MultiPolygon) a la lista
    de dicts {"tipo": "mineral", "coords": [...]} que usa la app."""
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    out = []
    for g in geoms:
        if g.is_empty or g.area < min_area:
            continue   # descartar astillas
        coords = [(float(x), float(y)) for x, y in g.exterior.coords[:-1]]
        out.append({"tipo": "mineral", "coords": coords})
    return out


def generate_demo_interpretations(case, cutoff=0.30):
    """Devuelve una lista de 5 interpretaciones (cada una es una lista de
    polígonos mineral) construidas SOLO con los sondajes mineralizados."""
    s = case["samples"]
    m = s[s["Cu_pct"] >= cutoff]
    pts = list(zip(m["X"].to_numpy(), m["Y"].to_numpy()))
    mp = MultiPoint(pts)

    # Proyección de cada punto a lo largo del rumbo regional (37°)
    az = np.radians(case.get("azimuth", 37.0))
    t = m["X"].to_numpy() * np.sin(az) + m["Y"].to_numpy() * np.cos(az)
    order = np.argsort(t)

    escenarios = []

    # 1. Conservador: envolvente convexa ajustada
    escenarios.append(mp.convex_hull.buffer(25))

    # 2. Optimista: envolvente amplia
    escenarios.append(mp.convex_hull.buffer(70))

    # 3. Tres cuerpos separados (terciles a lo largo del rumbo)
    q = np.quantile(t, [1 / 3, 2 / 3])
    parts = []
    for sel in [t <= q[0], (t > q[0]) & (t <= q[1]), t > q[1]]:
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        parts.append(MultiPoint([pts[i] for i in idx])
                     .convex_hull.buffer(40))
    escenarios.append(unary_union(parts))

    # 4. Veta continua angosta: banda que sigue los puntos ordenados
    escenarios.append(LineString([pts[i] for i in order]).buffer(35))

    # 5. Dos cuerpos (mitades a lo largo del rumbo)
    med = np.median(t)
    parts2 = []
    for sel in [t <= med, t > med]:
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        parts2.append(MultiPoint([pts[i] for i in idx])
                      .convex_hull.buffer(45))
    escenarios.append(unary_union(parts2))

    # Recortar al dominio y simplificar vértices
    out = []
    for g in escenarios:
        g = g.intersection(DOMAIN_BOX).simplify(5.0)
        out.append(_to_poly_dicts(g))
    return out
