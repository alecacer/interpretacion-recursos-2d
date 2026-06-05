"""
geometry.py
-----------
Utilidades geométricas:
  - conversión entre coordenadas del canvas (píxeles) y coordenadas
    reales del dominio (metros);
  - parseo de los objetos dibujados en streamlit-drawable-canvas;
  - validación de polígonos con Shapely;
  - determinación de qué bloques caen dentro de una interpretación.
"""

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union


# ------------------------------------------------------------------
# Conversión canvas <-> mundo real
# ------------------------------------------------------------------
def canvas_to_world(px, py, canvas_px, domain):
    """Convierte un punto del canvas (origen arriba-izquierda, Y hacia
    abajo) a coordenadas reales (origen abajo-izquierda, Y hacia arriba)."""
    x = px / canvas_px * domain
    y = (1.0 - py / canvas_px) * domain
    return x, y


def world_to_canvas(x, y, canvas_px, domain):
    """Conversión inversa: mundo real -> píxeles del canvas."""
    px = x / domain * canvas_px
    py = (1.0 - y / domain) * canvas_px
    return px, py


# ------------------------------------------------------------------
# Parseo de objetos del canvas (fabric.js)
# ------------------------------------------------------------------
def _extract_points(obj):
    """Extrae la lista de vértices (px, py) de un objeto fabric.js.

    En modo 'polygon' streamlit-drawable-canvas crea objetos tipo 'path'
    cuyos comandos son ['M', x, y], ['L', x, y], ..., ['z'].
    Se cubren también los tipos 'polygon'/'polyline' por robustez.
    """
    t = obj.get("type", "")
    if t == "path":
        return [(c[1], c[2]) for c in obj.get("path", []) if len(c) >= 3]
    if t in ("polygon", "polyline"):
        pts = obj.get("points", [])
        left = obj.get("left", 0.0)
        top = obj.get("top", 0.0)
        po = obj.get("pathOffset", {"x": 0.0, "y": 0.0})
        return [(p["x"] + left - po.get("x", 0.0),
                 p["y"] + top - po.get("y", 0.0)) for p in pts]
    return []


def parse_canvas_polygons(json_data, canvas_px, domain):
    """Convierte los objetos dibujados en el canvas a polígonos en
    coordenadas reales.

    El tipo de polígono se infiere del color del trazo:
      rojo  -> mineralizado
      azul  -> estéril / baja ley

    Devuelve lista de dicts: {"tipo": str, "coords": [(x, y), ...]}
    """
    out = []
    if not json_data:
        return out
    for obj in json_data.get("objects", []):
        pts = _extract_points(obj)
        if len(pts) < 3:
            continue  # no es un polígono cerrado válido
        stroke = str(obj.get("stroke", "#ff0000")).lower()
        tipo = "mineral" if stroke.startswith("#ff") or "255, 0, 0" in stroke \
            else "esteril"
        coords = [canvas_to_world(px, py, canvas_px, domain) for px, py in pts]
        out.append({"tipo": tipo, "coords": coords})
    return out


# ------------------------------------------------------------------
# Validación y operaciones con polígonos
# ------------------------------------------------------------------
def validate_polygon(coords, min_area=100.0):
    """Valida y repara un polígono a partir de sus vértices.

    - requiere al menos 3 vértices;
    - cierra el polígono automáticamente (Shapely lo hace);
    - repara auto-intersecciones con buffer(0);
    - descarta polígonos con área menor a `min_area` m² (ruido de clic).

    Devuelve un shapely Polygon válido o None.
    """
    if coords is None or len(coords) < 3:
        return None
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)   # reparación estándar
        if poly.is_empty or poly.area < min_area:
            return None
        return poly
    except Exception:
        return None


def polygon_area(coords):
    """Área (m²) de un polígono definido por sus vértices."""
    poly = validate_polygon(coords)
    return poly.area if poly is not None else 0.0


def interpretation_geometry(polygons):
    """Geometría Shapely de la interpretación mineralizada de un escenario:
    unión de los polígonos 'mineral' menos la unión de los 'esteril'.

    Devuelve la geometría (o None si no hay polígonos minerales válidos).
    """
    minerales = [validate_polygon(p["coords"]) for p in polygons
                 if p["tipo"] == "mineral"]
    minerales = [p for p in minerales if p is not None]
    esteriles = [validate_polygon(p["coords"]) for p in polygons
                 if p["tipo"] == "esteril"]
    esteriles = [p for p in esteriles if p is not None]

    if not minerales:
        return None
    geom = unary_union(minerales)
    if esteriles:
        geom = geom.difference(unary_union(esteriles))
    return None if geom.is_empty else geom


def scenario_mask(polygons, blocks):
    """Determina qué bloques quedan DENTRO de la interpretación mineralizada
    de un escenario.

    polygons : lista de dicts {"tipo", "coords"}.
               Los polígonos 'esteril' se restan de los 'mineral'.
    blocks   : DataFrame con centroides x, y.

    Devuelve (mask booleano de largo N, área mineralizada del polígono m²).
    """
    geom = interpretation_geometry(polygons)
    if geom is None:
        return np.zeros(len(blocks), dtype=bool), 0.0
    # Un bloque pertenece a la interpretación si su centroide está dentro
    mask = shapely.contains_xy(geom, blocks["x"].to_numpy(),
                               blocks["y"].to_numpy())
    return mask, geom.area


def samples_inside(polygons, samples):
    """Determina qué MUESTRAS quedan dentro de la interpretación
    mineralizada (misma geometría que los bloques). Se usa para definir
    los dominios de estimación: dentro se estima con estas muestras,
    fuera con el resto."""
    geom = interpretation_geometry(polygons)
    if geom is None:
        return np.zeros(len(samples), dtype=bool)
    return shapely.contains_xy(geom, samples["X"].to_numpy(),
                               samples["Y"].to_numpy())


def blocks_inside_polygons(polygons, blocks):
    """Índices de los bloques dentro de la interpretación mineralizada."""
    mask, _ = scenario_mask(polygons, blocks)
    return np.where(mask)[0]
