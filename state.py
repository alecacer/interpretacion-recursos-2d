"""
state.py
--------
Manejo del estado de la sesión Streamlit: valores por defecto, creación
del caso sintético y administración de la lista de escenarios.
"""

from datetime import datetime

import streamlit as st

import synthetic_case

# Valores por defecto de todos los controles (especificación)
DEFAULTS = {
    "seed": 12345,        # semilla aleatoria
    "spacing": 85.0,      # espaciamiento nominal de sondajes (70-100 m)
    "jitter": 8.0,        # perturbación aleatoria (5-10 m)
    "cutoff": 0.30,       # cutoff visual mineral/estéril en sondajes (% Cu)
    "ley_corte": 0.20,    # ley de corte para reportes y comparaciones (%CuT)
    "espesor": 20.0,      # espesor constante (m)
    "densidad": 2.6,      # densidad (t/m³)
    "azimuth": 37.0,      # azimut del variograma / pista regional
    # Alcances del variograma de KRIGING: calibrados al alcance práctico
    # EFECTIVO del campo simulado (el núcleo gaussiano con sigma=rango/2
    # produce ~1.7x el rango nominal 50/25 -> ~90/45 m medidos en el
    # variograma experimental del campo verdadero)
    "r_major": 90.0,      # alcance mayor de ley (m)
    "r_minor": 45.0,      # alcance menor de ley (m)
    "nugget": 0.0,        # sin efecto pepita (fijo, no editable)
    "sill": 1.0,          # meseta
    "vmodel": "esferico",
    "d_med": 50.0,        # distancia para Medido
    "d_ind": 100.0,       # distancia para Indicado
    "d_inf": 150.0,       # distancia para Inferido
    "meta_escenarios": 5,
}


def init_state():
    """Inicializa session_state la primera vez que corre la app."""
    ss = st.session_state
    if "initialized" in ss:
        return
    ss.initialized = True
    ss.scenarios = []            # lista de dicts de escenarios guardados
    ss.active_scenario = None    # id del escenario activo (o None)
    ss.pending_active = None     # mueve el slider tras guardar/regenerar
    ss.pending_view = None       # navegación programática entre vistas
    # Flujo guiado por etapas (secuencial, se activa solo):
    # interpretar 5 escenarios -> estimar -> categorizar ->
    # incertidumbre -> develar
    ss.categorized = False          # Categorización ya ejecutada
    ss.canvas_version = 0        # se incrementa para limpiar el canvas
    ss.p_mineral = None          # probabilidad de mineralización
    ss.revealed = False          # ¿se develó la realidad?
    regenerate_case(DEFAULTS["seed"], DEFAULTS["spacing"], DEFAULTS["jitter"])


def regenerate_case(seed, spacing, jitter):
    """(Re)genera el caso sintético y limpia todo lo derivado de él
    (escenarios, estimación, probabilidad, realidad develada)."""
    ss = st.session_state
    ss.case = synthetic_case.make_case(seed=int(seed), spacing=float(spacing),
                                       jitter=float(jitter))
    ss.scenarios = []
    ss.active_scenario = None
    ss.p_mineral = None
    ss.revealed = False
    ss.canvas_version += 1


def new_scenario_dict(scenario_id, name, polygons, mask, sample_mask,
                      area_poly, block_size, espesor, densidad):
    """Crea el dict de un escenario con la estructura de la especificación
    (sección 4). Los campos de resultados se completan al estimar.

    mask        : bloques dentro de la interpretación mineralizada
    sample_mask : muestras dentro de la interpretación (define los
                  dominios de estimación dentro/fuera)
    """
    return {
        "scenario_id": scenario_id,
        "name": name,
        "polygons": polygons,            # [{"tipo", "coords"}, ...]
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mask": mask,
        "sample_mask": sample_mask,
        "area_mineralizada": float(mask.sum()) * block_size * block_size,
        "area_poligono": float(area_poly),
        "est_result": None,              # estimación propia del escenario
        "toneladas": None,
        "ley_media": None,
        "metal_contenido": None,
        "resource_table": None,
        "estimated": False,
        "classified": False,
    }


def get_scenario(scenario_id):
    """Busca un escenario guardado por su id (None si no existe)."""
    for sc in st.session_state.scenarios:
        if sc["scenario_id"] == scenario_id:
            return sc
    return None


def delete_scenario(scenario_id):
    """Elimina un escenario guardado."""
    ss = st.session_state
    ss.scenarios = [sc for sc in ss.scenarios
                    if sc["scenario_id"] != scenario_id]
    if ss.active_scenario == scenario_id:
        ss.active_scenario = ss.scenarios[-1]["scenario_id"] \
            if ss.scenarios else None
    ss.p_mineral = None   # la probabilidad queda obsoleta
