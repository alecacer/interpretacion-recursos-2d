"""
app.py
------
Aplicación Streamlit docente: interpretación geológica, estimación,
categorización y análisis de incertidumbre en 2D.

Ejecución:
    streamlit run app.py

Flujo guiado del estudiante (etapas que se liberan en orden):
  1. dibujar 5 escenarios -> botón «Siguiente»
  2. estimar (kriging OK por dominios)  -> despliega la estimación
  3. categorizar -> despliega la categorización (Verde/Amarillo/Rojo)
  4. p(mineral) y p(1-p) -> pasa a Incertidumbre
  5. develar realidad.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import geometry
import plotting
import professor
import reporting
import state
import uncertainty
from classification import classify_resources
from estimation import estimate_scenario

# ------------------------------------------------------------------
# Shim de compatibilidad: streamlit-drawable-canvas 0.9.3 usa
# st_image.image_to_url (eliminada en Streamlit >= 1.41) para el fondo
# del lienzo. En vez de servir el PNG vía el media manager (frágil en
# despliegues en la nube: el lienzo aparece SIN los sondajes), se
# entrega la imagen incrustada como data-URI base64, que funciona
# igual en local y en Streamlit Community Cloud.
# ------------------------------------------------------------------
import base64
try:
    import streamlit.elements.image as _st_image

    def _image_to_data_url(image, *_args, **_kwargs):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return ("data:image/png;base64,"
                + base64.b64encode(buf.getvalue()).decode())

    _st_image.image_to_url = _image_to_data_url
except Exception:
    pass

# El canvas de dibujo es una dependencia externa: si falta, la app sigue
# funcionando (sin dibujo) y muestra cómo instalarla.
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

CANVAS_PX = 800   # tamaño del canvas en píxeles (igual al mapa principal)

st.set_page_config(page_title="Interpretación y recursos 2D — DIMIN USACH",
                   layout="wide", page_icon="⛏️")

# ------------------------------------------------------------------
# Identidad corporativa (Depto. de Ingeniería en Minas, USACH)
# Colores del logo de la plantilla PPTX:
#   naranjo #EE7700 | gris azulado #37404A | turquesa #00A79A
# ------------------------------------------------------------------
USACH_NARANJO = "#EE7700"
USACH_GRIS = "#37404A"
USACH_TURQUESA = "#00A79A"
LOGO_PATH = Path(__file__).parent / "assets" / "logo_dimin_usach.png"
if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")

# Acentos visuales: títulos en gris corporativo, etapas con borde
# naranjo y línea turquesa bajo el encabezado principal
st.markdown(f"""
<style>
h1, h2, h3 {{ color: {USACH_GRIS}; }}
h2 {{ border-bottom: 3px solid {USACH_TURQUESA}; padding-bottom: 4px; }}
[data-testid="stSidebar"] h4 {{
    color: {USACH_GRIS};
    border-left: 4px solid {USACH_NARANJO};
    padding-left: 8px;
}}
[data-testid="stSidebar"] hr {{ margin: 8px 0; }}
</style>
""", unsafe_allow_html=True)

state.init_state()
ss = st.session_state
D = state.DEFAULTS

# Supuestos FIJOS del caso (sin controles en la interfaz)
cutoff = D["cutoff"]        # 0.30 %Cu (clase visual de sondajes)
ley_corte = D["ley_corte"]  # 0.20 %CuT (reportes y comparaciones)
espesor = D["espesor"]      # 20 m
densidad = D["densidad"]    # 2.6 t/m³
ton_b = reporting.block_tonnage(ss.case["block_size"], espesor, densidad)


# ==================================================================
# CALLBACKS DEL FLUJO GUIADO
# Se ejecutan ANTES de renderizar la página, por lo que los bloqueos
# de etapa quedan actualizados en el mismo run (sin st.rerun()).
# ==================================================================
FIXED_VARIOGRAM = {"azimuth": D["azimuth"], "r_major": D["r_major"],
                   "r_minor": D["r_minor"], "nugget": 0.0,
                   "sill": D["sill"], "model": D["vmodel"]}


def cb_estimar():
    """Estima TODOS los escenarios por dominios y despliega la
    estimación en el mapa."""
    if not ss.scenarios:
        return
    c = ss.case
    for sc in ss.scenarios:
        sc["est_result"] = estimate_scenario(
            "Kriging ordinario", c["samples"], c["blocks"],
            sc["mask"], sc["sample_mask"], FIXED_VARIOGRAM)
        sc["estimated"] = True
    # Desplegar la estimación recién calculada en el mapa
    ss.pending_view = "🗺️ Mapa"
    ss.layer_opt = "Ley estimada"


def cb_categorizar():
    """Marca la categorización, libera Incertidumbre y despliega la
    categorización en el mapa."""
    if not any(sc.get("est_result") for sc in ss.scenarios):
        return
    for sc in ss.scenarios:
        sc["classified"] = True
    ss.categorized = True
    ss.pending_view = "🗺️ Mapa"
    ss.layer_opt = "Categorías de recurso"


def cb_gen_p():
    """Computa p(mineral) desde los escenarios y va a Incertidumbre."""
    if not ss.scenarios:
        return
    ss.p_mineral = uncertainty.compute_p_mineral(
        ss.scenarios, len(ss.case["blocks"]))
    ss.pending_view = "🎲 Incertidumbre"


def cb_demo():
    """MODO PROFESOR: reemplaza los escenarios por 5 interpretaciones
    generadas automáticamente (sólo para probar la aplicación)."""
    c = ss.case
    sets = professor.generate_demo_interpretations(c, cutoff)
    ss.scenarios = []
    for i, polys in enumerate(sets, start=1):
        mask, area = geometry.scenario_mask(polys, c["blocks"])
        smask = geometry.samples_inside(polys, c["samples"])
        ss.scenarios.append(state.new_scenario_dict(
            i, f"Escenario {i}", polys, mask, smask, area,
            c["block_size"], espesor, densidad))
    # Reiniciar el flujo guiado con los escenarios nuevos
    ss.active_scenario = 1
    ss.pending_active = 1
    ss.p_mineral = None
    ss.categorized = False
    ss.pending_view = "🔍 Interpretar"


# ==================================================================
# PANEL IZQUIERDO: CONTROLES
# ==================================================================
with st.sidebar:
    # Logo institucional en la cabecera del panel
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.title("⛏️ Controles")

    # (Sin controles de caso sintético: el caso es fijo —
    #  cutoff 0.30 %Cu, espesor 20 m, densidad 2.6 t/m³, 20 800 t/bloque)

    # ---------- Controles del mapa ----------
    with st.expander("🗺️ Controles del mapa", expanded=False):
        # key=layer_opt permite que los callbacks (estimar/categorizar)
        # cambien la capa mostrada programáticamente
        layer_opt = st.radio("Capa a mostrar en el mapa", [
            "Ninguna", "Ley estimada", "Categorías de recurso",
            "p(mineral)", "p(1-p)", "Distancia a sondajes"],
            index=0, key="layer_opt")
        show_labels = st.checkbox("Mostrar valores de ley (Cu%)", True)
        show_grid = st.checkbox("Mostrar grilla de bloques", False)
        show_saved = st.checkbox("Mostrar escenarios guardados", False)
        # Control opcional: zonas Medido/Indicado/Inferido sobre el mapa
        show_cat_zones = st.checkbox(
            "Mostrar zonas Medido/Indicado/Inferido",
            value=False, disabled=not ss.categorized,
            help="Se habilita después de categorizar recursos.")
        point_size = st.slider("Tamaño de puntos", 15, 100, 45, 5)

    # ==================================================================
    # ETAPAS DEL EJERCICIO (siempre visibles, se activan en orden)
    # ==================================================================
    st.divider()
    st.markdown("#### Etapa 1 · Interpretación")
    n_esc = len(ss.scenarios)
    meta = D["meta_escenarios"]
    st.progress(min(n_esc / meta, 1.0),
                text=f"Escenarios guardados: {n_esc} / {meta}")

    # Sólo se interpretan polígonos MINERALIZADOS (envolvente)
    draw_mode = st.radio("Herramienta",
                         ["Interpretar polígono", "Mover / editar"],
                         horizontal=True)

    # Nombres FIJOS: Escenario 1..5. Al guardar se ocupa el primer
    # espacio libre; para reinterpretar uno hay que borrarlo primero.
    usados = {sc["scenario_id"] for sc in ss.scenarios}
    libres = [i for i in range(1, meta + 1) if i not in usados]
    if libres:
        st.caption(f"El próximo guardado será **Escenario {libres[0]}**.")
    else:
        st.caption(f"Los {meta} escenarios están completos "
                   f"(borre uno para reinterpretarlo).")

    # Al guardar, el lienzo se limpia solo y queda listo para el
    # siguiente escenario (el ícono de basurero del lienzo permite
    # borrar un dibujo a medias sin guardar).
    guardar = st.button("💾 Guardar interpretación como escenario",
                        use_container_width=True, type="primary")

    # Escenario activo con flechas −/+ (antes / después). Tras guardar,
    # salta al recién creado vía pending_active (los widgets no toman
    # nuevos valores por defecto después de creados).
    if "active_sel" not in ss:
        ss.active_sel = 1
    if ss.get("pending_active"):
        ss.active_sel = ss.pending_active
        ss.pending_active = None
    sel = st.number_input("Escenario activo", 1, meta, step=1,
                          key="active_sel")
    ss.active_scenario = int(sel)
    if state.get_scenario(ss.active_scenario) is None:
        st.caption(f"⚠️ El Escenario {ss.active_scenario} aún no está "
                   f"guardado.")
    if st.button("🗑️ Borrar escenario activo", use_container_width=True,
                 disabled=state.get_scenario(ss.active_scenario) is None):
        state.delete_scenario(ss.active_scenario)
        st.rerun()

    # ---------- Etapa 2: Estimación (sólo el botón) ----------
    # Se activa automáticamente al completar los 5 escenarios
    st.divider()
    st.markdown("#### Etapa 2 · Estimación")
    st.button("⚙️ Estimar leyes", use_container_width=True,
              type="primary", key="btn_estimar", on_click=cb_estimar,
              disabled=n_esc < meta)
    any_estimated = any(sc.get("est_result") for sc in ss.scenarios)

    # ---------- Etapa 3: Categorización ----------
    st.divider()
    st.markdown("#### Etapa 3 · Categorización")
    # Distancias siempre visibles (sin colapsar)
    d_med = st.number_input("Distancia Medido (m)", 10.0, 500.0,
                            D["d_med"], 10.0,
                            disabled=not any_estimated)
    d_ind = st.number_input("Distancia Indicado (m)", 10.0, 500.0,
                            D["d_ind"], 10.0,
                            disabled=not any_estimated)
    d_inf = st.number_input("Distancia Inferido (m)", 10.0, 500.0,
                            D["d_inf"], 10.0,
                            disabled=not any_estimated)
    # Ámbito: categorizar sólo dentro de los polígonos interpretados,
    # o todo el dominio (para comparar con la incertidumbre)
    cat_scope = st.radio("Ámbito de categorización",
                         ["Solo polígonos de mineral", "Todo el dominio"],
                         horizontal=False, key="cat_scope",
                         disabled=not any_estimated)
    st.button("🏷️ Categorizar recursos",
              use_container_width=True, type="primary",
              key="btn_categorizar", on_click=cb_categorizar,
              disabled=not any_estimated)

    # ---------- Etapa 4: Incertidumbre (sólo el botón) ----------
    st.divider()
    st.markdown("#### Etapa 4 · Incertidumbre de escenarios")
    st.button("🎲 Generar probabilidad de mineralización",
              use_container_width=True, type="primary",
              key="btn_genp", on_click=cb_gen_p,
              disabled=not ss.categorized)

    # ---------- Etapa final: Realidad oculta ----------
    # El flujo es secuencial: el botón se activa sólo al terminar la
    # Etapa 4, por lo que el checkbox de confirmación era redundante.
    st.divider()
    st.markdown("#### Etapa final · Realidad oculta")
    st.warning("⚠️ Use esta opción **sólo al final** del ejercicio.")
    if st.button("🔓 Develar realidad", use_container_width=True,
                 type="primary", disabled=ss.p_mineral is None):
        ss.revealed = True
    if ss.revealed:
        st.success("La realidad está develada (sección «Realidad»).")

    # ---------- Modo profesor (oculto: requiere ?profesor=1 en la URL) ----
    if st.query_params.get("profesor") in ("1", "true"):
        with st.expander("🧑‍🏫 Modo profesor", expanded=False):
            st.caption("Sólo para probar la aplicación: genera 5 "
                       "interpretaciones automáticas (reemplaza las "
                       "existentes) construidas desde los sondajes "
                       "mineralizados.")
            st.button("⚡ Generar 5 interpretaciones de prueba",
                      use_container_width=True, key="btn_demo",
                      on_click=cb_demo)

# (Las acciones de Estimar / Categorizar / Generar p se ejecutan en los
#  callbacks cb_estimar / cb_categorizar / cb_gen_p, ANTES del render)
case = ss.case
blocks = case["blocks"]
samples = case["samples"]
n_blocks = len(blocks)

# Productos derivados vigentes
p = ss.p_mineral
p1p = uncertainty.compute_p_one_minus_p(p) if p is not None else None
active_sc = state.get_scenario(ss.active_scenario)
active_cat = None
active_est = None   # estimación PROPIA del escenario activo
if active_sc is not None:
    # Ámbito de categorización: dentro de la interpretación o todo el
    # dominio (controlado en Etapa 3)
    if cat_scope == "Todo el dominio":
        scope_mask = np.ones(n_blocks, dtype=bool)
    else:
        scope_mask = active_sc["mask"]
    active_cat = classify_resources(case["dist"], scope_mask,
                                    d_med, d_ind, d_inf)
    active_est = active_sc.get("est_result")

LAYER_MAP = {"Ninguna": None, "Ley estimada": "estimada",
             "Categorías de recurso": "categorias", "p(mineral)": "p",
             "p(1-p)": "p1p", "Distancia a sondajes": "dist"}
layer = LAYER_MAP[layer_opt]


# ==================================================================
# PANEL CENTRAL: VISTAS NAVEGABLES
# (radio horizontal en vez de st.tabs para poder navegar
#  programáticamente: estimar/categorizar -> Mapa, p -> Incertidumbre)
# ==================================================================
# Encabezado institucional del curso
st.markdown("## Magíster en Gestión Minera — Módulo: Geología y Recursos")
st.caption("Prof. Tatiana Ordenes · Prof. Alejandro Cáceres — "
           "Departamento de Ingeniería en Minas, USACH")

VIEWS = ["📖 Instrucciones", "🔍 Interpretar", "🗺️ Mapa",
         "📊 Comparación", "🎲 Incertidumbre", "🔓 Realidad",
         "💾 Exportar"]
if "view" not in ss:
    ss.view = VIEWS[0]
if ss.get("pending_view"):
    ss.view = ss.pending_view     # navegación programática
    ss.pending_view = None
view = st.radio("Sección", VIEWS, horizontal=True, key="view",
                label_visibility="collapsed")
st.divider()


# ------------------------------------------------------------------
# VISTA 0: instrucciones de uso (vista inicial)
# ------------------------------------------------------------------
if view == "📖 Instrucciones":
    st.markdown("## 📖 Instrucciones de la actividad")
    st.markdown(
        "Se dispone de una campaña de sondajes sobre un sistema "
        "mineralizado de cobre tipo **veta / lente** en un dominio de "
        "**1000 × 1000 m** (planta). Cada punto es la intersección de un "
        "sondaje con la planta y entrega su ley de **Cu%**: "
        "🔴 mineralizado (≥ 0.30%) · 🔵 estéril / baja ley. La geología "
        "verdadera está **oculta** hasta el final.\n\n"
        "**Pista regional:** el sistema presenta una orientación "
        "dominante cercana a **37°**.\n\n"
        "El panel izquierdo guía el ejercicio por **etapas que se "
        "activan en orden**. Las vistas de arriba muestran los "
        "resultados de cada etapa.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
#### Etapa 1 · Interpretación (vista 🔍 Interpretar)
- Observe los sondajes y sus leyes (vista 🗺️ Mapa para ver con ejes,
  grilla y tabla de sondajes).
- **Interprete** la envolvente mineralizada en el lienzo: clic para
  agregar vértices, **clic derecho para cerrar** el polígono. Puede
  interpretar **más de un polígono** por escenario.
- Presione **💾 Guardar interpretación como escenario**: el lienzo se
  limpia y queda listo para el siguiente. El basurero del lienzo borra
  un dibujo a medias.
- Repita hasta completar **5 escenarios distintos y plausibles**
  (no existe una única interpretación correcta).
- Las flechas **−/+ de «Escenario activo»** seleccionan qué escenario
  se muestra en las vistas; **🗑️ Borrar** libera un slot para
  reinterpretarlo.

#### Etapa 2 · Estimación
- Se activa al completar los 5 escenarios.
- **⚙️ Estimar leyes** ejecuta kriging ordinario con variograma fijo
  (esférico, azimut 37°, alcances 90/45 m, sin efecto pepita) **por
  dominios**: los bloques dentro de su interpretación se estiman sólo
  con los sondajes interiores, y los de afuera con el resto. Por eso
  **cada escenario tiene su propia estimación**.
- Al estimar, el mapa despliega la ley estimada del escenario activo
  (bloques de 20 × 20 m, escala Jet 0–1 %Cu).

#### Etapa 3 · Categorización
- Defina las distancias para **Medido / Indicado / Inferido**
  (por defecto 50 / 100 / 150 m al sondaje más cercano) y el ámbito
  (sólo polígonos o todo el dominio).
- **🏷️ Categorizar recursos** despliega las zonas
  🟢 Medido · 🟡 Indicado · 🔴 Inferido en el mapa.
""")
    with c2:
        st.markdown(f"""
#### Etapa 4 · Incertidumbre de escenarios (vista 🎲)
- **🎲 Generar probabilidad de mineralización** calcula, por bloque,
  `p = n° de escenarios que lo incluyen / 5`.
- Verá 4 mapas: probabilidad de ser mineral, zonas más inciertas
  **p(1−p)** (máxima donde los escenarios discrepan), distancia al
  sondaje y categorización; más una tabla por categoría con el % de
  zona incierta (ley de corte {ley_corte:.1f}% CuT).
- Discuta: ¿estar cerca de un sondaje garantiza baja incertidumbre
  geológica?

#### Vista 📊 Comparación
- Tabla por escenario (**Ton en Mt, ley, metal en kt**, sobre ley de
  corte {ley_corte:.1f}% CuT), percentiles **P5 / P50 / P95**, el
  intervalo **I90 = P95 − P5** ilustrado, y boxplots con cada
  escenario etiquetado.
- Discuta: ¿qué variable es más sensible a la interpretación?

#### Etapa final · 🔓 Realidad
- **Sólo al terminar todo lo anterior**, presione «Develar realidad».
- Compare su interpretación y su estimación contra la geología y las
  leyes verdaderas (2×2), revise la tabla Ton/Ley/Metal estimado vs
  real con sus diferencias porcentuales, y el mapa de acierto
  **OO / OW / WO / WW** (mineral correcto / dilución / pérdida /
  estéril correcto).

#### Vista 💾 Exportar
- Descargue sus resultados como **Excel** (resumen, categorías,
  percentiles, polígonos y sondajes) para el informe.
""")
    # ----- Preguntas foco para el informe -----
    st.divider()
    st.markdown(f"""
### ❓ Preguntas foco para el informe

**P1 — El peso de la interpretación** *(vista 📊 Comparación)*
Con los **mismos sondajes**, sus 5 escenarios reportan recursos
distintos. Cuantifique: ¿cuál es el I90 relativo de tonelaje, ley y
metal entre sus escenarios? ¿Qué controla más el recurso reportado: el
método de estimación o la interpretación geológica? Justifique con su
tabla P5/P50/P95.

**P2 — ¿Basta la distancia para cuantificar la confianza?**
*(vista 🎲 Incertidumbre)*
Identifique en sus mapas: (a) bloques **Medido** (≤ {D['d_med']:.0f} m
de un sondaje) que caen en zona incierta (p(1−p) > 0.1), y (b) bloques
lejanos donde **todos** sus escenarios coinciden. ¿Es suficiente la
distancia como criterio de confianza? ¿Qué **modificación concreta**
propondría a la regla de categorización para incorporar la
incertidumbre interpretativa? (p. ej., ¿degradaría un Medido que cae
en zona incierta?)

**P3 — El costo de la continuidad optimista**
*(vistas 🔍 Interpretar y 📊 Comparación)*
Genere al menos un escenario que **incorpore deliberadamente muestras
de baja ley** dentro de la envolvente (interpretación continua /
optimista). ¿Qué efecto observa en la ley media y el metal de ese
escenario versus uno ajustado? ¿Por qué incluir estéril en el dominio
"diluye" la estimación de todos los bloques interiores?
(Pista: ¿con qué muestras se krigea adentro?)

**P4 — La realidad como juez** *(vista 🔓 Realidad)*
¿Cuál de sus escenarios se parece más a la realidad en forma y en
números (Δ% Ton/Ley/Metal)? ¿Sus 5 escenarios fueron sistemáticamente
optimistas o conservadores? En el mapa OO/OW/WO/WW: ¿domina la
**dilución** (OW) o la **pérdida** (WO), y qué sondajes adicionales
propondría para reducir ese error?

**P5 — Determinismo vs incertidumbre: ¿un modelo único para
reservas?** *(reflexión final)*
En la práctica industrial es habitual construir **un único modelo
geológico** ("el mejor") y sobre él estimar, categorizar y declarar
reservas. Usted acaba de comprobar que, con la misma información,
dibujó 5 interpretaciones plausibles con diferencias importantes en
metal — y que la realidad no coincidió exactamente con ninguna.
Reflexione: ¿qué opina de usar un modelo único y determinístico para
efectos de reservas, cuando la información es incompleta por
naturaleza? ¿Ese modelo único transmite al planificador y al
inversionista la incertidumbre que usted experimentó al interpretar?
¿Qué rol deberían jugar los escenarios múltiples y la probabilidad de
mineralización en la declaración de recursos, y qué riesgos ve en cada
enfoque (parálisis por análisis vs falsa certeza)?
""")

    st.info("➡️ Para comenzar, vaya a la vista **🔍 Interpretar** y "
            "dibuje su primer escenario.")


# ------------------------------------------------------------------
# VISTA 1: interpretación (dibujo de polígonos)
# ------------------------------------------------------------------
if view == "🔍 Interpretar":
    # (1) Título de la etapa, arriba
    st.markdown("## 🧭 Interpretación")
    st.markdown(
        "**Interprete** la envolvente mineralizada: haga clic para "
        "agregar vértices y **clic derecho para cerrar** el polígono. "
        "Se puede interpretar **más de un polígono** por escenario. "
        "Pista regional: orientación dominante ≈ **37°**.")

    canvas = None
    if not HAS_CANVAS:
        st.error("Falta el componente de dibujo. Instale con:\n\n"
                 "`pip install streamlit-drawable-canvas`")
    else:
        # (2) Mapa de interpretación EXCLUSIVO para interpretar:
        # sólo sondajes (+ grilla opcional). Sin capas, sin escenarios
        # previos y sin la realidad aunque esté develada.
        bg = plotting.render_canvas_background(
            case, cutoff, scenario_polys=None, layer=None,
            est=None, p=None, show_labels=show_labels,
            show_grid=show_grid, show_truth=False,
            size_px=CANVAS_PX)

        # Todos los polígonos interpretados son mineralizados (rojos)
        cc = st.columns([1, 12, 1])[1]   # canvas centrado
        with cc:
            canvas = st_canvas(
                background_image=bg,
                drawing_mode=("polygon"
                              if draw_mode == "Interpretar polígono"
                              else "transform"),
                stroke_color="#FF0000", fill_color="rgba(255,0,0,0.20)",
                stroke_width=2,
                height=CANVAS_PX, width=CANVAS_PX,
                update_streamlit=True,
                key=f"canvas_{ss.canvas_version}",
            )

    # ---- Estado del lienzo y validación ----
    poligonos = []
    if HAS_CANVAS and canvas is not None and canvas.json_data:
        poligonos = geometry.parse_canvas_polygons(
            canvas.json_data, CANVAS_PX, case["domain"])
    validos = [q for q in poligonos
               if geometry.validate_polygon(q["coords"]) is not None]
    cmid = st.columns([1, 3, 1])[1]
    with cmid:
        st.caption(f"Polígonos interpretados en el lienzo: "
                   f"**{len(validos)}**")
        if len(validos) < len(poligonos):
            st.warning(f"{len(poligonos) - len(validos)} polígono(s) "
                       f"inválido(s) o demasiado pequeño(s): se ignorarán.")

    if guardar:
        vmin = [q for q in validos if q["tipo"] == "mineral"]
        if not vmin:
            st.error("Interprete al menos un polígono cerrado antes "
                     "de guardar.")
        elif not libres:
            st.error(f"Ya tiene los {meta} escenarios guardados. "
                     f"Borre uno para reemplazarlo.")
        else:
            mask, area_poly = geometry.scenario_mask(validos, blocks)
            # Muestras dentro de la interpretación: definen el dominio
            # de estimación "dentro" de este escenario
            smask = geometry.samples_inside(validos, samples)
            if smask.sum() == 0:
                st.warning("Ojo: la interpretación no encierra ningún "
                           "sondaje — el dominio interior no podrá "
                           "estimarse.")
            # Ocupa el primer espacio libre (nombre fijo Escenario N)
            slot = libres[0]
            sc = state.new_scenario_dict(
                slot, f"Escenario {slot}", validos, mask, smask,
                area_poly, case["block_size"], espesor, densidad)
            ss.scenarios.append(sc)
            ss.scenarios.sort(key=lambda s: s["scenario_id"])
            ss.active_scenario = slot
            ss.pending_active = slot  # mover el slider a este escenario
            ss.p_mineral = None       # la probabilidad queda obsoleta
            ss.canvas_version += 1    # limpiar lienzo para el siguiente
            st.rerun()

    # ---- Tabla de escenarios guardados (centrada, simplificada) ----
    if ss.scenarios:
        cu_v = samples["Cu_pct"].to_numpy()
        resumen = pd.DataFrame([{
            "Escenario": sc["name"],
            "Polígonos": len(sc["polygons"]),
            "Tonelaje (Mt)": int(sc["mask"].sum()) * ton_b / 1e6,
            "Sondajes dentro": int(sc["sample_mask"].sum()),
            "Mineralizados": int((cu_v[sc["sample_mask"]] >= cutoff).sum()),
            "Estériles": int((cu_v[sc["sample_mask"]] < cutoff).sum()),
        } for sc in ss.scenarios])
        ctab = st.columns([1, 3, 1])[1]
        with ctab:
            st.markdown("#### Escenarios guardados")
            st.dataframe(
                resumen.style.format({"Tonelaje (Mt)": "{:.2f}"}),
                hide_index=True, use_container_width=True)


# ------------------------------------------------------------------
# VISTA 2: mapa principal con capas
# ------------------------------------------------------------------
if view == "🗺️ Mapa":
    faltantes = {"estimada": active_est is None,
                 "categorias": active_cat is None or active_est is None,
                 "p": p is None, "p1p": p1p is None}
    if layer and faltantes.get(layer, False):
        st.info("Esa capa aún no está disponible: genere primero la "
                "estimación / categorización / probabilidad según "
                "corresponda (botones del panel izquierdo).")
    if layer == "estimada" and active_est is not None and active_sc:
        st.caption(f"Ley estimada del escenario activo "
                   f"**{active_sc['name']}** (dominios dentro/fuera de su "
                   f"interpretación). Cambie el escenario activo para ver "
                   f"cómo cambia la estimación.")
    fig = plotting.make_main_map(
        case, cutoff, layer=layer, layer_name=layer_opt,
        est=active_est, categories=active_cat, p=p, p1p=p1p,
        cat_overlay=(active_cat if (show_cat_zones and ss.categorized)
                     else None),
        scenario_polys=(active_sc["polygons"] if active_sc else None),
        other_scenarios=(ss.scenarios if show_saved else None),
        show_samples=True, show_labels=show_labels, show_grid=show_grid,
        show_truth=ss.revealed, point_size=point_size,
        title=f"Dominio 1000 × 1000 m — capa: {layer_opt}")
    st.pyplot(fig, use_container_width=False)

    with st.expander("📄 Tabla de sondajes"):
        tabla_s = samples.copy()
        tabla_s["clase"] = np.where(tabla_s["Cu_pct"] >= cutoff,
                                    "mineralizado", "estéril")
        tabla_s = tabla_s.rename(columns={"X": "Este (m)",
                                          "Y": "Norte (m)"})
        st.dataframe(tabla_s.round(3), hide_index=True,
                     use_container_width=True)


# ------------------------------------------------------------------
# VISTA 4: comparación entre escenarios
# ------------------------------------------------------------------
if view == "📊 Comparación":
    estimados = [sc for sc in ss.scenarios if sc.get("est_result")]
    if len(ss.scenarios) == 0:
        st.info(f"Guarde escenarios para compararlos "
                f"(meta: {D['meta_escenarios']}).")
    elif not estimados:
        st.info("Presione «⚙️ Estimar leyes» para poder comparar.")
    else:
        if len(estimados) < len(ss.scenarios):
            st.warning(f"{len(ss.scenarios) - len(estimados)} escenario(s) "
                       f"aún sin estimar: presione «Estimar leyes» para "
                       f"incluirlos.")
        # Todo se computa sobre la ley de corte (0.2% CuT).
        # Unidades: tonelaje en Mt, metal en kt.
        totals = reporting.report_total_by_scenario(
            estimados, case["block_size"], espesor, densidad,
            cutoff_grade=ley_corte)
        st.subheader(f"Resumen por escenario — ley de corte "
                     f"{ley_corte:.1f}% CuT")
        st.dataframe(
            totals.style.format({"Ton (Mt)": "{:.2f}",
                                 "Ley media Cu%": "{:.3f}",
                                 "Metal Cu (kt)": "{:.2f}"}),
            hide_index=True, use_container_width=True)

        perc = reporting.scenario_percentiles(totals)
        st.subheader("Estadísticas entre escenarios (P5 / P50 / P95 / I90)")
        st.dataframe(
            perc.style.format({"P5": "{:,.2f}", "P50": "{:,.2f}",
                               "P95": "{:,.2f}", "I90 abs": "{:,.2f}",
                               "I90 rel %": "{:.1f}"}),
            hide_index=True, use_container_width=True)

        # «Foto» del I90: percentiles e intervalo sobre los escenarios
        st.pyplot(plotting.make_i90_figure(totals),
                  use_container_width=False)

        # Boxplots con la etiqueta de cada escenario
        st.pyplot(plotting.make_scenario_stripplots(totals),
                  use_container_width=False)


# ------------------------------------------------------------------
# VISTA 5: incertidumbre geológica
# ------------------------------------------------------------------
if view == "🎲 Incertidumbre":
    if p is None:
        st.info("Presione «🎲 Generar probabilidad de mineralización» en el "
                "panel izquierdo (requiere escenarios guardados).")
    else:
        n_used = len(ss.scenarios)
        st.markdown(f"Probabilidad calculada con **{n_used} escenario(s)**: "
                    f"`p = n° escenarios que incluyen el bloque / {n_used}`")
        FS = (6.2, 5.5)   # mismo tamaño para los 4 mapas

        # ----- Layout 2x2 -----
        c1, c2 = st.columns(2)
        with c1:
            # Superior izquierda: probabilidad de ser mineral
            st.pyplot(plotting.make_main_map(
                case, cutoff, layer="p", p=p, show_labels=False,
                point_size=16, title="Probabilidad de ser mineral",
                figsize=FS), use_container_width=False)
        with c2:
            # Superior derecha: zonas más inciertas p(1-p), escala Jet
            st.pyplot(plotting.make_main_map(
                case, cutoff, layer="p1p", p1p=p1p, show_labels=False,
                point_size=16, title="Zonas más inciertas — p(1-p)",
                figsize=FS), use_container_width=False)

        c3, c4 = st.columns(2)
        with c3:
            # Inferior izquierda: distancia al sondaje más cercano
            st.pyplot(plotting.make_main_map(
                case, cutoff, layer="dist", show_labels=False,
                point_size=16, title="Distancia al sondaje",
                figsize=FS), use_container_width=False)
        with c4:
            # Inferior derecha: categorización vigente
            if active_cat is not None and ss.categorized:
                st.pyplot(plotting.make_main_map(
                    case, cutoff, layer="categorias",
                    categories=active_cat, show_labels=False,
                    point_size=16,
                    title=f"Categorización — {active_sc['name']}",
                    figsize=FS), use_container_width=False)
            else:
                st.info("Categorice recursos (Etapa 3) para ver este "
                        "panel.")

        # ----- Tabla por categoría, supeditada a la ley de corte -----
        if active_cat is not None and active_est is not None:
            st.subheader(f"Incertidumbre por categoría — ley de corte "
                         f"{ley_corte:.1f}% CuT")
            tabla_u = uncertainty.category_uncertainty_table(
                active_cat, case["dist"], p1p, active_est["est"],
                cutoff_grade=ley_corte, p1p_threshold=0.1)
            st.dataframe(
                tabla_u.style.format({
                    "Distancia promedio (m)": "{:.1f}",
                    "% zona incierta (p(1-p) > 0.1)": "{:.1f}",
                    "% zona cierta": "{:.1f}"}),
                hide_index=True, use_container_width=True)
            st.caption("Bloques con ley estimada ≥ ley de corte, "
                       "categorización del escenario activo. Zona "
                       "incierta: p(1-p) > 0.1 (los escenarios "
                       "discrepan); zona cierta: p(1-p) ≤ 0.1.")
        else:
            st.info("Estime y categorice para ver la tabla de "
                    "incertidumbre por categoría.")


# ------------------------------------------------------------------
# VISTA 6: realidad develada
# ------------------------------------------------------------------
if view == "🔓 Realidad":
    if not ss.revealed:
        st.info("🔒 La realidad está oculta. Devélela al FINAL del ejercicio "
                "desde el panel izquierdo (sección 6).")
    else:
        # --------------------------------------------------------------
        # Comparación 2x2: interpretación vs realidad (escenario activo)
        #   [interpretación (polígono)] | [geología verdadera]
        #   [ley estimada escenario]    | [ley verdadera]
        # --------------------------------------------------------------
        nombre = active_sc["name"] if active_sc else "—"
        st.subheader(f"Comparación con la realidad — {nombre}")
        if active_sc is None:
            st.info("Use las flechas de «Escenario activo» para elegir "
                    "qué escenario comparar.")
        FS = (5.8, 5.2)   # mismo tamaño para las 4 vistas

        c1, c2 = st.columns(2)
        with c1:
            # Interpretación: polígono mineral del escenario activo
            st.pyplot(plotting.make_main_map(
                case, cutoff,
                scenario_polys=(active_sc["polygons"] if active_sc
                                else None),
                show_labels=False, point_size=18,
                title=f"Interpretación — {nombre}",
                figsize=FS), use_container_width=False)
        with c2:
            # Realidad: cuerpos mineralizados verdaderos
            st.pyplot(plotting.make_main_map(
                case, cutoff, show_truth=True, show_labels=False,
                point_size=18, title="Geología verdadera (3 cuerpos)",
                figsize=FS), use_container_width=False)

        c3, c4 = st.columns(2)
        with c3:
            # Ley estimada del escenario activo (por dominios)
            if active_est is not None:
                st.pyplot(plotting.make_main_map(
                    case, cutoff, layer="estimada", est=active_est,
                    scenario_polys=(active_sc["polygons"] if active_sc
                                    else None),
                    show_labels=False, point_size=18,
                    title=f"Ley estimada — {nombre}",
                    figsize=FS), use_container_width=False)
            else:
                st.info("Este escenario aún no tiene estimación "
                        "(presione «Estimar leyes»).")
        with c4:
            # Ley verdadera (misma escala Jet 0-1)
            st.pyplot(plotting.make_main_map(
                case, cutoff, layer="cu_true", show_labels=False,
                point_size=18, title="Ley verdadera Cu%",
                figsize=FS), use_container_width=False)

        # ----- Tabla escenario vs realidad (ley de corte 0.2% CuT) -----
        estimados_r = [sc for sc in ss.scenarios if sc.get("est_result")]
        if estimados_r:
            st.subheader(f"Escenarios vs realidad — ley de corte "
                         f"{ley_corte:.1f}% CuT")
            dif = reporting.compare_tonnage_with_truth(
                estimados_r, case["cu_true"], case["block_size"],
                espesor, densidad, cutoff_grade=ley_corte)
            st.dataframe(
                dif.style.format({
                    "Ton (Mt)": "{:.2f}", "Ley Cu%": "{:.3f}",
                    "Metal Cu (t)": "{:,.0f}",
                    "Ton real (Mt)": "{:.2f}", "Ley real Cu%": "{:.3f}",
                    "Metal real (t)": "{:,.0f}",
                    "Δ% Ton": "{:+.1f}", "Δ% Ley": "{:+.1f}",
                    "Δ% Metal": "{:+.1f}"}),
                hide_index=True, use_container_width=True)
            st.caption("Estimado: bloques dentro de la interpretación con "
                       "ley estimada ≥ corte. Real: bloques con ley "
                       "verdadera ≥ corte. Δ% = 100·(estimado − real)/real.")

        # ----- Mapa de acierto OO / OW / WO / WW (escenario activo) -----
        if active_est is not None and active_sc is not None:
            ore_est = (active_sc["mask"]
                       & np.isfinite(active_est["est"])
                       & (active_est["est"] >= ley_corte))
            ore_real = case["cu_true"] >= ley_corte
            cmm = st.columns([1, 4, 1])[1]
            with cmm:
                st.pyplot(plotting.make_confusion_map(
                    case, ore_est, ore_real, ley_corte,
                    scenario_name=active_sc["name"]),
                    use_container_width=False)
            st.caption("OO: mineral correcto · OW: dilución (estimado "
                       "mineral, estéril real) · WO: pérdida (estimado "
                       "estéril, mineral real) · WW: estéril correcto. "
                       "Cambie el escenario activo para comparar otros.")


# ------------------------------------------------------------------
# VISTA 7: exportación
# ------------------------------------------------------------------
if view == "💾 Exportar":
    st.subheader("Exportar resultados")
    if not ss.scenarios:
        st.info("No hay escenarios guardados que exportar.")
    else:
        # ----- Construir las hojas del libro Excel -----
        hojas = {}   # nombre de hoja -> DataFrame

        estimados_x = [sc for sc in ss.scenarios if sc.get("est_result")]
        if estimados_x:
            totals = reporting.report_total_by_scenario(
                estimados_x, case["block_size"], espesor, densidad,
                cutoff_grade=ley_corte)
            hojas["Resumen escenarios"] = totals
            # Cada escenario se reporta con SU PROPIA estimación
            hojas["Por categoria"] = pd.concat([
                reporting.report_resources_by_category(
                    sc["name"], sc["mask"],
                    classify_resources(case["dist"], sc["mask"],
                                       d_med, d_ind, d_inf),
                    sc["est_result"]["est"],
                    case["block_size"], espesor, densidad,
                    cutoff_grade=ley_corte)
                for sc in estimados_x], ignore_index=True)
            hojas["Percentiles I90"] = reporting.scenario_percentiles(totals)
        else:
            st.caption("Estime leyes para incluir las hojas de recursos "
                       "en el Excel (por ahora exporta polígonos y "
                       "sondajes).")

        # Polígonos interpretados: una fila por vértice
        filas = []
        for sc in ss.scenarios:
            for j, q in enumerate(sc["polygons"], start=1):
                for k, (x, y) in enumerate(q["coords"], start=1):
                    filas.append({"Escenario": sc["name"], "Polígono": j,
                                  "Vértice": k, "X": round(x, 2),
                                  "Y": round(y, 2)})
        hojas["Poligonos"] = pd.DataFrame(filas)

        # Sondajes con su clase visual (coordenadas Este/Norte)
        tabla_s = samples.copy().round(3)
        tabla_s["clase"] = np.where(tabla_s["Cu_pct"] >= cutoff,
                                    "mineralizado", "estéril")
        hojas["Sondajes"] = tabla_s.rename(
            columns={"X": "Este (m)", "Y": "Norte (m)"})

        # ----- Generar el .xlsx en memoria y ofrecer descarga -----
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for nombre_h, df_h in hojas.items():
                df_h.to_excel(xw, sheet_name=nombre_h[:31], index=False)
        st.download_button(
            "📥 Descargar resultados (Excel)", buf.getvalue(),
            "resultados_interpretacion.xlsx",
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet",
            type="primary")
        st.caption("Hojas incluidas: " + ", ".join(hojas.keys()))

        # Respaldo de las geometrías (por si se quieren recargar después)
        export = []
        for sc in ss.scenarios:
            export.append({
                "scenario_id": sc["scenario_id"], "name": sc["name"],
                "created_at": sc["created_at"],
                "polygons": [{"tipo": q["tipo"],
                              "coords": [[round(x, 2), round(y, 2)]
                                         for x, y in q["coords"]]}
                             for q in sc["polygons"]],
                "area_mineralizada_m2": sc["area_mineralizada"],
            })
        st.download_button("📥 Escenarios (JSON, respaldo)",
                           json.dumps(export, indent=2, ensure_ascii=False),
                           "escenarios.json", "application/json")
