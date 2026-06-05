"""
plotting.py
-----------
Toda la visualización con Matplotlib:
  - mapa principal en planta (capas: ley estimada, categorías, p(mineral),
    p(1-p), distancia, realidad);
  - imagen de fondo para el canvas de dibujo (sin ejes, alineada píxel a
    píxel con el dominio);
  - inspector de pesos de kriging;
  - gráficos de comparación entre escenarios.
"""

import io

import matplotlib
matplotlib.use("Agg")   # backend sin ventana (requisito para Streamlit)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from PIL import Image

from classification import CATEGORIES

# Colores de las categorías: Medido/Indicado/Inferido en
# Verde/Amarillo/Rojo (semi-transparente sobre el mapa)
CAT_COLORS = {
    "Medido": "#2ca02c",         # verde
    "Indicado": "#ffd60a",       # amarillo
    "Inferido": "#e02020",       # rojo
    "No clasificado": "#9e9e9e", # gris
}
COLOR_MINERAL = "#d62728"   # rojo
COLOR_ESTERIL = "#1f77b4"   # azul
SCEN_CYCLE = plt.get_cmap("tab10").colors   # un color por escenario


# ------------------------------------------------------------------
# Capas raster (campos por bloque)
# ------------------------------------------------------------------
def _edges(n, cell):
    """Bordes de celdas para pcolormesh."""
    return np.arange(n + 1) * cell


def draw_layer(ax, values, nx, ny, cell, cmap, vmin, vmax, label,
               colorbar=True, mask=None):
    """Dibuja un campo por bloque como mapa de colores.

    mask : si se entrega, sólo se pintan los bloques donde mask=True.
    """
    arr = np.asarray(values, dtype=float).reshape(ny, nx)
    if mask is not None:
        arr = np.ma.masked_where(~mask.reshape(ny, nx), arr)
    pm = ax.pcolormesh(_edges(nx, cell), _edges(ny, cell), arr,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    if colorbar:
        cb = ax.figure.colorbar(pm, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(label)
    return pm


def draw_categories(ax, categories, nx, ny, cell, colorbar=True,
                    alpha=0.85):
    """Capa de categorías de recurso con colores discretos
    (Medido verde / Indicado amarillo / Inferido rojo, semi-transparente).
    'No recurso' queda transparente."""
    code = np.zeros(len(categories))
    for i, cat in enumerate(CATEGORIES, start=1):
        code[categories == cat] = i
    arr = np.ma.masked_where(code.reshape(ny, nx) == 0,
                             code.reshape(ny, nx))
    cmap = ListedColormap([CAT_COLORS[c] for c in CATEGORIES])
    norm = BoundaryNorm(np.arange(0.5, 5.5), cmap.N)
    ax.pcolormesh(_edges(nx, cell), _edges(ny, cell), arr,
                  cmap=cmap, norm=norm, shading="flat", alpha=alpha)
    if colorbar:
        # leyenda manual (más clara que una barra continua)
        handles = [Line2D([0], [0], marker="s", linestyle="",
                          markerfacecolor=CAT_COLORS[c], markeredgecolor="k",
                          markersize=10, label=c) for c in CATEGORIES]
        ax.legend(handles=handles, loc="upper left", fontsize=8,
                  framealpha=0.9)


# ------------------------------------------------------------------
# Elementos vectoriales
# ------------------------------------------------------------------
def draw_samples(ax, samples, cutoff, show_labels=False, size=45):
    """Puntos de sondaje: rojo = mineralizado, azul = estéril/baja ley,
    con el valor de Cu% opcional sobre cada punto."""
    mineral = samples["Cu_pct"] >= cutoff
    ax.scatter(samples.loc[mineral, "X"], samples.loc[mineral, "Y"],
               s=size, c=COLOR_MINERAL, edgecolors="k", linewidths=0.5,
               zorder=5, label="Mineralizado")
    ax.scatter(samples.loc[~mineral, "X"], samples.loc[~mineral, "Y"],
               s=size, c=COLOR_ESTERIL, edgecolors="k", linewidths=0.5,
               zorder=5, label="Estéril / baja ley")
    if show_labels:
        for _, r in samples.iterrows():
            ax.annotate(f"{r.Cu_pct:.2f}", (r.X, r.Y),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6.5, zorder=6)


def draw_polygons(ax, polygons, color=None, lw=2.0, alpha=0.18, zorder=4):
    """Dibuja los polígonos de un escenario (contorno + relleno suave)."""
    for p in polygons:
        xs, ys = zip(*p["coords"])
        xs, ys = list(xs) + [xs[0]], list(ys) + [ys[0]]   # cerrar
        c = color
        if c is None:
            c = COLOR_MINERAL if p["tipo"] == "mineral" else COLOR_ESTERIL
        ax.fill(xs, ys, facecolor=c, alpha=alpha, zorder=zorder)
        ax.plot(xs, ys, color=c, lw=lw, zorder=zorder + 1)


def draw_grid(ax, nx, ny, cell):
    """Líneas de la grilla de bloques."""
    for v in _edges(nx, cell):
        ax.axvline(v, color="0.8", lw=0.3, zorder=1)
    for v in _edges(ny, cell):
        ax.axhline(v, color="0.8", lw=0.3, zorder=1)


def draw_truth(ax, truth_mask, nx, ny, cell, filled=True):
    """Cuerpos mineralizados verdaderos: relleno rojo suave + contorno."""
    arr = truth_mask.reshape(ny, nx).astype(float)
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    if filled:
        ax.contourf(xs, ys, arr, levels=[0.5, 1.5],
                    colors=[COLOR_MINERAL], alpha=0.25, zorder=2)
    ax.contour(xs, ys, arr, levels=[0.5], colors=[COLOR_MINERAL],
               linewidths=1.8, zorder=3)


def _setup_ax(ax, domain, with_axes=True, title=None):
    """Configura el eje: escala igual, límites del dominio."""
    ax.set_xlim(0, domain)
    ax.set_ylim(0, domain)
    ax.set_aspect("equal")
    if with_axes:
        ax.set_xlabel("Este (m)")
        ax.set_ylabel("Norte (m)")
        if title:
            ax.set_title(title, fontsize=11)
    else:
        ax.set_axis_off()


# ------------------------------------------------------------------
# Especificación de capas disponibles (nombre -> cmap, rango, etiqueta)
# ------------------------------------------------------------------
def layer_spec(name, case, est):
    """Devuelve (valores, cmap, vmin, vmax, etiqueta) para la capa pedida,
    o None si la capa no está disponible aún."""
    if name == "Ley estimada" and est is not None:
        return est["est"], GRADE_CMAP, GRADE_VMIN, GRADE_VMAX, "Cu% estimado"
    if name == "Distancia a sondajes":
        return case["dist"], "magma_r", 0.0, None, "Distancia (m)"
    if name == "Ley verdadera (Cu%)":
        return case["cu_true"], GRADE_CMAP, GRADE_VMIN, GRADE_VMAX, \
            "Cu% verdadero"
    return None


# Escala de colores para mapas de ley: Jet de 0 a 1 (%Cu)
GRADE_CMAP = "jet"
GRADE_VMIN, GRADE_VMAX = 0.0, 1.0
P_CMAP = "jet"           # p=0 azul, p=1 rojo (escala Jet 0-1)


# ------------------------------------------------------------------
# Figuras compuestas para la app
# ------------------------------------------------------------------
def make_main_map(case, cutoff, layer=None, layer_name="", est=None,
                  categories=None, p=None, p1p=None, cat_overlay=None,
                  scenario_polys=None, other_scenarios=None,
                  show_samples=True, show_labels=False, show_grid=False,
                  show_truth=False, point_size=45, title=None,
                  figsize=(8.2, 7.2)):
    """Mapa principal en planta con las capas seleccionadas.

    layer       : 'estimada' | 'categorias' | 'p' | 'p1p' | 'dist' |
                  'cu_true' | None
    cat_overlay : categorías para sobreponer semi-transparentes
                  (zonas Medido/Indicado/Inferido) sobre cualquier capa
    scenario_polys  : polígonos del escenario activo (destacados)
    other_scenarios : lista de escenarios guardados a mostrar tenue
    """
    nx, ny, cell = case["nx"], case["ny"], case["block_size"]
    fig, ax = plt.subplots(figsize=figsize)

    if layer == "estimada" and est is not None:
        draw_layer(ax, est["est"], nx, ny, cell, GRADE_CMAP,
                   GRADE_VMIN, GRADE_VMAX, "Cu% estimado")
    elif layer == "categorias" and categories is not None:
        draw_categories(ax, categories, nx, ny, cell)
    elif layer == "p" and p is not None:
        draw_layer(ax, p, nx, ny, cell, P_CMAP, 0.0, 1.0, "p(mineral)")
    elif layer == "p1p" and p1p is not None:
        draw_layer(ax, p1p, nx, ny, cell, "jet", 0.0, 0.25, "p(1-p)")
    elif layer == "dist":
        draw_layer(ax, case["dist"], nx, ny, cell, "magma_r", 0.0, None,
                   "Distancia a sondaje (m)")
    elif layer == "cu_true":
        draw_layer(ax, case["cu_true"], nx, ny, cell, GRADE_CMAP,
                   GRADE_VMIN, GRADE_VMAX, "Cu% verdadero")

    # Zonas de categorización semi-transparentes (control opcional)
    if cat_overlay is not None and layer != "categorias":
        draw_categories(ax, cat_overlay, nx, ny, cell, alpha=0.45)

    if show_grid:
        draw_grid(ax, nx, ny, cell)
    if show_truth:
        draw_truth(ax, case["truth_mask"], nx, ny, cell)

    # Escenarios guardados (tenues, un color por escenario)
    if other_scenarios:
        for i, sc in enumerate(other_scenarios):
            col = SCEN_CYCLE[i % len(SCEN_CYCLE)]
            draw_polygons(ax, sc["polygons"], color=col, lw=1.0, alpha=0.05)

    # Escenario activo destacado
    if scenario_polys:
        draw_polygons(ax, scenario_polys, lw=2.2, alpha=0.18)

    if show_samples:
        draw_samples(ax, case["samples"], cutoff, show_labels, point_size)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    _setup_ax(ax, case["domain"], with_axes=True,
              title=title or layer_name)
    fig.tight_layout()
    return fig


def render_canvas_background(case, cutoff, scenario_polys=None,
                             layer=None, est=None, p=None,
                             show_labels=True, show_grid=False,
                             show_truth=False, size_px=640):
    """Imagen PIL para el fondo del canvas de dibujo.

    IMPORTANTE: el eje ocupa exactamente todo el lienzo (sin márgenes),
    de modo que la conversión píxel -> metro sea lineal y exacta:
        x = px / size_px * dominio ;  y = (1 - py / size_px) * dominio
    """
    nx, ny, cell = case["nx"], case["ny"], case["block_size"]
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])   # sin márgenes
    ax.set_facecolor("#fafafa")

    if layer == "estimada" and est is not None:
        draw_layer(ax, est["est"], nx, ny, cell, GRADE_CMAP,
                   GRADE_VMIN, GRADE_VMAX, "", colorbar=False)
    elif layer == "p" and p is not None:
        draw_layer(ax, p, nx, ny, cell, P_CMAP, 0.0, 1.0, "",
                   colorbar=False)
    if show_grid:
        draw_grid(ax, nx, ny, cell)
    if show_truth:
        draw_truth(ax, case["truth_mask"], nx, ny, cell)
    if scenario_polys:
        draw_polygons(ax, scenario_polys, lw=1.5, alpha=0.10)
    draw_samples(ax, case["samples"], cutoff, show_labels, size=40)
    _setup_ax(ax, case["domain"], with_axes=False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def make_kriging_weights_map(case, cutoff, info, top_n=12):
    """Mapa del inspector de pesos: bloque seleccionado + líneas hacia las
    muestras más influyentes (ancho de línea proporcional al |peso|)."""
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    draw_samples(ax, case["samples"], cutoff, show_labels=False, size=30)

    bx, by = info["x"], info["y"]
    cell = case["block_size"]
    # bloque seleccionado
    ax.add_patch(plt.Rectangle((bx - cell / 2, by - cell / 2), cell, cell,
                               facecolor="yellow", edgecolor="k", zorder=6))

    tabla = info["samples_used"].head(top_n)
    wmax = max(abs(tabla["peso"]).max(), 1e-9)
    for _, r in tabla.iterrows():
        lw = 0.5 + 4.0 * abs(r["peso"]) / wmax
        col = "green" if r["peso"] >= 0 else "purple"
        ax.plot([bx, r["X"]], [by, r["Y"]], color=col, lw=lw,
                alpha=0.65, zorder=4)
        ax.annotate(f"{r['peso']:.2f}", ((bx + r["X"]) / 2,
                                         (by + r["Y"]) / 2),
                    fontsize=7, color=col, zorder=7)

    _setup_ax(ax, case["domain"], title=(
        f"Pesos de kriging — bloque ({bx:.0f}, {by:.0f}) | "
        f"Cu* = {info['estimated_grade']:.3f}%"))
    fig.tight_layout()
    return fig


# Especificación de las 3 variables de comparación (columna, etiqueta)
COMP_SPECS = [("Ton (Mt)", "Tonelaje (Mt)"),
              ("Ley media Cu%", "Ley media Cu %"),
              ("Metal Cu (kt)", "Metal Cu (kt)")]


def _short_name(nombre):
    """'Escenario 3' -> 'E3' (etiqueta compacta para los gráficos)."""
    return nombre.replace("Escenario ", "E")


def make_scenario_stripplots(totals):
    """Boxplots de tonelaje (Mt), ley y metal (kt) por escenario, con la
    etiqueta de cada escenario junto a su punto."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    nombres = totals["Escenario"].tolist()
    for ax, (col, label) in zip(axes, COMP_SPECS):
        sel = totals[col].notna()
        vals = totals.loc[sel, col].to_numpy()
        nms = [n for n, s in zip(nombres, sel) if s]
        if len(vals) == 0:
            ax.set_axis_off()
            continue
        # caja + puntos individuales (un punto por escenario, con label)
        ax.boxplot(vals, vert=True, widths=0.45, showfliers=False)
        jitter = np.random.default_rng(0).uniform(-0.10, 0.02, len(vals))
        ax.scatter(1 + jitter, vals, color="#d62728", zorder=5, s=30)
        for x, v, nm in zip(1 + jitter, vals, nms):
            ax.annotate(_short_name(nm), (x, v),
                        textcoords="offset points", xytext=(7, -3),
                        fontsize=8, color="#37404A", zorder=6)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def make_i90_figure(totals):
    """«Foto» del I90: para cada variable muestra los escenarios sobre un
    eje, los percentiles P5 / P50 / P95 y el intervalo I90 sombreado."""
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.0))
    nombres = totals["Escenario"].tolist()
    for ax, (col, label) in zip(axes, COMP_SPECS):
        sel = totals[col].notna()
        vals = totals.loc[sel, col].to_numpy()
        nms = [n for n, s in zip(nombres, sel) if s]
        if len(vals) == 0:
            ax.set_axis_off()
            continue
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        i90 = p95 - p5
        i90rel = 50.0 * i90 / p50 if p50 > 0 else np.nan

        # Intervalo I90 sombreado + percentiles
        ax.axvspan(p5, p95, color="#EE7700", alpha=0.15)
        for v, etiqueta, color in [(p5, "P5", "#1f77b4"),
                                   (p50, "P50", "#37404A"),
                                   (p95, "P95", "#d62728")]:
            ax.axvline(v, color=color, ls="--", lw=1.4)
            ax.annotate(etiqueta, (v, 0.85), xycoords=("data",
                        "axes fraction"), ha="center", fontsize=9,
                        color=color, fontweight="bold")

        # Escenarios como puntos con su etiqueta
        ax.scatter(vals, np.zeros(len(vals)), s=55, color="#37404A",
                   zorder=5)
        for v, nm in zip(vals, nms):
            ax.annotate(_short_name(nm), (v, 0.0),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, zorder=6)

        # Flecha del intervalo I90 con su valor
        ax.annotate("", xy=(p95, -0.45), xytext=(p5, -0.45),
                    arrowprops=dict(arrowstyle="<->", color="#EE7700",
                                    lw=1.8))
        ax.annotate(f"I90 = {i90:,.2f}  ({i90rel:.0f}% rel)",
                    ((p5 + p95) / 2, -0.78), ha="center", fontsize=9,
                    color="#EE7700", fontweight="bold")

        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks([])
        ax.set_title(label, fontsize=10, loc="left")
        margen = max(i90, 1e-6) * 0.35
        ax.set_xlim(min(vals.min(), p5) - margen,
                    max(vals.max(), p95) + margen)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Intervalo I90 entre escenarios (P95 − P5)",
                 fontsize=11, color="#37404A")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def make_dist_vs_p1p(case, p1p):
    """Comparación distancia a sondajes vs p(1-p): dos mapas + dispersión."""
    nx, ny, cell = case["nx"], case["ny"], case["block_size"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    draw_layer(axes[0], case["dist"], nx, ny, cell, "magma_r", 0.0, None,
               "Distancia (m)")
    draw_samples(axes[0], case["samples"], 0.30, size=12)
    _setup_ax(axes[0], case["domain"], title="Distancia al sondaje más cercano")

    draw_layer(axes[1], p1p, nx, ny, cell, "inferno", 0.0, 0.25, "p(1-p)")
    draw_samples(axes[1], case["samples"], 0.30, size=12)
    _setup_ax(axes[1], case["domain"], title="Incertidumbre geológica p(1-p)")

    # Dispersión bloque a bloque: ¿más lejos == más incierto?
    axes[2].scatter(case["dist"], p1p, s=6, alpha=0.35, color="#444")
    axes[2].set_xlabel("Distancia a sondaje (m)")
    axes[2].set_ylabel("p(1-p)")
    axes[2].set_title("¿Lejos = incierto?  (cada punto es un bloque)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_confusion_map(case, ore_est, ore_real, cutoff_grade,
                       scenario_name="", figsize=(7.4, 6.6)):
    """Mapa de confusión mineral/estéril estimado vs real (OO/OW/WO/WW):

        OO : estimado mineral  y  realmente mineral  (acierto, verde)
        OW : estimado mineral  pero estéril real     (dilución, rojo)
        WO : estimado estéril  pero mineral real     (pérdida, naranjo)
        WW : estéril en ambos                        (acierto, gris claro)

    ore_est / ore_real : máscaras booleanas por bloque (sobre la ley
    de corte indicada).
    """
    nx, ny, cell = case["nx"], case["ny"], case["block_size"]
    code = np.zeros(len(ore_est))            # WW = 0
    code[ore_est & ore_real] = 1             # OO
    code[ore_est & ~ore_real] = 2            # OW
    code[~ore_est & ore_real] = 3            # WO
    colores = {0: "#d8dde2", 1: "#2ca02c", 2: "#e02020", 3: "#EE7700"}
    etiquetas = {0: "WW (estéril correcto)", 1: "OO (mineral correcto)",
                 2: "OW (dilución: est. mineral, real estéril)",
                 3: "WO (pérdida: est. estéril, real mineral)"}

    fig, ax = plt.subplots(figsize=figsize)
    cmap = ListedColormap([colores[k] for k in range(4)])
    norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)
    ax.pcolormesh(_edges(nx, cell), _edges(ny, cell),
                  code.reshape(ny, nx), cmap=cmap, norm=norm,
                  shading="flat")
    handles = [Line2D([0], [0], marker="s", linestyle="",
                      markerfacecolor=colores[k], markeredgecolor="k",
                      markersize=10, label=etiquetas[k])
               for k in (1, 2, 3, 0)]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.95)
    _setup_ax(ax, case["domain"], title=(
        f"Acierto mineral/estéril — {scenario_name} "
        f"(corte {cutoff_grade:.1f}% CuT)"))
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    """Convierte una figura a bytes PNG (para descargas)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
