"""
synthetic_case.py
-----------------
Generación del caso sintético 2D en planta:
  - grilla de bloques 20 x 20 m sobre un dominio de 1000 x 1000 m;
  - geología verdadera oculta (3 cuerpos tipo veta/lente, azimut ~37°,
    ambiguamente conectados);
  - campo verdadero de leyes de cobre (Cu%) controlado parcialmente
    por la geología;
  - malla de sondajes casi regular con perturbación aleatoria.

Toda la aleatoriedad usa numpy.random.default_rng(seed) para que el
caso sea 100% reproducible con la misma semilla.
"""

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

# Parámetros geométricos por defecto del caso
DOMAIN = 1000.0   # lado del dominio en metros
BLOCK = 20.0      # lado del bloque en metros


# ------------------------------------------------------------------
# Grilla de bloques
# ------------------------------------------------------------------
def generate_block_grid(domain_size=DOMAIN, block_size=BLOCK):
    """Crea la grilla regular de bloques.

    Devuelve:
        blocks : DataFrame con block_id, ix, iy, x, y (centroides)
        nx, ny : número de bloques en X e Y

    El orden de los bloques es fila por fila (iy, luego ix), de modo que
    cualquier arreglo 1D alineado con `blocks` se puede reorganizar como
    imagen 2D con  arr.reshape(ny, nx).
    """
    nx = int(round(domain_size / block_size))
    ny = int(round(domain_size / block_size))
    xs = (np.arange(nx) + 0.5) * block_size       # centroides en X
    ys = (np.arange(ny) + 0.5) * block_size       # centroides en Y
    XX, YY = np.meshgrid(xs, ys)                  # forma (ny, nx)
    blocks = pd.DataFrame({
        "block_id": np.arange(nx * ny),
        "ix": np.tile(np.arange(nx), ny),
        "iy": np.repeat(np.arange(ny), nx),
        "x": XX.ravel(),
        "y": YY.ravel(),
    })
    return blocks, nx, ny


# ------------------------------------------------------------------
# Campo aleatorio gaussiano anisotrópico (herramienta interna)
# ------------------------------------------------------------------
def gaussian_random_field(nx, ny, cell, r_major, r_minor, azimuth_deg, rng):
    """Genera un campo aleatorio gaussiano estandarizado N(0,1) con
    anisotropía geométrica (rango mayor según el azimut indicado).

    Método: ruido blanco convolucionado (FFT) con un núcleo gaussiano
    anisotrópico rotado. El "rango práctico" se aproxima como 2*sigma,
    por lo que sigma = rango / 2.
    """
    noise = rng.standard_normal((ny, nx))

    # Núcleo gaussiano rotado: tamaño suficiente para cubrir 3 sigma
    half = int(np.ceil(1.5 * max(r_major, r_minor) / cell))
    coords = np.arange(-half, half + 1) * cell
    XX, YY = np.meshgrid(coords, coords)
    az = np.radians(azimuth_deg)
    # u = coordenada a lo largo del azimut (eje mayor), v = perpendicular
    U = XX * np.sin(az) + YY * np.cos(az)
    V = XX * np.cos(az) - YY * np.sin(az)
    sig_u = max(r_major / 2.0, cell / 2.0)
    sig_v = max(r_minor / 2.0, cell / 2.0)
    K = np.exp(-0.5 * ((U / sig_u) ** 2 + (V / sig_v) ** 2))

    field = fftconvolve(noise, K, mode="same")
    field = (field - field.mean()) / field.std()   # estandarizar
    return field


# ------------------------------------------------------------------
# Geología verdadera (oculta)
# ------------------------------------------------------------------
def generate_true_geology(blocks, nx, ny, cell, seed, azimuth_deg=37.0):
    """Genera la geología verdadera: 3 cuerpos angostos tipo veta/lente
    orientados ~37°, ambiguamente conectados entre sí.

    Construcción:
      1. Tres elipses alargadas alineadas según el azimut, separadas por
         brechas de ~50-80 m (menores que el espaciamiento de sondajes,
         por eso la conexión es ambigua) y con leve disposición en echelón.
      2. Un ruido espacial de corta escala (rango ~60 m) que deforma los
         bordes y a veces tiende puentes / corta los cuerpos.

    Devuelve dict con:
        mask : arreglo booleano (N,) — bloque mineralizado verdadero
        D    : distancia elíptica normalizada al cuerpo más cercano
               (D < 1 dentro de la elipse base)
    """
    rng = np.random.default_rng(seed)
    az = np.radians(azimuth_deg)
    e1 = np.array([np.sin(az), np.cos(az)])    # dirección del rumbo (37°)
    e2 = np.array([np.cos(az), -np.sin(az)])   # perpendicular

    # Posiciones de los 3 cuerpos a lo largo del rumbo (t) y lateral (s),
    # con jitter aleatorio leve para variar entre semillas.
    t_centers = np.array([-310.0, 0.0, 310.0]) + rng.uniform(-25, 25, 3)
    s_centers = rng.uniform(-45, 45, 3)                       # en echelón
    half_len = np.array([155.0, 135.0, 150.0]) * rng.uniform(0.9, 1.1, 3)
    half_wid = np.array([55.0, 65.0, 48.0]) * rng.uniform(0.9, 1.1, 3)

    center0 = np.array([500.0, 500.0])  # los cuerpos se alinean por el centro
    xy = blocks[["x", "y"]].to_numpy()

    # Distancia elíptica normalizada al cuerpo más cercano
    D = np.full(len(blocks), np.inf)
    for k in range(3):
        ck = center0 + t_centers[k] * e1 + s_centers[k] * e2
        d = xy - ck
        u = d @ e1   # componente a lo largo del rumbo
        v = d @ e2   # componente perpendicular
        dk = np.sqrt((u / half_len[k]) ** 2 + (v / half_wid[k]) ** 2)
        D = np.minimum(D, dk)

    # Ruido de corta escala que deforma los bordes (variabilidad que la
    # malla de sondajes difícilmente puede capturar)
    noise = gaussian_random_field(nx, ny, cell, 60.0, 30.0, azimuth_deg, rng)
    mask = (D + 0.30 * noise.ravel()) < 1.0

    return {"mask": mask, "D": D}


# ------------------------------------------------------------------
# Campo verdadero de leyes Cu%
# ------------------------------------------------------------------
def generate_true_cu_field(blocks, nx, ny, cell, geology, seed,
                           azimuth_deg=37.0, r_major=50.0, r_minor=25.0):
    """Genera el campo verdadero de Cu% en toda la grilla.

    - La continuidad de la ley es MENOR que la continuidad geológica
      (~150 m). Nota: el núcleo gaussiano usa sigma = rango/2, por lo que
      el alcance práctico EFECTIVO del campo es ~1.7x el nominal:
      con 50/25 nominal se obtiene ~90/45 m (verificado con variograma
      experimental). El variograma del kriging usa 90/45 por coherencia.
    - Dentro del cuerpo: ~0.30 a 1.50 %Cu.
    - Fuera del cuerpo: ~0.01 a 0.10 %Cu, con un leve halo de
      enriquecimiento cerca del contacto (hasta ~0.20 %Cu) para que la
      relación geología-ley no sea trivial.
    - Incluye componente pepítica (nugget) de 30%.
    """
    rng = np.random.default_rng(seed + 1)

    # Campo regionalizado anisotrópico + componente pepítica
    grf = gaussian_random_field(nx, ny, cell, r_major, r_minor,
                                azimuth_deg, rng).ravel()
    pepita = rng.standard_normal(len(blocks))
    z = np.sqrt(0.70) * grf + np.sqrt(0.30) * pepita   # varianza total 1

    # Transformar a probabilidad acumulada (0-1) vía función logística
    # (aproximación suave de la CDF normal, suficiente para fines docentes)
    q = 1.0 / (1.0 + np.exp(-1.7 * z))

    mask = geology["mask"]
    D = geology["D"]

    cu = np.empty(len(blocks))
    # Dentro del cuerpo: 0.30 - 1.50 %Cu, sesgado hacia leyes medias-bajas
    cu[mask] = 0.30 + 1.20 * q[mask] ** 1.5
    # Fuera: 0.01 - 0.10 %Cu base
    cu[~mask] = 0.01 + 0.09 * q[~mask]
    # Halo de contacto: enriquecimiento suave que decae con la distancia
    halo = 0.10 * q * np.exp(-np.maximum(D - 1.0, 0.0) / 0.3)
    cu[~mask] += halo[~mask]

    return cu


# ------------------------------------------------------------------
# Sondajes
# ------------------------------------------------------------------
def generate_drill_samples(spacing, jitter, seed, domain=DOMAIN):
    """Genera la malla de sondajes casi regular con perturbación aleatoria.

    spacing : espaciamiento nominal (70-100 m)
    jitter  : magnitud de la perturbación aleatoria en X e Y (5-10 m)
    """
    rng = np.random.default_rng(seed + 2)
    coords = np.arange(spacing / 2.0, domain, spacing)  # malla nominal
    XX, YY = np.meshgrid(coords, coords)
    GX, GY = np.meshgrid(np.arange(len(coords)), np.arange(len(coords)))
    x = XX.ravel() + rng.uniform(-jitter, jitter, XX.size)
    y = YY.ravel() + rng.uniform(-jitter, jitter, YY.size)
    # Mantener los collares dentro del dominio
    x = np.clip(x, 1.0, domain - 1.0)
    y = np.clip(y, 1.0, domain - 1.0)

    samples = pd.DataFrame({
        "ID": [f"DH-{i+1:03d}" for i in range(len(x))],
        "X": x,
        "Y": y,
        # Índices de la malla nominal (para el raleo "1 por medio")
        "gx": GX.ravel(),
        "gy": GY.ravel(),
    })
    return samples


def sample_cu_at_points(samples, cu_field, nx, ny, cell):
    """Extrae la ley verdadera en la posición de cada sondaje mediante
    interpolación bilineal del campo de bloques."""
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    interp = RegularGridInterpolator(
        (ys, xs), cu_field.reshape(ny, nx),
        bounds_error=False, fill_value=None)  # extrapola en el borde
    pts = np.column_stack([
        np.clip(samples["Y"].to_numpy(), ys[0], ys[-1]),
        np.clip(samples["X"].to_numpy(), xs[0], xs[-1]),
    ])
    return interp(pts)


def prune_samples(samples, truth_D, nx, ny, cell, cutoff=0.30,
                  n_interface=2):
    """Depura la base de sondajes para el ejercicio docente:

    (a) Elimina ~2 muestras ESTÉRILES de la interfaz (las más pegadas al
        contacto del cuerpo): así se agranda la zona donde la
        interpretación queda abierta (menos información en el borde).
    (b) Ralea la periferia puramente estéril "1 por medio" (patrón de
        tablero de ajedrez sobre la malla nominal), lejos del cuerpo:
        la malla queda más espaciada donde no aporta a la interpretación.

    truth_D : distancia elíptica normalizada por bloque (D < 1 = dentro).
    Devuelve el DataFrame depurado con IDs renumerados.
    """
    # Distancia elíptica normalizada interpolada en cada sondaje
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    interp = RegularGridInterpolator((ys, xs), truth_D.reshape(ny, nx),
                                     bounds_error=False, fill_value=None)
    pts = np.column_stack([np.clip(samples["Y"], ys[0], ys[-1]),
                           np.clip(samples["X"], xs[0], xs[-1])])
    D_s = interp(pts)
    esteril = samples["Cu_pct"].to_numpy() < cutoff

    drop = np.zeros(len(samples), dtype=bool)

    # (a) interfaz: estériles con D entre ~0.9 y ~1.9; botar las
    #     n_interface más cercanas al contacto
    idx_int = np.where(esteril & (D_s > 0.9) & (D_s < 1.9))[0]
    if len(idx_int):
        drop[idx_int[np.argsort(D_s[idx_int])[:n_interface]]] = True

    # (b) periferia estéril lejana: 1 por medio (tablero de ajedrez)
    gpar = (samples["gx"].to_numpy() + samples["gy"].to_numpy()) % 2 == 1
    drop |= esteril & (D_s > 2.2) & gpar

    out = samples.loc[~drop].reset_index(drop=True).copy()
    out["ID"] = [f"DH-{i+1:03d}" for i in range(len(out))]
    return out.drop(columns=["gx", "gy"])


def classify_samples(samples, cutoff):
    """Clasifica cada muestra como mineralizada (Cu >= cutoff) o estéril."""
    out = samples.copy()
    out["clase"] = np.where(out["Cu_pct"] >= cutoff, "mineralizado", "esteril")
    return out


# ------------------------------------------------------------------
# Orquestador: caso completo
# ------------------------------------------------------------------
def make_case(seed=12345, spacing=85.0, jitter=8.0,
              domain=DOMAIN, block=BLOCK, azimuth_deg=37.0):
    """Construye el caso sintético completo y lo devuelve como dict.

    Claves del dict:
        blocks, nx, ny, block_size, domain
        truth_mask  : geología verdadera (oculta hasta 'Develar realidad')
        truth_D     : distancia elíptica normalizada (interna)
        cu_true     : campo verdadero de Cu% (oculto)
        samples     : DataFrame ID, X, Y, Cu_pct (lo único visible)
        dist        : distancia de cada bloque al sondaje más cercano
    """
    blocks, nx, ny = generate_block_grid(domain, block)

    geology = generate_true_geology(blocks, nx, ny, block, seed, azimuth_deg)
    cu_true = generate_true_cu_field(blocks, nx, ny, block, geology, seed,
                                     azimuth_deg)

    samples = generate_drill_samples(spacing, jitter, seed, domain)
    samples["Cu_pct"] = sample_cu_at_points(samples, cu_true, nx, ny, block)

    # Depurar la base: quitar 2 estériles de la interfaz y ralear la
    # periferia estéril 1 por medio (ver prune_samples)
    samples = prune_samples(samples, geology["D"], nx, ny, block)

    # Distancia de cada centroide de bloque al sondaje más cercano
    # (calculada con la base YA depurada)
    dist = cdist(blocks[["x", "y"]].to_numpy(),
                 samples[["X", "Y"]].to_numpy()).min(axis=1)

    return {
        "blocks": blocks, "nx": nx, "ny": ny,
        "block_size": block, "domain": domain,
        "truth_mask": geology["mask"], "truth_D": geology["D"],
        "cu_true": cu_true,
        "samples": samples,
        "dist": dist,
        "seed": seed, "spacing": spacing, "jitter": jitter,
        "azimuth": azimuth_deg,
    }
