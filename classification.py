"""
classification.py
-----------------
Categorización de recursos basada en la distancia al sondaje más cercano,
aplicada SOLO dentro de la interpretación mineralizada del escenario.
"""

import numpy as np
from scipy.spatial.distance import cdist

# Orden canónico de las categorías para reportes y colores
CATEGORIES = ["Medido", "Indicado", "Inferido", "No clasificado"]


def nearest_sample_distance(blocks, samples):
    """Distancia de cada centroide de bloque al sondaje más cercano."""
    return cdist(blocks[["x", "y"]].to_numpy(),
                 samples[["X", "Y"]].to_numpy()).min(axis=1)


def classify_resources(dist, mask, d_medido=50.0, d_indicado=100.0,
                       d_inferido=150.0):
    """Asigna categoría de recurso a cada bloque.

    dist : distancia al sondaje más cercano (todos los bloques)
    mask : bloques dentro de la interpretación mineralizada del escenario

    Regla (especificación, sección 6):
        fuera de la interpretación      -> "No recurso"
        dist <= d_medido                -> "Medido"
        dist <= d_indicado              -> "Indicado"
        dist <= d_inferido              -> "Inferido"
        más lejos                       -> "No clasificado"
    """
    cat = np.full(len(dist), "No recurso", dtype=object)
    cat[mask & (dist <= d_inferido)] = "Inferido"
    cat[mask & (dist <= d_indicado)] = "Indicado"
    cat[mask & (dist <= d_medido)] = "Medido"
    cat[mask & (dist > d_inferido)] = "No clasificado"
    return cat
