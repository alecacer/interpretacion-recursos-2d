# App Streamlit: interpretación geológica, estimación y recursos en 2D

Herramienta **docente** (no industrial) para una prueba práctica de
interpretación geológica y estimación de recursos sobre un caso sintético
2D en planta (1000 × 1000 m, bloques de 20 × 20 m).

## Instalación y ejecución

```bash
cd streamlit_resource_app
pip install -r requirements.txt
streamlit run app.py
```

> El dibujo de polígonos usa `streamlit-drawable-canvas`. Si no está
> instalado, la app lo indica y el resto sigue funcionando.

## Flujo de uso (estudiante)

1. Observar los sondajes y leyes Cu% (pestaña **Mapa**). Pista regional:
   orientación dominante ≈ **37°**.
2. En **Dibujo**, trazar la envolvente mineralizada (clic = vértice,
   clic derecho / doble clic = cerrar polígono) y **guardar como escenario**.
   El lienzo se limpia al guardar: cada interpretación se dibuja sin ver
   la anterior (los escenarios guardados se revisan en la pestaña **Mapa**).
3. Repetir hasta completar **5 escenarios** plausibles distintos.
4. **Estimar leyes** con kriging ordinario y variograma **fijo** (esférico,
   azimut 37°, alcances 90/45 m, sin efecto pepita, meseta 1 — no editable;
   calibrado al alcance práctico efectivo del campo simulado).
   La estimación es **por escenario y por dominios de frontera dura**: los
   bloques dentro de la interpretación se estiman sólo con las muestras
   interiores, y los de afuera sólo con el resto. Cada escenario tiene su
   propia estimación, porque al cambiar la interpretación cambia la unidad.
5. **Categorizar recursos** por distancia (Medido 50 m / Indicado 100 m /
   Inferido 150 m, editables).
6. Revisar el reporte por categoría y por escenario (**Resultados**).
7. Comparar escenarios: P5 / P50 / P95 / I90 (**Comparación**).
8. **Generar probabilidad de mineralización**: mapas p(mineral) y p(1−p),
   y comparación con la distancia a sondajes (**Incertidumbre**).
9. Sólo al final: **Develar realidad** (cuerpos verdaderos, campo de Cu%
   verdadero, IoU por escenario, error de estimación).
10. Exportar escenarios (JSON) y tablas (CSV) en **Exportar**.

## Estructura del proyecto

| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz Streamlit: sidebar de controles + pestañas |
| `synthetic_case.py` | Caso sintético: grilla, geología verdadera (3 cuerpos ~37°), campo de Cu%, sondajes |
| `geometry.py` | Conversión canvas↔mundo, validación de polígonos (Shapely), bloques dentro de la interpretación |
| `kriging.py` | Kriging ordinario propio (anisotropía por rotación/escalamiento) con pesos inspeccionables |
| `estimation.py` | Estimación por escenario con dominios dentro/fuera (frontera dura); kriging ordinario / IDW |
| `classification.py` | Categorización por distancia al sondaje más cercano |
| `reporting.py` | Reportes por categoría y escenario, P5/P50/P95/I90, IoU vs realidad, comentarios automáticos |
| `uncertainty.py` | p(mineral) desde escenarios, p(1−p) |
| `plotting.py` | Mapas Matplotlib, fondo del canvas, pesos de kriging, gráficos comparativos |
| `state.py` | Estado de sesión y manejo de escenarios |

## Supuestos del caso

- Bloque 20 × 20 m, espesor 20 m, densidad 2.6 t/m³ → **20 800 t/bloque**.
- Continuidad geológica (~150 m, cuerpos tipo veta/lente a ~37°) **mayor**
  que la continuidad de leyes (alcance práctico efectivo ~90/45 m a 37°,
  con 30% de pepita en el campo verdadero).
- 3 cuerpos ambiguamente conectados: las brechas entre cuerpos (~50–80 m)
  son menores que el espaciamiento de sondajes (~85 m), por lo que la
  malla no resuelve la conexión — ése es el punto pedagógico.
- El caso es **fijo** (semilla interna 12345, espaciamiento 85 m): todos
  los estudiantes trabajan sobre la misma realidad oculta, sin acceso a
  los parámetros de generación. Para cambiar el caso entre semestres,
  editar `DEFAULTS` en `state.py`.
- La base de sondajes se **depura** automáticamente (`prune_samples`):
  se quitan ~2 estériles pegados al contacto del cuerpo (agranda la zona
  de interpretación abierta) y se ralea la periferia puramente estéril
  "1 por medio" (tablero de ajedrez), quedando ~89 sondajes con la malla
  completa sólo cerca del corredor mineralizado.
- Sólo se dibujan polígonos **mineralizados** (envolvente del cuerpo).
- Los mapas de ley (estimada y verdadera) y p(mineral) usan escala
  **Jet de 0 a 1**.

## Modo profesor (pruebas)

Abrir la app con `?profesor=1` en la URL:

    http://localhost:8501/?profesor=1

Aparece la sección **🧑‍🏫 Modo profesor** al final del panel izquierdo, con
el botón **"⚡ Generar 5 interpretaciones de prueba"** (`professor.py`):
reemplaza los escenarios por 5 interpretaciones automáticas construidas
sólo desde los sondajes mineralizados (envolvente ajustada / amplia /
tres cuerpos / veta angosta / dos cuerpos) y reinicia el flujo guiado,
listo para probar Estimar → Categorizar → Incertidumbre → Develar.
Los estudiantes no ven esta sección (URL normal sin el parámetro).

## Notas técnicas

- El kriging es una implementación propia vectorizada que **guarda la
  matriz de pesos por dominio**, lo que permite el inspector "Mostrar
  pesos de kriging" (muestra el dominio del bloque y sólo las muestras
  de ese dominio).
- Si una interpretación no encierra ningún sondaje, su dominio interior
  queda sin estimar (NaN) y la app lo advierte al guardar y al reportar.
- La anisotropía se maneja rotando coordenadas según el azimut y escalando
  la componente perpendicular por (alcance mayor / alcance menor).
- El fondo del canvas se genera sin márgenes para que la conversión
  píxel→metro sea lineal y exacta (640 px = 1000 m).
- Limitación conocida: si se mueve un polígono con la herramienta
  "Mover / editar", las coordenadas exportadas corresponden al trazo
  original (los offsets de fabric.js no se aplican a objetos tipo `path`).
  Para corregir una interpretación, es preferible borrarla y redibujar.
