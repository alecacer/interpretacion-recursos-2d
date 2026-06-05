"""
make_demo_video.py
------------------
Genera AUTOMÁTICAMENTE un video continuo de uso de la aplicación,
recorriendo el flujo completo como lo haría un estudiante:

  Instrucciones -> interpretar y guardar 5 escenarios (dibujo real en el
  lienzo) -> Estimar -> Categorizar -> Incertidumbre -> Comparación ->
  Develar realidad -> Exportar.

Requisitos (sólo para generar el video, no para usar la app):
    pip install playwright imageio-ffmpeg
    python -m playwright install chromium
    (la app debe estar corriendo en http://localhost:8501)

Ejecución:
    python make_demo_video.py

Salida:
    demo_uso_app.webm  (grabación cruda de Playwright)
    demo_uso_app.mp4   (conversión H.264, lista para compartir)
"""

import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
W, H = 1920, 1080         # tamaño del video (pantalla completa Full HD)
CANVAS_PX = 800           # tamaño del lienzo de dibujo (px) en la app
DOMAIN = 1000.0           # dominio en metros
OUT = Path(__file__).parent / "demo_uso_app"

# Cursor falso visible en la grabación (naranjo USACH), inyectado en
# todos los frames (incluido el iframe del lienzo)
FAKE_CURSOR_JS = """
document.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.style.cssText = 'position:fixed;width:16px;height:16px;' +
    'border-radius:50%;background:rgba(238,119,0,.9);' +
    'border:2px solid #fff;z-index:2147483647;pointer-events:none;' +
    'transform:translate(-50%,-50%);box-shadow:0 0 8px rgba(0,0,0,.6)';
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => {
    c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px';
  }, true);
  // Ocultar la barra de herramientas de Streamlit (modo presentación)
  const s = document.createElement('style');
  s.textContent = '[data-testid="stToolbar"], [data-testid="stDecoration"]' +
                  '{visibility:hidden !important}';
  document.head.appendChild(s);
});
"""

# Los 5 escenarios a dibujar se generan A PRIORI con el mismo motor del
# modo profesor: interpretaciones construidas desde los sondajes
# MINERALIZADOS (envolvente ajustada / amplia con leyes bajas / tres
# cuerpos / veta angosta / dos cuerpos). Cada escenario puede tener
# varios polígonos. Se simplifican los vértices para que el dibujo en
# el lienzo sea fluido y visible.
def escenarios_demo(max_vertices=14):
    from shapely.geometry import Polygon

    import professor
    from synthetic_case import make_case

    case = make_case()   # mismo caso fijo que usa la app
    sets = professor.generate_demo_interpretations(case)

    out = []
    for polys in sets:
        escenario = []
        for q in polys:
            g = Polygon(q["coords"])
            # Simplificar progresivamente hasta tener pocos vértices
            tol = 15.0
            coords = list(g.exterior.coords)[:-1]
            while len(coords) > max_vertices and tol <= 80.0:
                gs = g.simplify(tol)
                coords = list(gs.exterior.coords)[:-1]
                tol += 10.0
            # Mantener los clics dentro del lienzo (margen de 4 px)
            margen = 4.0 / CANVAS_PX * DOMAIN
            coords = [(min(max(x, margen), DOMAIN - margen),
                       min(max(y, margen), DOMAIN - margen))
                      for x, y in coords]
            escenario.append(coords)
        out.append(escenario)
    return out


def world_to_canvas(x, y):
    """Metros -> píxeles del lienzo (Y invertida)."""
    return x / DOMAIN * CANVAS_PX, (1.0 - y / DOMAIN) * CANVAS_PX


class Demo:
    """Pequeño ayudante con movimientos de mouse suaves y esperas."""

    def __init__(self, page):
        self.page = page

    def pause(self, ms):
        self.page.wait_for_timeout(ms)

    def smooth_click(self, locator, button="left", pause_ms=900):
        """Mueve el mouse suavemente hasta el elemento y hace clic."""
        locator.scroll_into_view_if_needed()
        self.pause(250)
        box = locator.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.mouse.move(x, y, steps=22)
        self.pause(200)
        self.page.mouse.click(x, y, button=button)
        self.pause(pause_ms)

    def sidebar(self):
        return self.page.locator('section[data-testid="stSidebar"]')

    def click_sidebar_button(self, texto, pause_ms=1500):
        """Clic en un botón del panel izquierdo (match por subtexto,
        robusto frente a emojis)."""
        btn = self.sidebar().locator("button").filter(
            has_text=texto).first
        self.smooth_click(btn, pause_ms=pause_ms)

    def goto_view(self, etiqueta, pause_ms=1800):
        """Clic en una opción del selector de vistas (radio horizontal
        del área principal)."""
        grp = self.page.locator(
            '[data-testid="stMain"] [role="radiogroup"]').first
        opt = grp.locator("label").filter(has_text=etiqueta).first
        self.smooth_click(opt, pause_ms=pause_ms)

    def scroll(self, px, pausa=900):
        self.page.mouse.wheel(0, px)
        self.pause(pausa)

    # ----- dibujo en el lienzo (iframe del componente) -----
    def canvas_box(self):
        frame_el = self.page.locator(
            'iframe[title="streamlit_drawable_canvas.st_canvas"]').first
        frame_el.scroll_into_view_if_needed()
        self.pause(400)
        return frame_el.bounding_box()

    def draw_polygon(self, coords_world):
        """Dibuja un polígono: clic por vértice, clic derecho cierra."""
        box = self.canvas_box()
        pts = [world_to_canvas(x, y) for x, y in coords_world]
        for px, py in pts:
            ax, ay = box["x"] + px, box["y"] + py
            self.page.mouse.move(ax, ay, steps=18)
            self.pause(150)
            self.page.mouse.click(ax, ay)
            self.pause(350)
        # clic derecho sobre el último vértice para cerrar
        ax, ay = box["x"] + pts[-1][0], box["y"] + pts[-1][1]
        self.page.mouse.click(ax, ay, button="right")
        self.pause(1200)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(OUT.parent),
            record_video_size={"width": W, "height": H},
        )
        ctx.add_init_script(FAKE_CURSOR_JS)
        page = ctx.new_page()
        d = Demo(page)

        print("Abriendo la app...")
        page.goto(URL, wait_until="networkidle")
        d.pause(4000)

        # ---- 1. Instrucciones (vista inicial): lectura breve ----
        print("Instrucciones...")
        d.scroll(450, 1600)
        d.scroll(450, 1600)
        d.scroll(-900, 800)

        # ---- 2. Interpretar y guardar 5 escenarios ----
        print("Dibujando 5 escenarios...")
        d.goto_view("Interpretar", 2200)
        for i, escenario in enumerate(escenarios_demo(), start=1):
            print(f"  Escenario {i} ({len(escenario)} polígono(s))")
            for coords in escenario:   # puede haber varios polígonos
                d.draw_polygon(coords)
            d.click_sidebar_button("Guardar interpretación",
                                   pause_ms=2200)
        d.pause(1500)

        # ---- 3. Estimar (despliega la ley estimada en el mapa) ----
        print("Estimando...")
        d.click_sidebar_button("Estimar leyes", pause_ms=5000)
        d.pause(2500)

        # ---- 4. Categorizar (despliega las zonas en el mapa) ----
        print("Categorizando...")
        d.click_sidebar_button("Categorizar recursos", pause_ms=4500)
        d.pause(2000)

        # ---- 5. Incertidumbre (4 mapas + tabla) ----
        print("Incertidumbre...")
        d.click_sidebar_button("Generar probabilidad",
                               pause_ms=5000)
        d.scroll(550, 2000)
        d.scroll(550, 2000)
        d.scroll(-1100, 800)

        # ---- 6. Comparación (tablas, foto del I90, boxplots) ----
        print("Comparación...")
        d.goto_view("Comparación", 2500)
        d.scroll(500, 2000)
        d.scroll(500, 2200)
        d.scroll(500, 2000)

        # ---- 7. Develar realidad ----
        print("Develando realidad...")
        d.click_sidebar_button("Develar realidad", pause_ms=3000)
        d.goto_view("Realidad", 2500)
        d.scroll(550, 2200)
        d.scroll(550, 2200)
        d.scroll(550, 2200)

        # ---- 8. Exportar ----
        print("Exportar...")
        d.goto_view("Exportar", 2500)
        d.pause(2500)

        video = page.video
        ctx.close()
        browser.close()
        raw = Path(video.path())

    webm = OUT.with_suffix(".webm")
    if webm.exists():
        webm.unlink()
    raw.rename(webm)
    print(f"Video crudo: {webm}")

    # Conversión a MP4 (H.264) con el ffmpeg de imageio-ffmpeg
    import imageio_ffmpeg
    mp4 = OUT.with_suffix(".mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i",
                    str(webm), "-c:v", "libx264", "-crf", "22",
                    "-pix_fmt", "yuv420p", str(mp4)],
                   check=True, capture_output=True)
    print(f"Video final: {mp4}")


if __name__ == "__main__":
    sys.exit(main())
