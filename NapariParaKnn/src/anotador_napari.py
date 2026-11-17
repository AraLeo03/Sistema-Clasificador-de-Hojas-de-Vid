"""
anotador_napari.py
==================

Herramienta gráfica para marcar a mano 5 puntos anatómicos ("landmarks")
en fotos de hojas de vid. Usa napari, un visor de imágenes para Python.

¿Qué hace este archivo?
    1. Solicita al usuario seleccionar un archivo de imagen mediante un cuadro de diálogo.
    2. Abre la foto seleccionada en una ventana de napari.
    3. Tú das clic sobre la hoja para colocar los 5 puntos, en orden.
    4. Al presionar "Guardar", las coordenadas se guardan en un archivo CSV.
    5. Si IA_ANALISIS_AUTOMATICO=true (archivo .env), se manda la imagen anotada
       a una LLM (analizador.py) para obtener la escala, se calculan L1-L4 y los
       angulos, y el resultado se clasifica con KNN (clasificador_externo.py).
       Todo esto corre en un hilo aparte para no congelar la ventana.

Convencion de coordenadas: napari maneja los puntos como (y, x) -- fila primero --;
el CSV y el archivo de contexto de la IA usan (x, y). Se convierte al guardar.

Flujo de un guardado:
    Anotador.guardar_y_siguiente()
      -> guardar_imagen_actual()          escribe/actualiza el CSV de landmarks
      -> lanzar_analisis_ia()             crea archivos temporales + dialogo de espera
      -> pipeline_ia_en_hilo()            (hilo) Gemini -> KNN
      -> al_progreso_ia / al_terminar_ia  actualizan dialogo y panel, guardan medidas_hojas.csv
"""

import shutil
from pathlib import Path

import napari                                  # el visor de imágenes
import numpy as np                             # arreglos de números (las coordenadas)
import pandas as pd                            # tablas (para leer y escribir el CSV)
from PIL import Image, ImageDraw, ImageFont, ImageOps
from napari.qt.threading import thread_worker  # para no congelar la ventana mientras responde la IA
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)
import analizador                              # llamada a Gemini (analizador.py)
import clasificador_externo                    # clasificación KNN con las medidas (constantes de configuración)
from dialogo_ia import DialogoAnalisisIA       # ventana de espera


# ======================================================================
# 1. CONFIGURACIÓN
# ======================================================================

# (nombre del landmark, color en napari, descripcion que sale al pasar el raton)
# El ORDEN importa: es el orden en que se colocan y el que espera el archivo de contexto.
PUNTOS = [
    ("seno_peciolar", "red",    "Donde el pecíolo (el tallito) se une a la hoja"),
    ("punta_L1",      "purple", "Punta del lóbulo central L1 (el de hasta arriba)"),
    ("punta_L2",      "blue",   "Punta del lóbulo L2"),
    ("punta_L3",      "yellow", "Punta del lóbulo L3"),
    ("punta_L4",      "cyan",   "Punta del lóbulo L4 (el más cercano al pecíolo)"),
]

NOMBRES = [nombre for nombre, color, descripcion in PUNTOS]
COLORES = [color for nombre, color, descripcion in PUNTOS]
TOTAL_PUNTOS = len(PUNTOS)  # 5

TECLA_GUARDAR = "g"
TECLA_ANTERIOR = "r"
TECLA_DESHACER = "d"
TECLA_BORRAR = "q"

CONTROLES = [
    ("Clic izquierdo",       "coloca el siguiente punto"),
    (TECLA_GUARDAR.upper(),  "guardar y terminar"),
    (TECLA_ANTERIOR.upper(), "regresar a la imagen anterior"),
    (TECLA_DESHACER.upper(), "deshacer el último punto"),
    (TECLA_BORRAR.upper(),   "quitar todos los puntos de esta imagen"),
    ("Rueda del ratón",      "acercar / alejar"),
    ("Espacio + arrastrar",  "mover la imagen"),
    ("3, luego arrastrar",   "mover un punto ya colocado (2 para volver a añadir)"),
]

# Formatos de imagen aceptados y columnas del CSV de landmarks (una fila por punto).
EXTENSIONES = (".jpg", ".jpeg", ".png")
COLUMNAS_CSV = ["folder_name", "image_name", "variedad", "landmark", "x", "y"]


# ======================================================================
# 2. FUNCIONES DE APOYO
# ======================================================================

def seleccionar_imagen():
    """Abre un cuadro de diálogo del sistema operativo para elegir una imagen."""
    # Inicializa la aplicación Qt si aún no existe
    app = QApplication.instance() or QApplication([])
    
    archivo, _ = QFileDialog.getOpenFileName(
        caption="Selecciona una imagen de hoja",
        filter="Imágenes (*.jpg *.jpeg *.png)"
    )
    if not archivo:
        return None
    return Path(archivo)


def obtener_variedad(ruta_imagen, carpeta_base):
    """Deduce la variedad a partir del nombre o subcarpeta de la imagen."""
    if ruta_imagen.parent != carpeta_base:
        return ruta_imagen.parent.name
    if "_" in ruta_imagen.stem:
        return ruta_imagen.stem.split("_", 1)[0]
    return "sin_variedad"


def cargar_tabla(archivo_csv):
    """Lee el CSV de anotaciones; si no existe, crea una tabla vacía."""
    if archivo_csv.exists():
        return pd.read_csv(archivo_csv)
    return pd.DataFrame(columns=COLUMNAS_CSV)


def puntos_guardados(tabla, nombre_imagen):
    """Devuelve los puntos guardados de una imagen, o None si no tiene."""
    filas = tabla[tabla["image_name"] == nombre_imagen]
    if filas.empty:
        return None

    filas = filas.set_index("landmark").reindex(NOMBRES).dropna(subset=["x", "y"])
    return filas[["y", "x"]].to_numpy()

def avisar(mensaje):
    """Muestra un mensaje en la terminal y como aviso dentro de napari."""
    print(mensaje)
    show_info(mensaje)

def cargar_imagen_vertical(ruta):
    """Abre la foto con la rotación EXIF aplicada y la deja siempre vertical."""
    imagen = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    arreglo = np.array(imagen)
    alto, ancho = arreglo.shape[:2]
    if ancho > alto:                       # sigue horizontal (sin EXIF o EXIF ya aplicado)
        arreglo = np.rot90(arreglo, k=-1)  # 90 grados en sentido horario
    return np.ascontiguousarray(arreglo)

def dibujar_puntos(arreglo, puntos):
    """Dibuja los puntos sobre la imagen original, a su resolución real."""
    imagen = Image.fromarray(arreglo).convert("RGB")
    dibujo = ImageDraw.Draw(imagen)
    alto, ancho = arreglo.shape[:2]
    radio = max(6, int(min(alto, ancho) * 0.008))
    tam_fuente = max(18, int(min(alto, ancho) * 0.025))

    try:
        fuente = ImageFont.truetype("arial.ttf", tam_fuente)
    except OSError:
        try:
            fuente = ImageFont.truetype("DejaVuSans.ttf", tam_fuente)
        except OSError:
            fuente = ImageFont.load_default()

    for i, (nombre, (y, x)) in enumerate(zip(NOMBRES, puntos)):
        dibujo.ellipse([x - radio, y - radio, x + radio, y + radio],
                       fill=COLORES[i], outline="white", width=2)
        dibujo.text((x + radio + 4, y - radio), f"{i + 1}. {nombre}",
                    fill="white", font=fuente, stroke_width=2, stroke_fill="black")
    return imagen

def texto_coordenadas(filas, nombre_imagen, variedad, ancho, alto):
    """Texto con el mismo formato que espera el archivo de contexto (Parte 1)."""
    lineas = [
        f"imagen: {nombre_imagen}",
        f"variedad: {variedad}",
        f"tamano_px: {ancho} x {alto}",
        "",
        "landmark, x, y",
    ]
    for f in filas:
        lineas.append(f"{f['landmark']}, {f['x']:.2f}, {f['y']:.2f}")
    return "\n".join(lineas) + "\n"

@thread_worker
def pipeline_ia_en_hilo(carpeta):
    """Pasos que corren fuera de la ventana: 1) llamar a la IA, 2) clasificar con KNN.

    Cada 'yield' avisa a la ventana de espera en qué paso va; el 'return' final
    entrega todo junto. Si la IA falla, el error llega a 'errored'.
    """
    datos = analizador.analizar_hoja(carpeta)        # puede tardar varios segundos
    yield ("ia_lista", datos)

    resultado = {"datos": datos, "clasificacion": None, "error_clasificacion": None}
    if datos.get("estado") == "OK":
        yield ("clasificando", None)
        try:
            resultado["clasificacion"] = clasificador_externo.clasificar(datos["_local"])
        except Exception as error:                   # p. ej. dataset no encontrado
            resultado["error_clasificacion"] = str(error)
    return resultado


# ======================================================================
# 3. EL ANOTADOR (la ventana de napari)
# ======================================================================

class Anotador:
    """Ventana de napari para colocar los 5 landmarks de UNA imagen.

    Atributos principales:
        visor        : napari.Viewer (se muestra dentro de la ventana principal).
        capa_imagen  : capa con la foto (ya vertical).
        capa_puntos  : capa de puntos; su .data es un arreglo (n, 2) con (y, x).
        tabla        : DataFrame con el CSV de landmarks (todas las imagenes).
        workers      : analisis de IA en curso (para no perderlos de referencia).
        cambiando_puntos : bandera para ignorar eventos cuando el cambio lo hace el codigo.
    """
    def __init__(self, ruta_imagen, archivo_csv, callback_menu=None):
        """ruta_imagen : foto a anotar.
        archivo_csv : CSV de landmarks (se crea si no existe).
        callback_menu: funcion a llamar con el boton "Volver" (None = sin boton).
        """
        self.ruta_imagen = Path(ruta_imagen)
        self.carpeta = self.ruta_imagen.parent
        self.archivo_csv = Path(archivo_csv)
        self.callback_menu = callback_menu

        # Usamos una lista de un solo elemento (la imagen seleccionada)
        self.imagenes = [self.ruta_imagen]
        self.total_imagenes = 1
        self.tabla = cargar_tabla(self.archivo_csv)

        self.indice = 0
        self.cambiando_puntos = False
        self.workers = []          # análisis de IA en curso
        self.ia_lanzada = False

        # Crear la ventana y capas de napari
        self.visor = napari.Viewer(title="Anotador de hojas", show=False)
        #self.visor.window.resize(1200, 800)
        # Ventana de napari creada con show=False: main.py la incrusta en su QStackedWidget.
        # OCULTAR PANELES IZQUIERDOS DE NAPARI
        self.visor.window._qt_viewer.dockLayerControls.setVisible(False)
        self.visor.window._qt_viewer.dockLayerList.setVisible(False)
        self.visor.window._qt_viewer.viewerButtons.setVisible(False)

        self.capa_imagen = self.visor.add_image(cargar_imagen_vertical(self.imagenes[0]), name="Hoja")
        self.capa_puntos = self.visor.add_points(
            name="Puntos",
            ndim=2,
            size=16,
            border_width=0.15,
            border_color="white",
        )

        self.capa_puntos.events.data.connect(self.al_cambiar_puntos)

        self.crear_panel()
        self.asignar_teclas()
        self.mostrar_imagen(0)

    def crear_panel(self):
        """Construye el panel derecho: progreso, lista de puntos, imagen de ejemplo,
        resultado de la IA y botones (Guardar / Deshacer / Quitar todos).
        """
        panel = QWidget()
        diseno = QVBoxLayout(panel)

        diseno.setContentsMargins(8, 8, 8, 8)
        diseno.setSpacing(4)

        # Si existe el callback del menú, agrega el botón al inicio del panel
        if self.callback_menu:
            btn_menu = QPushButton("Volver")
            btn_menu.setStyleSheet("background-color: #2D82B7; color: white; font-weight: bold; padding: 6px;")
            btn_menu.clicked.connect(self.callback_menu)
            diseno.addWidget(btn_menu)

        btn_nueva_hoja = QPushButton("Cargar nueva hoja")
        btn_nueva_hoja.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold; padding: 6px;")
        btn_nueva_hoja.clicked.connect(self.cargar_nueva_hoja)
        diseno.addWidget(btn_nueva_hoja)

        self.texto_imagen = QLabel()
        self.texto_progreso = QLabel()
        self.texto_siguiente = QLabel()
        for texto in (self.texto_imagen, self.texto_progreso, self.texto_siguiente):
            diseno.addWidget(texto)

        diseno.addWidget(QLabel("<br><b>Orden de los puntos</b> (pasa el ratón para ver la descripción)"))
        self.renglones = []
        for nombre, color, descripcion in PUNTOS:
            renglon = QLabel()
            renglon.setToolTip(descripcion)
            diseno.addWidget(renglon)
            self.renglones.append(renglon)

        ruta_ejemplo = Path(__file__).resolve().parent.parent / "assets" / "EjemploBase.png"   # src/ -> raiz -> assets/
        if ruta_ejemplo.exists():
            lbl_ejemplo = QLabel()
            pixmap = QPixmap(str(ruta_ejemplo))
            # Ajusta el ancho a 240px manteniendo la proporción de la imagen
            lbl_ejemplo.setPixmap(pixmap.scaledToWidth(240, Qt.SmoothTransformation))
            lbl_ejemplo.setAlignment(Qt.AlignCenter)
            lbl_ejemplo.setMargin(8)
            diseno.addWidget(lbl_ejemplo)

        self.texto_ia = QLabel()
        self.texto_ia.setWordWrap(True)
        diseno.addWidget(self.texto_ia)

        diseno.addWidget(QLabel(""))
        botones = [
            (f"Guardar y Analizar [{TECLA_GUARDAR.upper()}]", self.guardar_y_siguiente),
            (f"Deshacer último punto  [{TECLA_DESHACER.upper()}]", self.deshacer_ultimo),
            (f"Quitar todos los puntos  [{TECLA_BORRAR.upper()}]", self.borrar_todos),
        ]
        for texto, funcion in botones:
            boton = QPushButton(texto)
            boton.clicked.connect(funcion)
            diseno.addWidget(boton)

        diseno.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(290)

        dock_widget = self.visor.window.add_dock_widget(scroll, area="right", name="Anotación")
        dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock_widget.setTitleBarWidget(QWidget())

    def asignar_teclas(self):
        """Vincula las teclas G, R, D y Q (ver constantes TECLA_*) a sus acciones."""
        self.visor.bind_key(TECLA_GUARDAR, lambda visor: self.guardar_y_siguiente(), overwrite=True)
        self.visor.bind_key(TECLA_ANTERIOR, lambda visor: self.imagen_anterior(), overwrite=True)
        self.visor.bind_key(TECLA_DESHACER, lambda visor: self.deshacer_ultimo(), overwrite=True)
        self.visor.bind_key(TECLA_BORRAR, lambda visor: self.borrar_todos(), overwrite=True)

    def poner_puntos(self, coordenadas):
        """Reemplaza los puntos de la capa SIN disparar al_cambiar_puntos
        (la bandera cambiando_puntos evita un bucle de eventos).
        """
        self.cambiando_puntos = True
        self.capa_puntos.data = coordenadas
        self.cambiando_puntos = False
        self.actualizar_pantalla()

    def al_cambiar_puntos(self, evento):
        """Se ejecuta cada vez que el usuario agrega, mueve o borra un punto.
        Si se pasa de TOTAL_PUNTOS recorta el sobrante y avisa.
        """
        if self.cambiando_puntos:
            return

        if len(self.capa_puntos.data) > TOTAL_PUNTOS:
            self.poner_puntos(self.capa_puntos.data[:TOTAL_PUNTOS])
            avisar(f"Ya están los {TOTAL_PUNTOS} puntos. Presiona {TECLA_GUARDAR.upper()} para guardar.")
            return

        self.actualizar_pantalla()

    def actualizar_pantalla(self):
        """Recolorea los puntos, pone sus etiquetas ("1. seno_peciolar", ...),
        actualiza el panel y el titulo de la ventana con el siguiente punto a colocar.
        """
        n = len(self.capa_puntos.data)

        if n > 0:
            self.capa_puntos.face_color = COLORES[:n]
            self.capa_puntos.features = {
                "etiqueta": [f"{i + 1}. {nombre}" for i, nombre in enumerate(NOMBRES[:n])]
            }
            self.capa_puntos.text = {
                "string": "{etiqueta}",
                "size": 10,
                "color": "white",
                "translation": np.array([12, 0]),
            }

        self.actualizar_panel(n)

        ruta = self.imagenes[self.indice]
        if n < TOTAL_PUNTOS:
            siguiente = f"Coloca: {n + 1}. {NOMBRES[n]}"
        else:
            siguiente = f"Completa: presiona {TECLA_GUARDAR.upper()}"
        self.visor.title = f"Anotador de hojas  |  {ruta.name}  |  {siguiente}"

    def actualizar_panel(self, n):
        """Refresca los textos del panel; n = cuantos puntos hay colocados (✓ hechos, ▶ siguiente)."""
        ruta = self.imagenes[self.indice]
        anotadas = self.tabla["image_name"].nunique()

        self.texto_imagen.setText(
            f"<b>{ruta.name}</b> ({obtener_variedad(ruta, self.carpeta)})<br>"
        )
        self.texto_progreso.setText(f"Guardadas en el CSV: {anotadas}")

        if n < TOTAL_PUNTOS:
            self.texto_siguiente.setText(
                f"<b>Siguiente:</b> {n + 1}. {NOMBRES[n]}<br><i>{PUNTOS[n][2]}</i>"
            )
        else:
            self.texto_siguiente.setText(
                f"<b style='color:lime'>¡Los {TOTAL_PUNTOS} puntos listos!</b> "
                f"Presiona {TECLA_GUARDAR.upper()}."
            )

        for i, (nombre, color, descripcion) in enumerate(PUNTOS):
            if i < n:
                marca = "✓"
            elif i == n:
                marca = "▶"
            else:
                marca = "&nbsp;&nbsp;"
            texto = f"{marca} <span style='color:{color}'>●</span> {i + 1:2d}. {nombre}"
            if i == n:
                texto = f"<b>{texto}</b>"
            self.renglones[i].setText(texto)

    def mostrar_imagen(self, indice):
        """Carga la imagen (vertical), reinicia la vista y recupera sus puntos del CSV si ya estaba anotada."""
        self.indice = indice
        ruta = self.imagenes[indice]

        self.capa_imagen.data = cargar_imagen_vertical(ruta)
        self.visor.reset_view()

        guardados = puntos_guardados(self.tabla, ruta.name)
        if guardados is None:
            guardados = np.empty((0, 2))
        self.poner_puntos(guardados)

        self.visor.layers.selection.active = self.capa_puntos
        self.capa_puntos.mode = "add"

        print(f"\nImagen seleccionada: {ruta.name}")

    def guardar_imagen_actual(self):
        """Guarda los 5 puntos en el CSV (reemplazando los de esta imagen si ya existian)
        y lanza el analisis con IA si esta activado.

        Devuelve False si faltan puntos (no guarda nada). Si True, deja en
        self.ia_lanzada si se inicio el analisis. Las coordenadas se recortan al
        tamano de la imagen y se convierten de (y, x) de napari a columnas x, y.
        """
        puntos = self.capa_puntos.data
        if len(puntos) != TOTAL_PUNTOS:
            faltan = TOTAL_PUNTOS - len(puntos)
            if faltan == 1:
                avisar(f"Falta 1 punto; se guarda solo cuando están los {TOTAL_PUNTOS}.")
            else:
                avisar(f"Faltan {faltan} puntos; se guarda solo cuando están los {TOTAL_PUNTOS}.")
            return False

        ruta = self.imagenes[self.indice]
        alto, ancho = self.capa_imagen.data.shape[:2]

        filas = []
        for nombre, (y, x) in zip(NOMBRES, puntos):
            filas.append({
                "folder_name": self.carpeta.name,
                "image_name": ruta.name,
                "variedad": obtener_variedad(ruta, self.carpeta),
                "landmark": nombre,
                "x": float(np.clip(x, 0, ancho - 1)),
                "y": float(np.clip(y, 0, alto - 1)),
            })
        nuevas = pd.DataFrame(filas, columns=COLUMNAS_CSV)

        otras = self.tabla[self.tabla["image_name"] != ruta.name]
        if otras.empty:
            self.tabla = nuevas
        else:
            self.tabla = pd.concat([otras, nuevas], ignore_index=True)

        self.archivo_csv.parent.mkdir(parents=True, exist_ok=True)
        self.tabla.to_csv(self.archivo_csv, index=False)
        print(f"  Guardado en {self.archivo_csv}")

        self.ia_lanzada = self.lanzar_analisis_ia(ruta, filas)
        return True

    def guardar_y_siguiente(self):
        """Accion de Guardar (tecla G): guarda y avisa si la IA quedo analizando.
        (El nombre viene de cuando el programa recorria varias imagenes; hoy es una sola.)
        """
        if not self.guardar_imagen_actual():
            return

        self.actualizar_pantalla()
        if self.ia_lanzada:
            avisar(f"Puntos guardados en {self.archivo_csv}. Analizando con la IA... "
                   "espera el resultado antes de cerrar la ventana.")
        else:
            avisar(f"¡Terminaste! Los datos están guardados en {self.archivo_csv}. "
                   "Ya puedes cerrar la ventana.")

    def imagen_anterior(self):
        """Vuelve a la imagen anterior (sin efecto con una sola imagen; resto de la version con carpeta)."""
        if self.indice == 0:
            avisar("Ya estás en la primera imagen.")
            return
        self.mostrar_imagen(self.indice - 1)

    def deshacer_ultimo(self):
        """Quita el ultimo punto colocado (tecla D)."""
        if len(self.capa_puntos.data) > 0:
            self.capa_puntos.data = self.capa_puntos.data[:-1]

    def borrar_todos(self):
        """Quita todos los puntos de la imagen (tecla Q)."""
        self.capa_puntos.data = np.empty((0, 2))

    def guardar_archivos_para_contexto(self, ruta, filas):
        """Guarda imagen limpia, imagen anotada y coordenadas en resultados/archivos_ia/<foto>/."""
        arreglo = self.capa_imagen.data
        alto, ancho = arreglo.shape[:2]

        # Carpeta fija junto a los CSV (si la foto se vuelve a guardar, se sobrescribe).
        carpeta_salida = self.archivo_csv.parent / "archivos_ia" / ruta.stem
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        # puntos como (y, x), igual que en la capa de napari
        puntos = np.array([[f["y"], f["x"]] for f in filas])

        Image.fromarray(arreglo).save(carpeta_salida / f"{ruta.stem}_vertical.jpg", quality=95)
        dibujar_puntos(arreglo, puntos).save(carpeta_salida / f"{ruta.stem}_anotada.jpg", quality=95)

        texto = texto_coordenadas(filas, ruta.name, filas[0]["variedad"], ancho, alto)
        (carpeta_salida / f"{ruta.stem}_coordenadas.txt").write_text(texto, encoding="utf-8")

        print(f"  Archivos para la IA en {carpeta_salida}")
        return carpeta_salida

    def lanzar_analisis_ia(self, ruta, filas):
        """Si IA_ANALISIS_AUTOMATICO=true: guarda temporales, abre la ventana de espera y analiza."""
        config = analizador.cargar_config()

        carpeta_temp = self.guardar_archivos_para_contexto(ruta, filas)

        if not config["automatico"]:
            return False

        dialogo = DialogoAnalisisIA(self.visor.window._qt_window, ruta.name, config["timeout_s"])
        dialogo.mostrar()

        worker = pipeline_ia_en_hilo(carpeta_temp)
        worker.yielded.connect(lambda evento, d=dialogo: self.al_progreso_ia(evento, d))
        worker.returned.connect(lambda res, c=carpeta_temp, d=dialogo: self.al_terminar_ia(res, c, d))
        worker.errored.connect(lambda error, c=carpeta_temp, d=dialogo: self.al_fallar_ia(error, c, d))
        self.workers.append(worker)
        worker.start()

        self.texto_ia.setText("<i>Analizando con la IA...</i>")
        return True

    def al_progreso_ia(self, evento, dialogo):
        """Recibe los 'yield' del hilo (ver pipeline_ia_en_hilo) y mueve los pasos del dialogo."""
        etapa, contenido = evento
        if etapa == "ia_lista":
            dialogo.ia_lista(analizador.resumen_texto(contenido))
        elif etapa == "clasificando":
            dialogo.clasificando()

    def al_terminar_ia(self, resultado, carpeta_temp, dialogo):
        """Gemini respondió y el KNN terminó: muestra, guarda las medidas y borra lo temporal."""
        try:
            if dialogo.cancelado:
                return
            datos = resultado["datos"]
            clasificacion = resultado["clasificacion"]
            texto_medidas = analizador.resumen_texto(datos)

            if datos.get("estado") != "OK":
                dialogo.mostrar_error(texto_medidas)
                self.texto_ia.setText(f"<span style='color:tomato'>{texto_medidas}</span>")
                return

            datos["clasificacion"] = clasificacion
            archivo_medidas = self.archivo_csv.parent / "medidas_hojas.csv"
            analizador.guardar_medidas(datos, archivo_medidas)

            if clasificacion:
                texto_clasif = clasificador_externo.formatear_resultado(clasificacion)
                resumen = f"Clase predicha: <b>{clasificacion['clase']}</b> ({clasificacion['confianza']:.0f}%)"
            else:
                texto_clasif = f"No se pudo clasificar:\n{resultado['error_clasificacion']}"
                resumen = "<span style='color:tomato'>No se pudo clasificar</span>"

            dialogo.mostrar_resultado(
                texto_medidas, texto_clasif,
                clasificacion["clase"] if clasificacion else None,
                clasificacion["confianza"] if clasificacion else None,
            )
            self.texto_ia.setText("<b>Resultados IA</b><br>" + texto_medidas.replace("\n", "<br>")
                                  + "<br>" + resumen)
            avisar(f"Análisis listo (modelo {datos['modelo_usado']}). Medidas en {archivo_medidas}")
        except Exception as error:
            dialogo.mostrar_error(f"Error al procesar el resultado: {error}")
        finally:
            self.limpiar_temporal(carpeta_temp)

    def al_fallar_ia(self, error, carpeta_temp, dialogo):
        """Se ejecuta si el hilo lanzo una excepcion (sin internet, API key invalida, etc.)."""
        try:
            if not dialogo.cancelado:
                dialogo.mostrar_error(f"Error en el análisis IA:\n{error}")
                self.texto_ia.setText(f"<span style='color:tomato'>Error en la IA: {error}</span>")
            print(f"  Error en el análisis IA: {error}")
        finally:
            self.limpiar_temporal(carpeta_temp)

    def limpiar_temporal(self, carpeta_temp):
        """Borra la carpeta temporal con imagenes/coordenadas enviadas a la IA (salvo que CONSERVAR_ARCHIVOS_IA=true.)."""
        if analizador.cargar_config()["conservar_archivos"]:
            print(f"  (se conservan los archivos temporales en {carpeta_temp})")
            return
        shutil.rmtree(carpeta_temp, ignore_errors=True)

    def cargar_nueva_hoja(self):
        """Abre el diálogo para elegir otra imagen y reinicia el anotador con ella."""
        nueva_ruta = seleccionar_imagen()
        if not nueva_ruta:
            return  # Si el usuario cancela el diálogo, no hace nada

        self.ruta_imagen = Path(nueva_ruta)
        self.carpeta = self.ruta_imagen.parent
        self.imagenes = [self.ruta_imagen]
        self.ia_lanzada = False
        self.texto_ia.clear()  # Limpia los resultados de IA de la hoja anterior

        # Reemplaza la imagen en napari, reinicia el zoom y limpia/carga puntos del CSV
        self.mostrar_imagen(0)

# ======================================================================
# 4. FUNCIÓN PRINCIPAL
# ======================================================================

def iniciar(archivo_csv, ruta_imagen=None, revisar=False):
    """Solicita una imagen si no se especifica y abre la ventana del anotador."""
    if ruta_imagen is None:
        ruta_imagen = seleccionar_imagen()

    if not ruta_imagen:
        print("No se seleccionó ninguna imagen. Saliendo del programa.")
        return

    anotador = Anotador(ruta_imagen, archivo_csv)

    linea = "=" * 60
    print(linea)
    print("  ANOTADOR DE HOJAS")
    print(linea)
    print(f"  Imagen seleccionada: {anotador.ruta_imagen}")
    print(f"  CSV:                {anotador.archivo_csv}")
    print("\n  Controles:")
    for tecla, accion in CONTROLES:
        print(f"    {tecla:<22} {accion}")
    print(linea)

    napari.run()