"""
main.py
=======

Punto de entrada del programa. Se ejecuta con:   python main.py

Abre UNA sola ventana (Qt) con un menu que cambia de pantalla usando un
QStackedWidget (sin abrir ventanas nuevas):

    Menu principal --> "Clasificador Guiado"    -> anotador_napari.Anotador
                                                    (marcar 5 puntos, medir con IA y clasificar)
                   --> "Experimentador KNN"     -> knnQt.PanelKNN
                                                    (probar KNN sobre un dataset CSV)
                   --> "Cerrar"

Archivos que usa:
    assets/HojaMedidaCab1.jpg  imagen de fondo del menu (opcional)
    resultados/hojas_landmarks.csv   donde el anotador guarda los puntos
"""

import sys
from pathlib import Path

# Carpeta donde vive este script; de aqui se derivan las rutas a recursos y resultados.
CARPETA_DEL_PROGRAMA = Path(__file__).resolve().parent
# Los modulos de src/ se importan entre si por nombre plano (import analizador),
# asi que src/ debe estar en el path de Python.
sys.path.insert(0, str(CARPETA_DEL_PROGRAMA / "src"))

try:
    import anotador_napari as anotador
    import knnQt as knn_module
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QApplication,
        QLabel,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    print(f"No se pudo cargar un modulo: {error}")   # mensaje completo, no solo el nombre
    sys.exit(1)


# Los CSV (landmarks y medidas) se escriben aqui; la carpeta se crea sola al guardar.
CARPETA_RESULTADOS = CARPETA_DEL_PROGRAMA / "resultados"

class VentanaUnica(QWidget):
    """Ventana principal: contiene el menu y, apiladas, las demas pantallas.

    Pantallas del QStackedWidget (self.stack):
        - pantalla_menu : menu con tres botones.
        - panel_knn     : panel del experimentador KNN (siempre creado al inicio).
        - vista de napari: se agrega al presionar "Clasificador Guiado".
    """

    def __init__(self, archivo_csv, ruta_imagen=None, revisar=False):
        """archivo_csv: CSV donde se guardan los landmarks.
        ruta_imagen: imagen a anotar; si es None se pide con un cuadro de dialogo.
        revisar: reservado (hoy no se usa; viene de la version con linea de comandos).
        """
        super().__init__()
        self.archivo_csv = archivo_csv
        self.ruta_imagen = ruta_imagen
        self.revisar = revisar

        self.setWindowTitle("Sistema de Análisis de Hojas de Vid")
        self.resize(1100, 750)

        self.layout_principal = QVBoxLayout(self)
        self.layout_principal.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)
        self.layout_principal.addWidget(self.stack)

        # Menú Principal
        self.pantalla_menu = self.crear_pantalla_menu()
        self.stack.addWidget(self.pantalla_menu)

        # Panel KNN
        self.panel_knn = knn_module.PanelKNN(self)
        self.panel_knn.volver_al_menu.connect(self.mostrar_menu)
        self.stack.addWidget(self.panel_knn)

    def mostrar_menu(self):
        """Regresa la vista al menú principal."""
        self.stack.setCurrentWidget(self.pantalla_menu)

    def crear_pantalla_menu(self):
        """Construye la pantalla del menu: fondo con imagen + titulo + tres botones.

        El estilo usa el nombre de objeto "pantallaMenu" para que la imagen de fondo
        solo afecte a este widget y no a los botones y etiquetas que contiene.
        """
        widget = QWidget()
        
        # 1. Asignar un nombre de objeto único para que el fondo no afecte a los botones/labels
        widget.setObjectName("pantallaMenu")
        
        # 2. Definir la ruta de la imagen de fondo (convertida a formato POSIX para CSS)
        ruta_fondo = (CARPETA_DEL_PROGRAMA / "assets" / "HojaMedidaCab1.jpg").as_posix()
        
        # 3. Aplicar el estilo de fondo escalable
        widget.setStyleSheet(f"""
            #pantallaMenu {{
                background-image: url("{ruta_fondo}");
                background-position: center;
                background-repeat: no-repeat;
                background-color: #1a1a1a; /* Fondo oscuro para los bordes */
            }}
            QLabel {{
                color: white;
                background-color: rgba(0, 0, 0, 0.65);
                border-radius: 5px;
                padding: 6px 12px;
            }}
            QPushButton {{
                background-color: rgba(45, 45, 45, 0.9);
                color: white;
                border: 1px solid #555555;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgba(65, 65, 65, 0.9);
            }}
        """)

        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)

        titulo = QLabel("Clasificador de Hojas de Vid")
        titulo.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 30px;")
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        btn_comenzar = QPushButton("Clasificador Guiado")
        btn_comenzar.setFixedSize(260, 45)
        btn_comenzar.setStyleSheet("font-size: 14px;")
        btn_comenzar.clicked.connect(self.iniciar_anotador)
        layout.addWidget(btn_comenzar, alignment=Qt.AlignCenter)

        btn_knn = QPushButton("Experimentador KNN")
        btn_knn.setFixedSize(260, 45)
        btn_knn.setStyleSheet("font-size: 14px;")
        btn_knn.clicked.connect(self.mostrar_knn)
        layout.addWidget(btn_knn, alignment=Qt.AlignCenter)

        btn_cerrar = QPushButton("Cerrar")
        btn_cerrar.setFixedSize(260, 45)
        btn_cerrar.setStyleSheet("font-size: 14px;")
        btn_cerrar.clicked.connect(self.close)
        layout.addWidget(btn_cerrar, alignment=Qt.AlignCenter)

        return widget

    def iniciar_anotador(self):
        """Abre el anotador de napari dentro de esta misma ventana.

        La primera vez pide la imagen; las siguientes reutiliza la misma
        (self.ruta_imagen se conserva). Cada clic crea un Anotador nuevo, y el boton
        "Volver" del anotador regresa aqui mediante callback_menu.
        """
        if self.ruta_imagen is None:
            self.ruta_imagen = anotador.seleccionar_imagen()

        if not self.ruta_imagen:
            return

        self.instancia_anotador = anotador.Anotador(
            self.ruta_imagen, 
            self.archivo_csv, 
            callback_menu=self.mostrar_menu
        )
        vista_napari = self.instancia_anotador.visor.window._qt_window
        self.stack.addWidget(vista_napari)
        self.stack.setCurrentWidget(vista_napari)

    def mostrar_knn(self):
        """Cambia la vista al panel del experimentador KNN."""
        self.stack.setCurrentWidget(self.panel_knn)


def main():
    """Crea la aplicacion Qt, la ventana principal y arranca el ciclo de eventos."""
    app = QApplication.instance() or QApplication([])

    archivo_csv = CARPETA_RESULTADOS / "hojas_landmarks.csv"

    ventana = VentanaUnica(archivo_csv)
    # Al salir (boton Cerrar o la X de la ventana) se borran los archivos temporales del KNN.
    app.aboutToQuit.connect(ventana.panel_knn.limpiar_temporales)
    ventana.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()