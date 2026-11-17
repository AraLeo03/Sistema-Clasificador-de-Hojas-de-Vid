"""
dialogo_ia.py
=============

Ventana de espera que se muestra al presionar Guardar en el anotador.

Muestra los pasos (guardar, esperar a la IA, clasificar), un indicador de
progreso con el tiempo transcurrido y, al terminar, las medidas y la clase
predicha por el KNN.

El anotador la maneja con estos metodos:
    dialogo = DialogoAnalisisIA(parent, "cab_1.jpeg", timeout_s=60)
    dialogo.mostrar()
    dialogo.ia_lista(texto_medidas)       # la IA ya respondio
    dialogo.clasificando()                # empieza el KNN
    dialogo.mostrar_resultado(texto_medidas, texto_clasificacion, clase, confianza)
    dialogo.mostrar_error("mensaje")
    dialogo.cancelado                     # True si el usuario cerro la ventana antes de terminar
"""

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QTextEdit, QVBoxLayout,
)

PENDIENTE, EN_CURSO, LISTO, FALLO = "•", "⏳", "✓", "✗"


class DialogoAnalisisIA(QDialog):
    """Dialogo modal con 3 pasos (guardado, IA, KNN), barra de progreso y resultado.

    Atributos utiles para quien lo usa: cancelado (el usuario cerro antes de terminar)
    y terminado.
    """
    def __init__(self, parent=None, nombre_imagen="", timeout_s=120):
        super().__init__(parent)
        self.setWindowTitle("Analizando hoja")
        self.setMinimumWidth(520)
        self.setWindowModality(Qt.ApplicationModal)

        self.cancelado = False
        self.terminado = False
        self.timeout_s = timeout_s
        self.segundos = 0

        diseno = QVBoxLayout(self)

        titulo = QLabel(f"<b>Analizando {nombre_imagen}</b>")
        titulo.setStyleSheet("font-size: 15px;")
        diseno.addWidget(titulo)

        self.textos_pasos = [
            "Puntos guardados y archivos temporales listos",
            "Esperando la respuesta de la IA (escala y medidas)",
            "Clasificando con KNN",
        ]
        self.pasos = []
        for _ in self.textos_pasos:
            etiqueta = QLabel()
            diseno.addWidget(etiqueta)
            self.pasos.append(etiqueta)

        self.barra = QProgressBar()
        self.barra.setRange(0, 0)            # modo "ocupado": barra animada sin porcentaje
        self.barra.setTextVisible(False)
        diseno.addWidget(self.barra)

        self.lbl_tiempo = QLabel()
        diseno.addWidget(self.lbl_tiempo)

        self.lbl_clase = QLabel()
        self.lbl_clase.setAlignment(Qt.AlignCenter)
        self.lbl_clase.setStyleSheet("font-size: 20px; font-weight: bold; padding: 8px;")
        self.lbl_clase.hide()
        diseno.addWidget(self.lbl_clase)

        self.salida = QTextEdit()
        self.salida.setReadOnly(True)
        self.salida.setFontFamily("Courier New")
        self.salida.setLineWrapMode(QTextEdit.NoWrap)
        self.salida.setMinimumHeight(260)
        self.salida.hide()
        diseno.addWidget(self.salida)

        fila = QHBoxLayout()
        fila.addStretch()
        self.boton = QPushButton("Cancelar")
        self.boton.clicked.connect(self.reject)
        fila.addWidget(self.boton)
        diseno.addLayout(fila)

        self.reloj = QTimer(self)
        self.reloj.timeout.connect(self._tic)

        self._estado(0, LISTO)
        self._estado(1, EN_CURSO)
        self._estado(2, PENDIENTE)
        self._actualizar_tiempo()

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _estado(self, i, simbolo):
        """Cambia el simbolo (•, ⏳, ✓, ✗) del paso i."""
        self.pasos[i].setText(f"{simbolo}  {self.textos_pasos[i]}")

    def _actualizar_tiempo(self):
        """Refresca la etiqueta de tiempo transcurrido."""
        self.lbl_tiempo.setText(
            f"Tiempo transcurrido: {self.segundos} s "
            f"(espera maxima promedio por intento: {self.timeout_s:.0f} s)"
        )

    def _tic(self):
        """Se llama cada segundo por el QTimer."""
        self.segundos += 1
        self._actualizar_tiempo()

    def _terminar(self):
        """Detiene el reloj y la barra; el boton pasa de 'Cancelar' a 'Cerrar'."""
        self.terminado = True
        self.reloj.stop()
        self.barra.hide()
        self.boton.setText("Cerrar")

    def reject(self):
        """Cerrar antes de terminar = cancelar (el resultado se descarta)."""
        if not self.terminado:
            self.cancelado = True
        super().reject()

    # ------------------------------------------------------------------
    # Uso desde el anotador
    # ------------------------------------------------------------------

    def mostrar(self):
        """Inicia el reloj y muestra el dialogo."""
        self.reloj.start(1000)
        self.show()

    def ia_lista(self, texto_medidas):
        """La IA ya respondio: marca el paso 2 y muestra las medidas; empieza el paso 3."""
        self._estado(1, LISTO)
        self._estado(2, EN_CURSO)
        self.salida.setPlainText(texto_medidas)
        self.salida.show()

    def clasificando(self):
        """Marca el paso 3 (KNN) como en curso."""
        self._estado(2, EN_CURSO)

    def mostrar_resultado(self, texto_medidas, texto_clasificacion, clase=None, confianza=None):
        """Muestra medidas + clasificacion y la clase predicha (con su confianza). Si clase es None, marca fallo en KNN."""
        self._estado(1, LISTO)
        self._estado(2, LISTO if clase else FALLO)
        if clase:
            self.lbl_clase.setText(f"Clase predicha: {clase}  ({confianza:.0f}%)")
            self.lbl_clase.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 8px; color: #27AE60;")
            self.lbl_clase.show()
        self.salida.setPlainText(texto_medidas + "\n\n" + texto_clasificacion)
        self.salida.show()
        self._terminar()
        self.adjustSize()

    def mostrar_error(self, mensaje, paso=1):
        """Muestra el mensaje de error y marca como fallido el paso indicado (por defecto el de la IA)."""
        self._estado(paso, FALLO)
        self.lbl_clase.setText("No se pudo completar el análisis")
        self.lbl_clase.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 8px; color: tomato;")
        self.lbl_clase.show()
        self.salida.setPlainText(mensaje)
        self.salida.show()
        self._terminar()
        self.adjustSize()