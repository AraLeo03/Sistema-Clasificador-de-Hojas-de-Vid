"""
conftest.py
===========

Configuracion comun de las pruebas (pytest la carga sola).

- Agrega src/ al path (los modulos se importan por nombre plano, igual que en main.py).
- Fuerza Qt "offscreen" para que ninguna prueba abra ventanas.
- Aisla las variables de entorno de la IA: las pruebas NUNCA usan tus API keys ni llaman a la red.
- Si napari NO esta instalado, importar_anotador() lo reemplaza por un doble minimo,
  asi se pueden probar las funciones de apoyo del anotador sin abrir el visor.
"""

import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import pytest

# ----------------------------------------------------------------------
# Datos de ejemplo compartidos
# ----------------------------------------------------------------------

# Hoja del ejemplo del archivo de contexto (cab_1.jpeg, 1204 x 1600 px)
PUNTOS_EJEMPLO = {
    "seno_peciolar": (584.24, 964.21),
    "punta_L1": (471.02, 204.86),
    "punta_L2": (1118.02, 525.32),
    "punta_L3": (1022.39, 1144.53),
    "punta_L4": (728.67, 1282.92),
}
ESCALA_EJEMPLO = 58.67

BLOQUE_OK = """INICIO_RESULTADOS
imagen: cab_1.jpeg
variedad: cabernet_sauvignon
estado: OK
advertencias: ninguna

escala_horizontal_px_cm: 58.92
escala_vertical_px_cm: 58.42
escala_promedio_px_cm: 58.67

L1_px: 767.74
L1_cm: 13.09
A1_grados: 59.05
FIN_RESULTADOS"""

# Filas de un dataset pequeno con 3 clases bien separadas: columnas x, y, clase
FILAS_SINTETICAS = [
    ("1", "1", "A"), ("1", "2", "A"), ("2", "1", "A"),
    ("10", "10", "B"), ("10", "11", "B"), ("11", "10", "B"),
    ("1", "10", "C"), ("1", "11", "C"), ("2", "10", "C"),
]


def escribir_csv(ruta, filas, encabezado=("x", "y", "clase")):
    ruta = Path(ruta)
    lineas = [",".join(encabezado)] + [",".join(f) for f in filas]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def texto_coordenadas_anotador(puntos=PUNTOS_EJEMPLO):
    """Mismo formato que escribe el anotador en *_coordenadas.txt."""
    lineas = ["imagen: cab_1.jpeg", "variedad: cabernet_sauvignon", "tamano_px: 1204 x 1600", "",
              "landmark, x, y"]
    lineas += [f"{n}, {x:.2f}, {y:.2f}" for n, (x, y) in puntos.items()]
    return "\n".join(lineas) + "\n"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Quita del entorno todo lo relacionado con la IA (incluye claves reales del .env)."""
    for var in list(os.environ):
        if var.startswith(("IA_", "GEMINI_", "ANTHROPIC_", "CLAUDE_")):
            monkeypatch.delenv(var, raising=False)


@pytest.fixture
def puntos_ejemplo():
    return dict(PUNTOS_EJEMPLO)


@pytest.fixture
def dataset_sintetico(tmp_path):
    return escribir_csv(tmp_path / "dataset.csv", FILAS_SINTETICAS)


@pytest.fixture(scope="session")
def qapp():
    """QApplication unica para las pruebas con widgets (se omiten si no hay Qt)."""
    try:
        from qtpy.QtWidgets import QApplication
    except Exception as error:
        pytest.skip(f"Qt no disponible: {error}")
    return QApplication.instance() or QApplication([])


def importar_anotador():
    """Importa anotador_napari; si napari no esta instalado usa un doble minimo."""
    try:
        import napari  # noqa: F401
    except Exception:
        napari = types.ModuleType("napari")
        qt = types.ModuleType("napari.qt")
        hilos = types.ModuleType("napari.qt.threading")
        hilos.thread_worker = lambda funcion: funcion          # decorador que no hace nada
        utils = types.ModuleType("napari.utils")
        avisos = types.ModuleType("napari.utils.notifications")
        avisos.show_info = lambda mensaje: None
        for nombre, modulo in {"napari": napari, "napari.qt": qt, "napari.qt.threading": hilos,
                               "napari.utils": utils, "napari.utils.notifications": avisos}.items():
            sys.modules[nombre] = modulo
    try:
        import anotador_napari
    except Exception as error:
        pytest.skip(f"No se pudo importar anotador_napari: {error}")
    return anotador_napari