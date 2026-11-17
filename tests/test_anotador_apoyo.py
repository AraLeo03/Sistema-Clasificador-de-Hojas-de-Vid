"""Pruebas de las funciones de apoyo de anotador_napari.py (sin abrir napari).

Si napari no esta instalado, conftest.importar_anotador() usa un doble minimo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageColor

import analizador
from conftest import importar_anotador

A = importar_anotador()


# ---------------------------------------------------------------- constantes

def test_los_5_puntos_coinciden_con_los_del_analizador():
    """Si cambias el orden o los nombres en un modulo y no en el otro, la IA recibiria datos mal."""
    assert A.NOMBRES == analizador.NOMBRES_PUNTOS
    assert A.TOTAL_PUNTOS == 5


def test_cada_punto_tiene_un_color_distinto_y_valido():
    assert len(set(A.COLORES)) == len(A.COLORES)
    for color in A.COLORES:
        ImageColor.getrgb(color)               # lanza ValueError si no es un color valido


def test_columnas_del_csv_de_landmarks():
    assert A.COLUMNAS_CSV == ["folder_name", "image_name", "variedad", "landmark", "x", "y"]


# ---------------------------------------------------------------- obtener_variedad

def test_variedad_desde_la_subcarpeta(tmp_path):
    ruta = tmp_path / "syrah" / "syr_1.jpeg"
    assert A.obtener_variedad(ruta, tmp_path) == "syrah"


def test_variedad_desde_el_nombre_si_la_foto_esta_suelta(tmp_path):
    assert A.obtener_variedad(tmp_path / "cab_1.jpeg", tmp_path) == "cab"


def test_variedad_sin_pista_es_sin_variedad(tmp_path):
    assert A.obtener_variedad(tmp_path / "foto.jpeg", tmp_path) == "sin_variedad"


# ---------------------------------------------------------------- cargar_tabla / puntos_guardados

def test_cargar_tabla_inexistente_da_tabla_vacia_con_columnas(tmp_path):
    tabla = A.cargar_tabla(tmp_path / "no_existe.csv")
    assert tabla.empty and list(tabla.columns) == A.COLUMNAS_CSV


def test_cargar_tabla_lee_el_csv(tmp_path):
    ruta = tmp_path / "l.csv"
    pd.DataFrame([["f", "a.jpg", "v", "seno_peciolar", 1.5, 2.5]], columns=A.COLUMNAS_CSV).to_csv(ruta, index=False)
    tabla = A.cargar_tabla(ruta)
    assert len(tabla) == 1 and tabla.iloc[0]["landmark"] == "seno_peciolar"


def _tabla(imagen="a.jpg", nombres=None):
    nombres = nombres or A.NOMBRES
    filas = [["f", imagen, "v", n, 100.0 + i, 200.0 + i] for i, n in enumerate(A.NOMBRES) if n in nombres]
    return pd.DataFrame(filas, columns=A.COLUMNAS_CSV)


def test_puntos_guardados_devuelve_y_x_en_el_orden_de_los_landmarks():
    puntos = A.puntos_guardados(_tabla(), "a.jpg")
    assert puntos.shape == (5, 2)
    # el primer punto (seno_peciolar) tiene x=100, y=200 -> napari espera (y, x)
    assert tuple(puntos[0]) == (200.0, 100.0)
    assert tuple(puntos[4]) == (204.0, 104.0)


def test_puntos_guardados_respeta_el_orden_aunque_el_csv_venga_desordenado():
    tabla = _tabla().sample(frac=1, random_state=1).reset_index(drop=True)
    puntos = A.puntos_guardados(tabla, "a.jpg")
    assert [tuple(p) for p in puntos] == [(200.0 + i, 100.0 + i) for i in range(5)]


def test_puntos_guardados_imagen_sin_anotar_es_none():
    assert A.puntos_guardados(_tabla("a.jpg"), "otra.jpg") is None


def test_puntos_guardados_con_puntos_incompletos_devuelve_solo_los_que_hay():
    tabla = _tabla(nombres=["seno_peciolar", "punta_L1"])
    assert A.puntos_guardados(tabla, "a.jpg").shape == (2, 2)


def test_puntos_guardados_ignora_otras_imagenes():
    tabla = pd.concat([_tabla("a.jpg"), _tabla("b.jpg")], ignore_index=True)
    assert A.puntos_guardados(tabla, "b.jpg").shape == (5, 2)


# ---------------------------------------------------------------- cargar_imagen_vertical

def test_imagen_vertical_se_deja_igual(tmp_path):
    ruta = tmp_path / "v.png"
    Image.new("RGB", (20, 40), "white").save(ruta)
    assert A.cargar_imagen_vertical(ruta).shape == (40, 20, 3)


def test_imagen_horizontal_se_gira_90_grados_en_sentido_horario(tmp_path):
    ruta = tmp_path / "h.png"
    imagen = Image.new("RGB", (40, 20), "white")
    imagen.putpixel((0, 0), (255, 0, 0))             # esquina superior izquierda en rojo
    imagen.save(ruta)

    arreglo = A.cargar_imagen_vertical(ruta)
    assert arreglo.shape == (40, 20, 3)              # ahora es vertical (alto > ancho)
    assert tuple(arreglo[0, -1]) == (255, 0, 0)      # la esquina sup-izq pasa a sup-der (giro horario)
    assert arreglo.flags["C_CONTIGUOUS"]


def test_imagen_siempre_sale_en_rgb(tmp_path):
    ruta = tmp_path / "gris.png"
    Image.new("L", (20, 40), 128).save(ruta)
    assert A.cargar_imagen_vertical(ruta).shape[-1] == 3


# ---------------------------------------------------------------- dibujar_puntos

def test_dibujar_puntos_pinta_cada_punto_con_su_color():
    arreglo = np.zeros((800, 400, 3), dtype=np.uint8)
    puntos = [(100 + 150 * i, 80 + 50 * i) for i in range(5)]            # (y, x), bien separados
    imagen = A.dibujar_puntos(arreglo, puntos)

    assert imagen.size == (400, 800)                                      # PIL: (ancho, alto)
    for (y, x), color in zip(puntos, A.COLORES):
        assert imagen.getpixel((x, y))[:3] == ImageColor.getrgb(color)[:3]


def test_dibujar_puntos_no_modifica_el_arreglo_original():
    arreglo = np.zeros((400, 200, 3), dtype=np.uint8)
    A.dibujar_puntos(arreglo, [(50, 50)] * 5)
    assert arreglo.sum() == 0


# ---------------------------------------------------------------- texto_coordenadas

def test_texto_coordenadas_formato_para_el_contexto():
    filas = [{"landmark": "seno_peciolar", "x": 584.2399, "y": 964.2}, {"landmark": "punta_L1", "x": 471.0, "y": 204.86}]
    texto = A.texto_coordenadas(filas, "cab_1.jpeg", "cabernet_sauvignon", 1204, 1600)
    lineas = texto.splitlines()
    assert lineas[0] == "imagen: cab_1.jpeg"
    assert lineas[1] == "variedad: cabernet_sauvignon"
    assert lineas[2] == "tamano_px: 1204 x 1600"
    assert "landmark, x, y" in lineas
    assert "seno_peciolar, 584.24, 964.20" in lineas                      # 2 decimales
    assert texto.endswith("\n")


def test_el_texto_de_coordenadas_se_puede_leer_con_el_analizador(tmp_path, puntos_ejemplo):
    """Contrato entre los dos modulos: lo que escribe el anotador lo entiende el analizador."""
    filas = [{"landmark": n, "x": x, "y": y} for n, (x, y) in puntos_ejemplo.items()]
    ruta = tmp_path / "cab_1_coordenadas.txt"
    ruta.write_text(A.texto_coordenadas(filas, "cab_1.jpeg", "cab", 1204, 1600), encoding="utf-8")
    _, puntos = analizador.leer_coordenadas(ruta)
    for nombre, (x, y) in puntos_ejemplo.items():
        assert puntos[nombre] == (pytest.approx(x, abs=0.01), pytest.approx(y, abs=0.01))