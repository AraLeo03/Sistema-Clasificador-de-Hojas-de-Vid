"""Pruebas de analizador.py: lectura de la respuesta de la IA y CSV de medidas."""

import csv

import pytest

import analizador
from conftest import BLOQUE_OK


# ---------------------------------------------------------------- extraer / parsear

def test_extraer_bloque_ignora_texto_alrededor():
    texto = "Claro, aqui esta:\n" + BLOQUE_OK + "\nEspero que sirva."
    bloque = analizador.extraer_bloque(texto)
    assert bloque.startswith("INICIO_RESULTADOS")
    assert bloque.endswith("FIN_RESULTADOS")
    assert "Claro" not in bloque and "Espero" not in bloque


@pytest.mark.parametrize("texto", [
    "",
    "no se puede medir",
    "INICIO_RESULTADOS\nestado: OK",       # sin fin
    "estado: OK\nFIN_RESULTADOS",          # sin inicio
])
def test_extraer_bloque_incompleto_devuelve_none(texto):
    assert analizador.extraer_bloque(texto) is None


def test_parsear_bloque_numeros_y_texto():
    datos = analizador.parsear_bloque(BLOQUE_OK)
    assert datos["estado"] == "OK"
    assert datos["imagen"] == "cab_1.jpeg"
    assert datos["advertencias"] == "ninguna"
    assert datos["escala_promedio_px_cm"] == 58.67
    assert isinstance(datos["L1_px"], float)
    assert datos["A1_grados"] == 59.05


def test_parsear_bloque_respeta_los_dos_puntos_dentro_del_valor():
    datos = analizador.parsear_bloque("motivo: no se ve: la cuadricula")
    assert datos["motivo"] == "no se ve: la cuadricula"


def test_parsear_bloque_de_error():
    bloque = ("INICIO_RESULTADOS\nimagen: x.jpg\nvariedad: v\nestado: ERROR\n"
              "motivo: no se ve la cuadricula\nFIN_RESULTADOS")
    datos = analizador.parsear_bloque(bloque)
    assert datos["estado"] == "ERROR"
    assert datos["motivo"] == "no se ve la cuadricula"


# ---------------------------------------------------------------- mensaje al usuario

def test_armar_texto_usuario_con_y_sin_imagen_limpia():
    con = analizador.armar_texto_usuario("seno_peciolar, 1, 2", True)
    sin = analizador.armar_texto_usuario("seno_peciolar, 1, 2", False)
    assert "seno_peciolar, 1, 2" in con and "INICIO_RESULTADOS" in con
    assert "segunda" in con
    assert "segunda" not in sin


# ---------------------------------------------------------------- fila / CSV de medidas

def _datos(imagen="cab_1.jpeg", clase="Cab", confianza=66.666):
    local = {"L1_px": 767.7412, "L1_cm": 13.0856, "A1_grados": 59.0512}
    datos = {
        "imagen": imagen, "variedad": "cabernet_sauvignon", "modelo_usado": "claude/claude-sonnet-5-5",
        "advertencias": "ninguna", "escala_promedio_px_cm": 58.67, "_local": local,
    }
    if clase:
        datos["clasificacion"] = {"clase": clase, "confianza": confianza}
    return datos


def test_fila_medidas_redondea_a_2_decimales_y_toma_la_clase():
    fila = analizador.fila_medidas(_datos())
    assert fila["L1_px"] == 767.74
    assert fila["L1_cm"] == 13.09
    assert fila["clase_predicha"] == "Cab"
    assert fila["confianza_pct"] == 66.67
    assert fila["modelo"] == "claude/claude-sonnet-5-5"
    assert fila["escala_promedio_px_cm"] == 58.67


def test_fila_medidas_sin_clasificacion_deja_vacios_los_campos():
    fila = analizador.fila_medidas(_datos(clase=None))
    assert fila["clase_predicha"] == "" and fila["confianza_pct"] == ""


def test_guardar_medidas_crea_el_csv_con_las_columnas(tmp_path):
    ruta = tmp_path / "sub" / "medidas.csv"          # la carpeta no existe: debe crearse
    analizador.guardar_medidas(_datos(), ruta)
    with open(ruta, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        filas = list(lector)
        assert lector.fieldnames == analizador.COLUMNAS_MEDIDAS
    assert len(filas) == 1 and filas[0]["imagen"] == "cab_1.jpeg"


def test_guardar_medidas_agrega_imagenes_distintas(tmp_path):
    ruta = tmp_path / "medidas.csv"
    analizador.guardar_medidas(_datos("a.jpg"), ruta)
    analizador.guardar_medidas(_datos("b.jpg"), ruta)
    with open(ruta, newline="", encoding="utf-8") as f:
        assert [r["imagen"] for r in csv.DictReader(f)] == ["a.jpg", "b.jpg"]


def test_guardar_medidas_reemplaza_la_fila_de_la_misma_imagen(tmp_path):
    ruta = tmp_path / "medidas.csv"
    analizador.guardar_medidas(_datos("a.jpg", clase="Cab"), ruta)
    analizador.guardar_medidas(_datos("b.jpg"), ruta)
    analizador.guardar_medidas(_datos("a.jpg", clase="Syrah"), ruta)     # re-analisis de a.jpg
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert [r["imagen"] for r in filas] == ["b.jpg", "a.jpg"]
    assert filas[-1]["clase_predicha"] == "Syrah"