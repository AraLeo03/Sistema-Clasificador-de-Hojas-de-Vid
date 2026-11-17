"""Pruebas de coherencia entre archivos del proyecto (contrato prompt <-> codigo)."""

from pathlib import Path

import pytest

import analizador
from conftest import RAIZ

CONTEXTO = RAIZ / "prompts" / "contextFile.md"
pytestmark = pytest.mark.skipif(not CONTEXTO.exists(), reason="no esta prompts/contextFile.md")


@pytest.fixture(scope="module")
def contexto():
    return CONTEXTO.read_text(encoding="utf-8")


def test_el_analizador_encuentra_el_archivo_de_contexto():
    assert analizador.CONTEXTO_POR_DEFECTO.exists()


def test_el_contexto_menciona_los_5_landmarks(contexto):
    for nombre in analizador.NOMBRES_PUNTOS:
        assert nombre in contexto, nombre


def test_el_contexto_pide_todos_los_angulos_que_el_codigo_calcula(contexto):
    for nombre, _, _ in analizador.ANGULOS:
        assert nombre in contexto, nombre


def test_el_contexto_define_el_bloque_que_el_codigo_espera(contexto):
    assert "INICIO_RESULTADOS" in contexto and "FIN_RESULTADOS" in contexto
    for campo in ("estado", "escala_promedio_px_cm"):
        assert campo in contexto


def test_el_ejemplo_del_contexto_se_parsea_y_cuadra_con_el_calculo_local(contexto, puntos_ejemplo):
    """El ejemplo ilustrativo del prompt debe dar lo mismo que calcular_local()."""
    bloque = analizador.extraer_bloque(contexto[contexto.index("PARTE 7"):])
    datos = analizador.parsear_bloque(bloque)
    assert datos["estado"] == "OK"
    local = analizador.calcular_local(puntos_ejemplo, datos["escala_promedio_px_cm"])
    for campo in ("L1_px", "L4_px", "L1_cm", "L4_cm", "A1_grados", "A3_grados", "A2_4_grados"):
        assert local[campo] == pytest.approx(datos[campo], abs=0.011), campo


def test_las_columnas_del_csv_de_medidas_cubren_lo_que_el_prompt_devuelve(contexto):
    for campo in ("escala_horizontal_px_cm", "escala_vertical_px_cm", "escala_promedio_px_cm",
                  "L1_px", "L1_cm", "L4_cm", "A1_grados", "A2_4_grados"):
        assert campo in analizador.COLUMNAS_MEDIDAS, campo