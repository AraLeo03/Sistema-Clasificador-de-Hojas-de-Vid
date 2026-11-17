"""Pruebas de analizador.py: lectura de archivos y calculo local (longitudes y angulos)."""

import pytest

import analizador
from conftest import ESCALA_EJEMPLO, PUNTOS_EJEMPLO, texto_coordenadas_anotador


# ---------------------------------------------------------------- vector / angulo

def test_vector_es_punta_menos_seno(puntos_ejemplo):
    dx, dy = analizador.vector(puntos_ejemplo, "L1")
    assert dx == pytest.approx(-113.22, abs=0.01)
    assert dy == pytest.approx(-759.35, abs=0.01)


@pytest.mark.parametrize("a, b, esperado", [
    ((0, -1), (1, 0), 90),      # arriba -> derecha: giro horario en pantalla (y hacia abajo)
    ((1, 0), (0, -1), 270),     # derecha -> arriba: horario seria 270
    ((1, 0), (0, 1), 90),       # derecha -> abajo
    ((1, 0), (-1, 0), 180),
    ((1, 0), (1, 0), 0),
])
def test_angulo_horario_casos_conocidos(a, b, esperado):
    assert analizador.angulo_horario(a, b) == pytest.approx(esperado)


def test_angulo_horario_siempre_entre_0_y_360():
    for a in [(1, 0), (0, 1), (-1, 0), (0, -1), (3, 4)]:
        for b in [(1, 0), (0, 1), (-1, 0), (0, -1), (-3, 4)]:
            assert 0 <= analizador.angulo_horario(a, b) < 360


# ---------------------------------------------------------------- calcular_local

def test_calcular_local_con_el_ejemplo_del_contexto(puntos_ejemplo):
    r = analizador.calcular_local(puntos_ejemplo, ESCALA_EJEMPLO)
    # Valores del PARTE 7 del archivo de contexto
    assert r["L1_px"] == pytest.approx(767.74, abs=0.01)
    assert r["L2_px"] == pytest.approx(691.05, abs=0.01)
    assert r["L3_px"] == pytest.approx(473.80, abs=0.01)
    assert r["L4_px"] == pytest.approx(349.91, abs=0.01)
    assert r["L1_cm"] == pytest.approx(13.09, abs=0.01)
    assert r["L4_cm"] == pytest.approx(5.96, abs=0.01)
    assert r["A1_grados"] == pytest.approx(59.05, abs=0.01)
    assert r["A2_grados"] == pytest.approx(120.85, abs=0.01)
    assert r["A3_grados"] == pytest.approx(164.10, abs=0.01)
    assert r["A2_3_grados"] == pytest.approx(61.80, abs=0.01)
    assert r["A2_4_grados"] == pytest.approx(105.05, abs=0.01)


def test_calcular_local_sin_escala_no_devuelve_cm(puntos_ejemplo):
    r = analizador.calcular_local(puntos_ejemplo)
    assert "L1_px" in r and "A1_grados" in r
    assert not any(k.endswith("_cm") for k in r)


def test_calcular_local_escala_cero_o_none_no_da_cm(puntos_ejemplo):
    assert "L1_cm" not in analizador.calcular_local(puntos_ejemplo, 0)
    assert "L1_cm" not in analizador.calcular_local(puntos_ejemplo, None)


def test_angulos_son_coherentes_entre_si(puntos_ejemplo):
    """A2 = A1 + A2-3 y A3 = A1 + A2-4 (las comprobaciones del archivo de contexto)."""
    r = analizador.calcular_local(puntos_ejemplo, ESCALA_EJEMPLO)
    assert r["A2_grados"] == pytest.approx(r["A1_grados"] + r["A2_3_grados"], abs=0.1)
    assert r["A3_grados"] == pytest.approx(r["A1_grados"] + r["A2_4_grados"], abs=0.1)
    assert r["A1_grados"] < r["A2_grados"] < r["A3_grados"]


def test_la_escala_dobla_los_px_y_divide_los_cm(puntos_ejemplo):
    a = analizador.calcular_local(puntos_ejemplo, 50)
    b = analizador.calcular_local(puntos_ejemplo, 100)
    assert a["L1_px"] == pytest.approx(b["L1_px"])
    assert a["L1_cm"] == pytest.approx(2 * b["L1_cm"])


def test_calcular_local_nombres_de_las_claves(puntos_ejemplo):
    r = analizador.calcular_local(puntos_ejemplo, ESCALA_EJEMPLO)
    esperadas = {f"L{i}_{u}" for i in range(1, 5) for u in ("px", "cm")}
    esperadas |= {nombre for nombre, _, _ in analizador.ANGULOS}
    assert set(r) == esperadas


# ---------------------------------------------------------------- leer_coordenadas

def test_leer_coordenadas_formato_del_anotador(tmp_path):
    ruta = tmp_path / "cab_1_coordenadas.txt"
    ruta.write_text(texto_coordenadas_anotador(), encoding="utf-8")
    texto, puntos = analizador.leer_coordenadas(ruta)
    assert set(puntos) == set(analizador.NOMBRES_PUNTOS)
    assert puntos["seno_peciolar"] == (584.24, 964.21)
    assert "landmark, x, y" in texto          # el texto completo se conserva para mandarlo a la IA


def test_leer_coordenadas_faltan_puntos(tmp_path):
    incompletos = {k: v for k, v in PUNTOS_EJEMPLO.items() if k != "punta_L3"}
    ruta = tmp_path / "x_coordenadas.txt"
    ruta.write_text(texto_coordenadas_anotador(incompletos), encoding="utf-8")
    with pytest.raises(ValueError, match="punta_L3"):
        analizador.leer_coordenadas(ruta)


def test_leer_coordenadas_valor_no_numerico(tmp_path):
    ruta = tmp_path / "x_coordenadas.txt"
    ruta.write_text("seno_peciolar, abc, 10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        analizador.leer_coordenadas(ruta)


# ---------------------------------------------------------------- encontrar_archivos

def test_encontrar_archivos_completo(tmp_path):
    for nombre in ("a_anotada.jpg", "a_vertical.jpg", "a_coordenadas.txt"):
        (tmp_path / nombre).write_bytes(b"x")
    anotada, limpia, coords = analizador.encontrar_archivos(tmp_path)
    assert anotada.name == "a_anotada.jpg"
    assert limpia.name == "a_vertical.jpg"
    assert coords.name == "a_coordenadas.txt"


def test_encontrar_archivos_sin_imagen_limpia(tmp_path):
    for nombre in ("a_anotada.jpg", "a_coordenadas.txt"):
        (tmp_path / nombre).write_bytes(b"x")
    _, limpia, _ = analizador.encontrar_archivos(tmp_path)
    assert limpia is None


@pytest.mark.parametrize("faltante", ["a_anotada.jpg", "a_coordenadas.txt"])
def test_encontrar_archivos_faltan_obligatorios(tmp_path, faltante):
    for nombre in {"a_anotada.jpg", "a_coordenadas.txt"} - {faltante}:
        (tmp_path / nombre).write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        analizador.encontrar_archivos(tmp_path)