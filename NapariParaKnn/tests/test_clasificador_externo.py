"""Pruebas de clasificador_externo.py: carga del dataset, configuracion y clasificacion."""

import pytest

import clasificador_externo as ce
from conftest import ESCALA_EJEMPLO, PUNTOS_EJEMPLO, escribir_csv


@pytest.fixture
def config_sintetica(monkeypatch):
    """Dataset de 2 atributos (L1, L2) y constantes del modulo ajustadas a el."""
    monkeypatch.setattr(ce, "DATASET_EQUIVALENCIAS", {"L1": "L1_cm", "L2": "L2_cm"})
    monkeypatch.setattr(ce, "KNN_K", 3)
    monkeypatch.setattr(ce, "KNN_ESCALADO", "normalizar")
    monkeypatch.setattr(ce, "KNN_ATRIBUTOS_USADOS", None)
    monkeypatch.setattr(ce, "KNN_PESOS", {})
    monkeypatch.setattr(ce, "KNN_PESO_POR_DEFECTO", 1.0)


@pytest.fixture
def csv_hojas(tmp_path):
    """3 clases separadas en el plano (L1, L2)."""
    filas = [("1", "1", "A"), ("1", "2", "A"), ("2", "1", "A"),
             ("10", "10", "B"), ("10", "11", "B"), ("11", "10", "B"),
             ("1", "10", "C"), ("1", "11", "C"), ("2", "10", "C")]
    return escribir_csv(tmp_path / "hojas.csv", filas, encabezado=("L1", "L2", "clase"))


# ---------------------------------------------------------------- cargar_dataset

def test_cargar_dataset_devuelve_encabezados_y_objetos(csv_hojas):
    encabezados, datos = ce.cargar_dataset(csv_hojas)
    assert encabezados == ["L1", "L2", "clase"]
    assert len(datos) == 9 and datos[0].clase == "A"


def test_cargar_dataset_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATASET_RUTA"):
        ce.cargar_dataset(tmp_path / "no_existe.csv")


def test_cargar_dataset_solo_encabezado(tmp_path):
    ruta = escribir_csv(tmp_path / "vacio.csv", [], encabezado=("L1", "L2", "clase"))
    with pytest.raises(ValueError, match="vacio"):
        ce.cargar_dataset(ruta)


def test_cargar_dataset_sin_atributos(tmp_path):
    ruta = tmp_path / "una_columna.csv"
    ruta.write_text("clase\nA\nB\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ce.cargar_dataset(ruta)


# ---------------------------------------------------------------- _configuracion

def test_configuracion_por_defecto_usa_todas_las_columnas(config_sintetica):
    assert ce._configuracion(["L1", "L2"]) == ([0, 1], [1.0, 1.0])


def test_configuracion_atributos_y_pesos(config_sintetica, monkeypatch):
    monkeypatch.setattr(ce, "KNN_ATRIBUTOS_USADOS", ["L2"])
    monkeypatch.setattr(ce, "KNN_PESOS", {"L2": 2.5})
    assert ce._configuracion(["L1", "L2"]) == ([1], [2.5])


def test_configuracion_columna_desconocida(config_sintetica, monkeypatch):
    monkeypatch.setattr(ce, "KNN_ATRIBUTOS_USADOS", ["L9"])
    with pytest.raises(ValueError, match="L9"):
        ce._configuracion(["L1", "L2"])


def test_configuracion_sin_atributos_seleccionados(config_sintetica, monkeypatch):
    monkeypatch.setattr(ce, "KNN_ATRIBUTOS_USADOS", [])
    with pytest.raises(ValueError):
        ce._configuracion(["L1", "L2"])


# ---------------------------------------------------------------- clasificar

@pytest.mark.parametrize("escalado", ["ninguno", "normalizar", "estandarizar"])
@pytest.mark.parametrize("medidas, esperada", [
    ({"L1_cm": 1.3, "L2_cm": 1.4}, "A"),
    ({"L1_cm": 10.4, "L2_cm": 10.6}, "B"),
    ({"L1_cm": 1.4, "L2_cm": 10.3}, "C"),
])
def test_clasificar_acierta_en_clusters_separados(config_sintetica, csv_hojas, monkeypatch,
                                                 escalado, medidas, esperada):
    monkeypatch.setattr(ce, "KNN_ESCALADO", escalado)
    r = ce.clasificar(medidas, csv_hojas)
    assert r["clase"] == esperada
    assert r["confianza"] == pytest.approx(100.0)
    assert r["votos"] == {esperada: 3}


def test_clasificar_estructura_del_resultado(config_sintetica, csv_hojas):
    r = ce.clasificar({"L1_cm": 1.3, "L2_cm": 1.4}, csv_hojas)
    assert r["k"] == 3 and r["escalado"] == "normalizar"
    assert r["columnas_usadas"] == ["L1", "L2"]
    assert len(r["vecinos"]) == 3
    assert r["vecinos"] == sorted(r["vecinos"], key=lambda v: v["distancia"])     # del mas cercano al mas lejano
    assert [a["nombre"] for a in r["atributos"]] == ["L1", "L2"]
    assert float(r["atributos"][0]["valor"]) == pytest.approx(1.3)


def test_clasificar_ignora_medidas_que_el_dataset_no_usa(config_sintetica, csv_hojas):
    medidas = {"L1_cm": 1.3, "L2_cm": 1.4, "A1_grados": 99.0, "L1_px": 5000}
    assert ce.clasificar(medidas, csv_hojas)["clase"] == "A"


def test_clasificar_falta_una_medida(config_sintetica, csv_hojas):
    with pytest.raises(ValueError, match="L2"):
        ce.clasificar({"L1_cm": 1.3}, csv_hojas)


def test_clasificar_escalado_invalido(config_sintetica, csv_hojas, monkeypatch):
    monkeypatch.setattr(ce, "KNN_ESCALADO", "logaritmico")
    with pytest.raises(ValueError, match="KNN_ESCALADO"):
        ce.clasificar({"L1_cm": 1, "L2_cm": 1}, csv_hojas)


def test_clasificar_limita_k_al_tamano_del_entrenamiento(config_sintetica, csv_hojas, monkeypatch):
    monkeypatch.setattr(ce, "KNN_K", 100)
    r = ce.clasificar({"L1_cm": 1.3, "L2_cm": 1.4}, csv_hojas)
    assert r["k"] == 9 and sum(r["votos"].values()) == 9


def test_clasificar_k_cero_o_negativo_se_sube_a_1(config_sintetica, csv_hojas, monkeypatch):
    monkeypatch.setattr(ce, "KNN_K", 0)
    assert ce.clasificar({"L1_cm": 1.3, "L2_cm": 1.4}, csv_hojas)["k"] == 1


def test_clasificar_con_un_solo_atributo(config_sintetica, csv_hojas, monkeypatch):
    monkeypatch.setattr(ce, "KNN_ATRIBUTOS_USADOS", ["L1"])
    r = ce.clasificar({"L1_cm": 10.4}, csv_hojas)          # L2_cm no hace falta si no se usa
    assert r["clase"] == "B" and r["columnas_usadas"] == ["L1"]


def test_clasificar_los_pesos_pueden_cambiar_la_clase(config_sintetica, csv_hojas, monkeypatch):
    # (6, 6) queda "entre" A, B y C; el peso decide a quien se parece
    monkeypatch.setattr(ce, "KNN_K", 1)
    monkeypatch.setattr(ce, "KNN_ESCALADO", "ninguno")
    monkeypatch.setattr(ce, "KNN_PESOS", {"L1": 1.0, "L2": 0.0})
    por_l1 = ce.clasificar({"L1_cm": 9.0, "L2_cm": 1.0}, csv_hojas)["clase"]
    monkeypatch.setattr(ce, "KNN_PESOS", {"L1": 0.0, "L2": 1.0})
    por_l2 = ce.clasificar({"L1_cm": 9.0, "L2_cm": 1.0}, csv_hojas)["clase"]
    assert por_l1 == "B" and por_l2 == "A"


def test_clasificar_usa_las_equivalencias_de_nombres(csv_hojas, monkeypatch):
    monkeypatch.setattr(ce, "DATASET_EQUIVALENCIAS", {"L1": "ancho", "L2": "alto"})
    monkeypatch.setattr(ce, "KNN_ATRIBUTOS_USADOS", None)
    monkeypatch.setattr(ce, "KNN_PESOS", {})
    r = ce.clasificar({"ancho": 10.4, "alto": 10.6}, csv_hojas)
    assert r["clase"] == "B"


# ---------------------------------------------------------------- formatear_resultado

def test_formatear_resultado_incluye_clase_confianza_y_vecinos(config_sintetica, csv_hojas):
    r = ce.clasificar({"L1_cm": 1.3, "L2_cm": 1.4}, csv_hojas)
    texto = ce.formatear_resultado(r)
    assert "CLASE PREDICHA : A" in texto
    assert "100.00% (3/3 votos)" in texto
    assert "3 vecinos mas cercanos" in texto
    assert "| L1 | L2 | Clase | Distancia |" in texto


# ---------------------------------------------------------------- con el dataset REAL del proyecto

@pytest.mark.skipif(not ce.DATASET_RUTA.exists(), reason="no esta data/dataset_70_30.csv")
def test_con_el_dataset_real_y_las_medidas_de_analizador():
    import analizador
    medidas = analizador.calcular_local(PUNTOS_EJEMPLO, ESCALA_EJEMPLO)
    _, datos = ce.cargar_dataset()
    clases = {d.clase for d in datos}
    r = ce.clasificar(medidas)
    assert r["clase"] in clases
    assert r["confianza"] == pytest.approx(r["votos"][r["clase"]] / sum(r["votos"].values()) * 100)
    assert len(r["vecinos"]) == r["k"]


@pytest.mark.skipif(not ce.DATASET_RUTA.exists(), reason="no esta data/dataset_70_30.csv")
def test_dataset_real_tiene_las_columnas_que_pide_la_equivalencia():
    """Si cambias los encabezados del CSV y olvidas DATASET_EQUIVALENCIAS, esta prueba lo avisa."""
    encabezados, _ = ce.cargar_dataset()
    medidas_validas = set(ce.MEDIDAS_DISPONIBLES)
    for columna in encabezados[:-1]:
        assert ce.DATASET_EQUIVALENCIAS.get(columna, columna) in medidas_validas, columna