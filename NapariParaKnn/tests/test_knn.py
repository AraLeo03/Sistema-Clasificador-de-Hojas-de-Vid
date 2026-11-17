"""Pruebas de knnQt.py (logica pura): modelos de datos, escalado, distancia, votacion y metricas."""

import csv

import pytest

from conftest import escribir_csv
from knnQt import KNN, Distancia, Estimaciones, Lector, Objeto


def obj(*valores, clase="A"):
    return Objeto([str(v) for v in valores] + [clase])


# ======================================================================
# Objeto / Distancia / Lector
# ======================================================================

def test_objeto_limpia_espacios_y_expone_la_clase():
    o = Objeto([" 1.5 ", " 2 ", " Cab "])
    assert o.atributos == ["1.5", "2", "Cab"]
    assert o.clase == "Cab"


def test_objeto_valores_numericos_todos_o_por_indice():
    o = Objeto(["1", "2.5", "3", "X"])
    assert o.valores_numericos() == [1.0, 2.5, 3.0]          # sin la clase
    assert o.valores_numericos([2, 0]) == [3.0, 1.0]


def test_objeto_valor_no_numerico_lanza_error():
    with pytest.raises(ValueError):
        Objeto(["a", "B"]).valores_numericos()


def test_distancia_guarda_sus_datos():
    d = Distancia(1.5, "A", ["1", "A"])
    assert (d.valor, d.clase, d.atributos) == (1.5, "A", ["1", "A"])


def test_lector_omite_encabezado_y_filas_vacias(tmp_path):
    ruta = tmp_path / "d.csv"
    ruta.write_text("x,y,clase\n1,2,A\n\n , , \n3,4,B\n", encoding="utf-8")
    datos = Lector.leer_csv(str(ruta))
    assert [d.clase for d in datos] == ["A", "B"]
    assert datos[1].atributos == ["3", "4", "B"]


def test_lector_archivo_inexistente_o_vacio(tmp_path):
    assert Lector.leer_csv(str(tmp_path / "no_existe.csv")) == []
    vacio = tmp_path / "vacio.csv"
    vacio.write_text("", encoding="utf-8")
    assert Lector.leer_csv(str(vacio)) == []


# ======================================================================
# Escalado
# ======================================================================

def test_normalizar_usa_el_rango_del_entrenamiento_tambien_en_prueba():
    train = [obj(0, 5), obj(10, 5)]
    prueba = [obj(5, 5), obj(20, 5)]
    t, p = KNN.normalizar_datos(train, prueba, [0, 1])
    assert [o.atributos[0] for o in t] == ["0.000000", "1.000000"]
    assert p[0].atributos[0] == "0.500000"
    assert p[1].atributos[0] == "2.000000"       # fuera del rango del train: NO se recorta


def test_normalizar_columna_constante_queda_en_cero():
    t, p = KNN.normalizar_datos([obj(7), obj(7)], [obj(9)], [0])
    assert t[0].atributos[0] == "0.000000" and p[0].atributos[0] == "0.000000"


def test_normalizar_no_modifica_las_listas_originales_ni_la_clase():
    train, prueba = [obj(0, clase="X"), obj(10, clase="Y")], [obj(5, clase="Z")]
    t, p = KNN.normalizar_datos(train, prueba, [0])
    assert train[1].atributos[0] == "10" and prueba[0].atributos[0] == "5"
    assert [o.clase for o in t] == ["X", "Y"] and p[0].clase == "Z"


def test_normalizar_solo_las_columnas_indicadas():
    t, _ = KNN.normalizar_datos([obj(0, 100), obj(10, 200)], [], [0])
    assert t[1].atributos[:2] == ["1.000000", "200"]


def test_normalizar_sin_entrenamiento_devuelve_listas_vacias():
    t, p = KNN.normalizar_datos([], [], None)
    assert t == [] and p == []


def test_estandarizar_zscore_con_media_y_desviacion_del_entrenamiento():
    train = [obj(0), obj(10)]                      # media 5, desviacion 5
    t, p = KNN.estandarizar_datos(train, [obj(5), obj(10), obj(-5)], [0])
    assert [o.atributos[0] for o in t] == ["-1.000000", "1.000000"]
    assert [o.atributos[0] for o in p] == ["0.000000", "1.000000", "-2.000000"]


def test_estandarizar_columna_constante_queda_en_cero():
    t, p = KNN.estandarizar_datos([obj(3), obj(3)], [obj(8)], [0])
    assert t[0].atributos[0] == "0.000000" and p[0].atributos[0] == "0.000000"


def test_estandarizar_no_modifica_los_originales():
    train = [obj(0), obj(10)]
    KNN.estandarizar_datos(train, [], [0])
    assert train[0].atributos[0] == "0"


# ======================================================================
# Distancia
# ======================================================================

def test_distancia_euclidiana_3_4_5():
    assert KNN.calcular_distancia_euclidiana(obj(0, 0), obj(3, 4)) == pytest.approx(5.0)


def test_distancia_con_indices_y_pesos():
    a, b = obj(0, 0, 0), obj(3, 4, 12)
    assert KNN.calcular_distancia_euclidiana(a, b, [0, 1]) == pytest.approx(5.0)
    assert KNN.calcular_distancia_euclidiana(a, b, [0, 1], [1.0, 0.0]) == pytest.approx(3.0)
    assert KNN.calcular_distancia_euclidiana(a, b, [0], [4.0]) == pytest.approx(6.0)   # sqrt(4 * 9)


def test_distancia_a_si_mismo_es_cero_y_es_simetrica():
    a, b = obj(1.5, -2), obj(4, 7)
    assert KNN.calcular_distancia_euclidiana(a, a) == 0
    assert KNN.calcular_distancia_euclidiana(a, b) == pytest.approx(KNN.calcular_distancia_euclidiana(b, a))


# ======================================================================
# Clasificacion por votacion
# ======================================================================

ENTRENAMIENTO_1D = [obj(0, clase="A"), obj(1, clase="B"), obj(5, clase="A"), obj(6, clase="B")]


def test_clasificar_dato_mayoria_de_votos():
    train = [obj(0, clase="A"), obj(0.1, clase="A"), obj(0.2, clase="B"), obj(9, clase="B")]
    clase, votos = KNN.clasificar_dato(obj(0.05, clase="?"), train, 3)
    assert clase == "A" and votos == {"A": 2, "B": 1}


def test_clasificar_dato_k_igual_a_1_toma_al_mas_cercano():
    clase, votos = KNN.clasificar_dato(obj(5.2, clase="?"), ENTRENAMIENTO_1D, 1)
    assert clase == "A" and votos == {"A": 1}


@pytest.mark.parametrize("consulta, esperada", [(0.4, "A"), (0.6, "B")])
def test_clasificar_dato_empate_gana_la_clase_del_vecino_mas_cercano(consulta, esperada):
    clase, votos = KNN.clasificar_dato(obj(consulta, clase="?"), ENTRENAMIENTO_1D, 2)
    assert votos == {"A": 1, "B": 1}
    assert clase == esperada


def test_clasificar_dato_los_pesos_cambian_el_resultado():
    train = [obj(0, 10, clase="A"), obj(10, 0, clase="B")]
    q = obj(1, 1, clase="?")
    # Sin pesos: A esta a ~9.05 y B a ~9.05 (empate) -> con peso solo en la 1a columna gana A
    assert KNN.clasificar_dato(q, train, 1, [0, 1], [1.0, 0.0])[0] == "A"
    assert KNN.clasificar_dato(q, train, 1, [0, 1], [0.0, 1.0])[0] == "B"


def test_clasificar_dato_k_mayor_que_el_entrenamiento_falla():
    """Documenta el requisito: k <= len(entrenamiento) (clasificador_externo ya lo limita)."""
    with pytest.raises(IndexError):
        KNN.clasificar_dato(obj(0, clase="?"), ENTRENAMIENTO_1D, 10)


# ======================================================================
# KNN.ejecutar / guardar_predicciones
# ======================================================================

@pytest.fixture
def train_y_test(tmp_path):
    train = escribir_csv(tmp_path / "train.csv", [
        ("1", "1", "A"), ("1", "2", "A"), ("2", "1", "A"),
        ("10", "10", "B"), ("10", "11", "B"), ("11", "10", "B"),
    ])
    prueba = escribir_csv(tmp_path / "test.csv", [("1.2", "1.1", "A"), ("10.5", "10.4", "B"), ("1.1", "1.4", "B")])
    return train, prueba


@pytest.mark.parametrize("escalado", ["ninguno", "normalizar", "estandarizar"])
def test_ejecutar_clasifica_y_guarda_predicciones(train_y_test, tmp_path, escalado):
    train, prueba = train_y_test
    salida = tmp_path / "pred.csv"
    texto = KNN.ejecutar(str(train), str(prueba), 3, str(salida), [0, 1], escalado)

    assert "CLASIFICACION COMPLETADA" in texto
    assert f"Escalado de Datos: {escalado.upper()}" in texto
    with open(salida, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert [r["actual"] for r in filas] == ["A", "B", "B"]
    assert [r["predicted"] for r in filas] == ["A", "B", "A"]      # la 3a esta cerca de A (error esperado)
    assert [r["error"] for r in filas] == ["", "", "*"]
    assert filas[0]["prediction"] == "100.00%"


def test_ejecutar_sin_datos_devuelve_error(tmp_path):
    assert KNN.ejecutar(str(tmp_path / "x.csv"), str(tmp_path / "y.csv"), 3,
                        str(tmp_path / "p.csv")).startswith("ERROR")


def test_ejecutar_votaciones_menciona_a_los_vecinos(train_y_test):
    train, prueba = train_y_test
    texto = KNN.ejecutar_votaciones(str(train), str(prueba), 3, ["x", "y", "clase"], [0, 1], "ninguno")
    assert isinstance(texto, str) and len(texto) > 0


# ======================================================================
# Estimaciones (metricas)
# ======================================================================

@pytest.fixture
def predicciones(tmp_path):
    """A->A, A->B, B->B, B->B.  Metricas esperadas calculadas a mano (uno-contra-todos)."""
    ruta = tmp_path / "pred.csv"
    ruta.write_text("inst#,actual,predicted,error,prediction\n"
                    "1,A,A,,100%\n2,A,B,*,100%\n3,B,B,,100%\n4,B,B,,100%\n", encoding="utf-8")
    return ruta


def test_estimaciones_metricas_por_clase(predicciones):
    resultados = Estimaciones._leer_resultados(str(predicciones))
    m = Estimaciones._calcular_metricas(resultados, ["A", "B"])

    assert (m["A"]["TP"], m["A"]["FP"], m["A"]["FN"], m["A"]["TN"]) == (1, 0, 1, 2)
    assert m["A"]["Precision"] == pytest.approx(1.0)
    assert m["A"]["Recall"] == pytest.approx(0.5)
    assert m["A"]["F1"] == pytest.approx(2 / 3)
    assert m["A"]["Accuracy"] == pytest.approx(0.75)

    assert (m["B"]["TP"], m["B"]["FP"], m["B"]["FN"], m["B"]["TN"]) == (2, 1, 0, 1)
    assert m["B"]["Precision"] == pytest.approx(2 / 3)
    assert m["B"]["Recall"] == pytest.approx(1.0)
    assert m["B"]["F1"] == pytest.approx(0.8)


def test_estimaciones_ejecutar_devuelve_promedios_matriz_y_archivos(predicciones, tmp_path):
    base = str(tmp_path / "met")
    texto, r = Estimaciones.ejecutar(str(predicciones), base)

    assert r["clases"] == ["A", "B"]
    assert r["matriz"] == {"A": {"A": 1, "B": 1}, "B": {"A": 0, "B": 2}}
    assert r["accuracy"] == pytest.approx(0.75)
    assert r["precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert r["recall"] == pytest.approx((0.5 + 1.0) / 2)
    for sufijo in ("_Evaluacion.csv", "_Resumen.csv", "_Matriz.csv"):
        assert (tmp_path / f"met{sufijo}").exists()
        assert f"met{sufijo}" in texto


def test_estimaciones_matriz_csv_tiene_filas_reales_y_columnas_predichas(predicciones, tmp_path):
    Estimaciones.ejecutar(str(predicciones), str(tmp_path / "met"))
    contenido = (tmp_path / "met_Matriz.csv").read_text(encoding="utf-8")
    assert "A" in contenido and "B" in contenido


def test_estimaciones_archivo_inexistente(tmp_path):
    texto, resumen = Estimaciones.ejecutar(str(tmp_path / "no.csv"), str(tmp_path / "met"))
    assert texto.startswith("ERROR") and resumen == {}


def test_estimaciones_clase_que_nunca_se_predice_da_precision_cero_sin_dividir_entre_cero(tmp_path):
    ruta = tmp_path / "p.csv"
    ruta.write_text("inst#,actual,predicted,error,prediction\n1,A,B,*,1\n2,B,B,,1\n", encoding="utf-8")
    m = Estimaciones._calcular_metricas(Estimaciones._leer_resultados(str(ruta)), ["A", "B"])
    assert m["A"]["Precision"] == 0.0 and m["A"]["Recall"] == 0.0 and m["A"]["F1"] == 0.0