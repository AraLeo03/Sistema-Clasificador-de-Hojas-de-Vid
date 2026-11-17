"""Pruebas de knnQt.py con Qt (sin ventanas): hilo de ejecucion y limpieza de temporales."""

import os

import pytest

from conftest import FILAS_SINTETICAS, escribir_csv

knnQt = pytest.importorskip("knnQt")


def _hilo(ruta_csv, carpeta, split=70, k=3):
    temp = (str(carpeta / "train.csv"), str(carpeta / "test.csv"),
            str(carpeta / "pred.csv"), str(carpeta / "met"))
    hilo = knnQt.HiloEjecucionKNN(str(ruta_csv), k, split, [0, 1], [1.0, 1.0], "normalizar", temp)
    resultado = {}
    hilo.finalizado.connect(lambda *a: resultado.update(ok=a))
    hilo.error.connect(lambda mensaje: resultado.update(error=mensaje))
    return hilo, resultado, temp


def test_hilo_ejecuta_todo_el_proceso(qapp, dataset_sintetico, tmp_path):
    hilo, resultado, _ = _hilo(dataset_sintetico, tmp_path)
    hilo.run()                                   # run() directo: sin necesidad de event loop

    assert "error" not in resultado, resultado.get("error")
    salida_knn, salida_est, salida_votos, config, metricas = resultado["ok"]
    assert "CLASIFICACION COMPLETADA" in salida_knn
    assert config == {"k": 3, "split": 70, "escalado": "normalizar"}
    assert set(metricas["clases"]) == {"A", "B", "C"}
    assert 0.0 <= metricas["accuracy"] <= 1.0


def test_split_estratificado_70_30(qapp, dataset_sintetico, tmp_path):
    """3 filas por clase al 70 % -> round(2.1)=2 en train y 1 en test, por clase."""
    hilo, _, (train, test, _, _) = _hilo(dataset_sintetico, tmp_path)
    hilo.run()
    filas_train = open(train, encoding="utf-8").read().split()[1:]
    filas_test = open(test, encoding="utf-8").read().split()[1:]
    clases = lambda filas: sorted(f.split(",")[-1] for f in filas)
    assert clases(filas_train) == ["A", "A", "B", "B", "C", "C"]
    assert clases(filas_test) == ["A", "B", "C"]
    assert sorted(filas_train + filas_test) == sorted(",".join(f) for f in FILAS_SINTETICAS)   # sin perder ni repetir


def test_split_es_reproducible(qapp, dataset_sintetico, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    ha, _, (ta, *_ ) = _hilo(dataset_sintetico, a)
    hb, _, (tb, *_ ) = _hilo(dataset_sintetico, b)
    ha.run(); hb.run()
    assert open(ta, encoding="utf-8").read() == open(tb, encoding="utf-8").read()


def test_hilo_dataset_solo_con_encabezado_emite_error(qapp, tmp_path):
    vacio = tmp_path / "vacio.csv"
    vacio.write_text("x,y,clase\n", encoding="utf-8")
    hilo, resultado, _ = _hilo(vacio, tmp_path)
    hilo.run()
    assert "vacío" in resultado["error"] or "vacio" in resultado["error"]


def test_hilo_archivo_inexistente_emite_error(qapp, tmp_path):
    hilo, resultado, _ = _hilo(tmp_path / "no_existe.csv", tmp_path)
    hilo.run()
    assert "error" in resultado


def test_panel_crea_los_temporales_en_su_propia_carpeta_y_los_borra(qapp):
    panel = knnQt.PanelKNN()
    carpeta = panel.dir_temp
    try:
        assert os.path.isdir(carpeta)
        for ruta in (panel.TEMP_TRAIN, panel.TEMP_TEST, panel.TEMP_PRED, panel.TEMP_MET):
            assert os.path.dirname(ruta) == carpeta          # nada en la carpeta del proyecto

        with open(panel.TEMP_TRAIN, "w", encoding="utf-8") as f:
            f.write("x,y,clase\n1,1,A\n")
        panel.limpiar_temporales()
        assert not os.path.exists(carpeta)
        panel.limpiar_temporales()                           # llamarla otra vez no debe fallar
    finally:
        panel.limpiar_temporales()


def test_dos_paneles_no_comparten_carpeta_temporal(qapp):
    p1, p2 = knnQt.PanelKNN(), knnQt.PanelKNN()
    try:
        assert p1.dir_temp != p2.dir_temp
    finally:
        p1.limpiar_temporales(); p2.limpiar_temporales()