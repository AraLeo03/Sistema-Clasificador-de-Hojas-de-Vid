"""Pruebas de dialogo_ia.py (ventana de espera), con Qt sin pantalla."""

import pytest

pytest.importorskip("qtpy")
from dialogo_ia import EN_CURSO, FALLO, LISTO, PENDIENTE, DialogoAnalisisIA


@pytest.fixture
def dialogo(qapp):
    d = DialogoAnalisisIA(None, "cab_1.jpeg", timeout_s=60)
    yield d
    d.reloj.stop()
    d.close()


def _simbolo(dialogo, paso):
    return dialogo.pasos[paso].text().split()[0]


def test_estado_inicial_de_los_tres_pasos(dialogo):
    assert [_simbolo(dialogo, i) for i in range(3)] == [LISTO, EN_CURSO, PENDIENTE]
    assert dialogo.cancelado is False and dialogo.terminado is False
    assert dialogo.boton.text() == "Cancelar"


def test_etiqueta_de_tiempo_muestra_el_timeout(dialogo):
    assert "0 s" in dialogo.lbl_tiempo.text() and "60 s" in dialogo.lbl_tiempo.text()


def test_cada_segundo_aumenta_el_tiempo(dialogo):
    dialogo._tic(); dialogo._tic()
    assert dialogo.segundos == 2
    assert "2 s" in dialogo.lbl_tiempo.text()


def test_mostrar_arranca_el_reloj(dialogo):
    dialogo.mostrar()
    assert dialogo.reloj.isActive()


def test_ia_lista_pasa_al_paso_del_knn_y_muestra_las_medidas(dialogo):
    dialogo.ia_lista("L1_cm: 13.09")
    assert [_simbolo(dialogo, i) for i in range(3)] == [LISTO, LISTO, EN_CURSO]
    assert not dialogo.salida.isHidden()
    assert "L1_cm: 13.09" in dialogo.salida.toPlainText()


def test_mostrar_resultado_exitoso(dialogo):
    dialogo.mostrar_resultado("medidas", "clasificacion", clase="Cab", confianza=66.7)
    assert [_simbolo(dialogo, i) for i in range(3)] == [LISTO, LISTO, LISTO]
    assert "Cab" in dialogo.lbl_clase.text() and "67%" in dialogo.lbl_clase.text()
    assert dialogo.salida.toPlainText() == "medidas\n\nclasificacion"
    assert dialogo.terminado and dialogo.boton.text() == "Cerrar"
    assert not dialogo.reloj.isActive() and dialogo.barra.isHidden()


def test_mostrar_resultado_sin_clase_marca_fallo_en_el_knn(dialogo):
    dialogo.mostrar_resultado("medidas", "dataset no encontrado")
    assert _simbolo(dialogo, 2) == FALLO
    assert dialogo.lbl_clase.isHidden()


def test_mostrar_error_marca_el_paso_y_muestra_el_mensaje(dialogo):
    dialogo.mostrar_error("Ningun proveedor de IA respondio")
    assert _simbolo(dialogo, 1) == FALLO
    assert "No se pudo completar" in dialogo.lbl_clase.text()
    assert dialogo.salida.toPlainText() == "Ningun proveedor de IA respondio"
    assert dialogo.terminado


def test_cerrar_antes_de_terminar_cuenta_como_cancelar(dialogo):
    dialogo.reject()
    assert dialogo.cancelado is True


def test_cerrar_despues_de_terminar_no_es_cancelar(dialogo):
    dialogo.mostrar_resultado("m", "c", clase="Cab", confianza=100)
    dialogo.reject()
    assert dialogo.cancelado is False