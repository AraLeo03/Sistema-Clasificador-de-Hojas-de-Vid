"""Pruebas de analizador.py: configuracion de proveedores, respaldo Gemini/Claude y analizar_hoja.

Ninguna prueba llama a la red: los SDK se reemplazan por dobles y las claves vienen de monkeypatch.
"""

import base64
import json
import sys
import types

import pytest
from PIL import Image

import analizador
from conftest import BLOQUE_OK, ESCALA_EJEMPLO, texto_coordenadas_anotador


# ======================================================================
# cargar_config
# ======================================================================

def _resumen(config):
    return [(p["nombre"], p["modelos"]) for p in config["proveedores"]]


def test_config_por_defecto_gemini_y_luego_claude():
    c = analizador.cargar_config()
    assert [p["nombre"] for p in c["proveedores"]] == ["claude", "gemini"]
    assert c["timeout_s"] == 60.0
    assert c["automatico"] is False


def test_config_claude_principal_gemini_de_respaldo(monkeypatch):
    monkeypatch.setenv("IA_PROVEEDOR_PRINCIPAL", "claude")
    c = analizador.cargar_config()
    assert [p["nombre"] for p in c["proveedores"]] == ["claude", "gemini"]


def test_config_secundario_ninguno_deja_un_solo_proveedor(monkeypatch):
    monkeypatch.setenv("IA_PROVEEDOR_PRINCIPAL", "claude")
    monkeypatch.setenv("IA_PROVEEDOR_SECUNDARIO", "ninguno")
    assert [p["nombre"] for p in analizador.cargar_config()["proveedores"]] == ["claude"]


def test_config_mismo_proveedor_como_principal_y_secundario_no_se_repite(monkeypatch):
    monkeypatch.setenv("IA_PROVEEDOR_PRINCIPAL", "gemini")
    monkeypatch.setenv("IA_PROVEEDOR_SECUNDARIO", "gemini")
    assert [p["nombre"] for p in analizador.cargar_config()["proveedores"]] == ["gemini"]


def test_config_valores_invalidos_usan_los_de_por_defecto(monkeypatch, capsys):
    monkeypatch.setenv("IA_PROVEEDOR_PRINCIPAL", "gpt")
    monkeypatch.setenv("IA_PROVEEDOR_SECUNDARIO", "otro")
    c = analizador.cargar_config()
    assert [p["nombre"] for p in c["proveedores"]] == ["claude", "gemini"]
    assert "Aviso" in capsys.readouterr().out


def test_config_mayusculas_y_espacios_se_toleran(monkeypatch):
    monkeypatch.setenv("IA_PROVEEDOR_PRINCIPAL", "  CLAUDE ")
    assert analizador.cargar_config()["proveedores"][0]["nombre"] == "claude"


def test_config_modelos_de_respaldo_sin_repetir_el_principal(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("CLAUDE_MODELOS_RESPALDO", "claude-haiku-5-5, claude-sonnet-5-5 ,")
    claude = [p for p in analizador.cargar_config()["proveedores"] if p["nombre"] == "claude"][0]
    assert claude["modelos"] == ["claude-sonnet-5-5", "claude-haiku-5-5"]


def test_config_lee_las_api_keys_de_cada_proveedor(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", " g-key ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "c-key")
    claves = {p["nombre"]: p["api_key"] for p in analizador.cargar_config()["proveedores"]}
    assert claves == {"gemini": "g-key", "claude": "c-key"}


def test_config_timeout_nuevo_y_compatibilidad_con_el_antiguo(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT_SEGUNDOS", "90")
    assert analizador.cargar_config()["timeout_s"] == 90.0
    monkeypatch.setenv("IA_TIMEOUT_SEGUNDOS", "30")                 # el nuevo manda
    assert analizador.cargar_config()["timeout_s"] == 30.0


@pytest.mark.parametrize("valor, esperado", [
    ("true", True), ("TRUE", True), ("1", True), ("si", True), ("sí", True), ("yes", True),
    ("false", False), ("0", False), ("", False), ("no", False),
])
def test_config_analisis_automatico(monkeypatch, valor, esperado):
    monkeypatch.setenv("IA_ANALISIS_AUTOMATICO", valor)
    assert analizador.cargar_config()["automatico"] is esperado


# ======================================================================
# llamar_ia (orden de respaldo)
# ======================================================================

def _prov(nombre, key="k"):
    return {"nombre": nombre, "api_key": key, "modelos": ["m"]}


def _falla(mensaje):
    def llamador(*args, **kwargs):
        raise RuntimeError(mensaje)
    return llamador


@pytest.fixture
def llamadores(monkeypatch):
    """Diccionario LLAMADORES editable, restaurado al terminar la prueba."""
    copia = dict(analizador.LLAMADORES)
    monkeypatch.setattr(analizador, "LLAMADORES", copia)
    return copia


def test_llamar_ia_usa_el_principal_y_no_toca_al_secundario(llamadores):
    llamadores["claude"] = lambda *a: ("c1", BLOQUE_OK)
    llamadores["gemini"] = _falla("no debia llamarse")
    proveedor, modelo, texto = analizador.llamar_ia([_prov("claude"), _prov("gemini")], "", "", [])
    assert (proveedor, modelo) == ("claude", "c1")
    assert "INICIO_RESULTADOS" in texto


def test_llamar_ia_pasa_al_secundario_si_el_principal_falla(llamadores):
    llamadores["claude"] = _falla("cuota agotada")
    llamadores["gemini"] = lambda *a: ("g1", BLOQUE_OK)
    assert analizador.llamar_ia([_prov("claude"), _prov("gemini")], "", "", [])[:2] == ("gemini", "g1")


def test_llamar_ia_omite_al_proveedor_sin_api_key(llamadores):
    llamadores["claude"] = _falla("no debia llamarse")
    llamadores["gemini"] = lambda *a: ("g1", BLOQUE_OK)
    assert analizador.llamar_ia([_prov("claude", ""), _prov("gemini")], "", "", [])[0] == "gemini"


def test_llamar_ia_respuesta_sin_bloque_cuenta_como_fallo(llamadores):
    llamadores["claude"] = lambda *a: ("c1", "no puedo medir esta hoja")
    llamadores["gemini"] = lambda *a: ("g1", BLOQUE_OK)
    assert analizador.llamar_ia([_prov("claude"), _prov("gemini")], "", "", [])[0] == "gemini"


def test_llamar_ia_si_todos_fallan_reporta_cada_error(llamadores):
    llamadores["claude"] = _falla("error-claude")
    llamadores["gemini"] = _falla("error-gemini")
    with pytest.raises(RuntimeError) as info:
        analizador.llamar_ia([_prov("claude"), _prov("gemini")], "", "", [])
    mensaje = str(info.value)
    assert "[claude] error-claude" in mensaje and "[gemini] error-gemini" in mensaje


def test_llamar_ia_sin_ninguna_api_key_explica_que_falta(llamadores):
    with pytest.raises(RuntimeError, match="API key"):
        analizador.llamar_ia([_prov("claude", ""), _prov("gemini", "")], "", "", [])


def test_llamar_ia_pasa_los_argumentos_al_proveedor(llamadores):
    recibido = {}

    def llamador(api_key, modelos, contexto, texto, imagenes, timeout):
        recibido.update(api_key=api_key, modelos=modelos, contexto=contexto,
                        texto=texto, imagenes=imagenes, timeout=timeout)
        return "m", BLOQUE_OK

    llamadores["claude"] = llamador
    analizador.llamar_ia([{"nombre": "claude", "api_key": "KEY", "modelos": ["a", "b"]}],
                         "CTX", "TXT", ["i1.jpg"], 12)
    assert recibido == {"api_key": "KEY", "modelos": ["a", "b"], "contexto": "CTX",
                        "texto": "TXT", "imagenes": ["i1.jpg"], "timeout": 12}


# ======================================================================
# llamar_claude (con un SDK de anthropic falso)
# ======================================================================

@pytest.fixture
def anthropic_falso(monkeypatch):
    estado = types.SimpleNamespace(llamadas=[], clientes=[], falla=set(), texto="respuesta de claude")

    class Bloque:
        def __init__(self, tipo, texto):
            self.type, self.text = tipo, texto

    class Respuesta:
        def __init__(self, texto):
            # un bloque que no es texto (por ejemplo 'thinking') debe ignorarse
            self.content = [Bloque("thinking", "IGNORAR"), Bloque("text", texto)]

    class Mensajes:
        def create(self, **kwargs):
            estado.llamadas.append(kwargs)
            if kwargs["model"] in estado.falla:
                raise RuntimeError(f"{kwargs['model']} no disponible")
            return Respuesta(estado.texto)

    class Cliente:
        def __init__(self, api_key, timeout):
            self.api_key, self.timeout, self.messages = api_key, timeout, Mensajes()
            estado.clientes.append(self)

    modulo = types.ModuleType("anthropic")
    modulo.Anthropic = Cliente
    monkeypatch.setitem(sys.modules, "anthropic", modulo)
    return estado


@pytest.fixture
def dos_imagenes(tmp_path):
    rutas = []
    for nombre, color in (("a_anotada.jpg", "red"), ("a_vertical.jpg", "green")):
        ruta = tmp_path / nombre
        Image.new("RGB", (20, 30), color).save(ruta)
        rutas.append(ruta)
    return rutas


def test_llamar_claude_arma_bien_la_peticion(anthropic_falso, dos_imagenes):
    modelo, texto = analizador.llamar_claude("sk-ant-x", ["claude-sonnet-5-5"], "CONTEXTO",
                                             "TEXTO", dos_imagenes, timeout_s=45)
    assert (modelo, texto) == ("claude-sonnet-5-5", "respuesta de claude")

    cliente = anthropic_falso.clientes[0]
    assert cliente.api_key == "sk-ant-x" and cliente.timeout == 45

    peticion = anthropic_falso.llamadas[0]
    assert peticion["model"] == "claude-sonnet-5-5"
    assert peticion["system"] == "CONTEXTO"
    assert peticion["max_tokens"] > 0
    assert "temperature" not in peticion                 # los modelos recientes no la aceptan

    (mensaje,) = peticion["messages"]
    assert mensaje["role"] == "user"
    tipos = [b["type"] for b in mensaje["content"]]
    assert tipos == ["image", "image", "text"]           # imagenes primero, texto al final
    assert mensaje["content"][-1]["text"] == "TEXTO"

    for bloque, ruta in zip(mensaje["content"][:2], dos_imagenes):
        assert bloque["source"]["type"] == "base64"
        assert bloque["source"]["media_type"] == "image/jpeg"
        assert base64.b64decode(bloque["source"]["data"]) == ruta.read_bytes()


def test_llamar_claude_solo_devuelve_los_bloques_de_texto(anthropic_falso, dos_imagenes):
    _, texto = analizador.llamar_claude("k", ["m"], "c", "t", dos_imagenes)
    assert "IGNORAR" not in texto


def test_llamar_claude_prueba_el_siguiente_modelo_si_uno_falla(anthropic_falso, dos_imagenes):
    anthropic_falso.falla = {"modelo-malo"}
    modelo, _ = analizador.llamar_claude("k", ["modelo-malo", "modelo-bueno"], "c", "t", dos_imagenes)
    assert modelo == "modelo-bueno"
    assert [c["model"] for c in anthropic_falso.llamadas] == ["modelo-malo", "modelo-bueno"]


def test_llamar_claude_si_todos_los_modelos_fallan(anthropic_falso, dos_imagenes):
    anthropic_falso.falla = {"a", "b"}
    with pytest.raises(RuntimeError, match="Ningun modelo de Claude"):
        analizador.llamar_claude("k", ["a", "b"], "c", "t", dos_imagenes)


def test_llamar_claude_sin_la_libreria_instalada(monkeypatch, dos_imagenes):
    monkeypatch.setitem(sys.modules, "anthropic", None)        # hace que 'import anthropic' falle
    with pytest.raises(RuntimeError, match="pip install"):
        analizador.llamar_claude("k", ["m"], "c", "t", dos_imagenes)


# ======================================================================
# llamar_gemini (con google.genai falso)
# ======================================================================

@pytest.fixture
def genai_falso(monkeypatch):
    estado = types.SimpleNamespace(clientes=[], llamadas=[], falla=set())

    tipos = types.ModuleType("google.genai.types")

    class HttpOptions:
        def __init__(self, timeout):
            self.timeout = timeout

    class Part:
        @staticmethod
        def from_bytes(data, mime_type):
            return ("imagen", data, mime_type)

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    tipos.HttpOptions, tipos.Part, tipos.GenerateContentConfig = HttpOptions, Part, GenerateContentConfig

    class Modelos:
        def generate_content(self, model, contents, config):
            estado.llamadas.append({"model": model, "contents": contents, "config": config})
            if model in estado.falla:
                raise RuntimeError(f"{model} saturado")
            return types.SimpleNamespace(text="respuesta de gemini")

    class Cliente:
        def __init__(self, api_key, http_options=None):
            self.api_key, self.http_options, self.models = api_key, http_options, Modelos()
            estado.clientes.append(self)

    genai = types.ModuleType("google.genai")
    genai.__path__ = []
    genai.Client, genai.types = Cliente, tipos
    google = types.ModuleType("google")
    google.__path__ = []
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", tipos)
    return estado


def test_llamar_gemini_arma_bien_la_peticion(genai_falso, dos_imagenes):
    modelo, texto = analizador.llamar_gemini("g-key", ["gemini-x"], "CONTEXTO", "TEXTO",
                                             dos_imagenes, timeout_s=5)
    assert (modelo, texto) == ("gemini-x", "respuesta de gemini")
    assert genai_falso.clientes[0].api_key == "g-key"
    assert genai_falso.clientes[0].http_options.timeout == 5000       # el SDK usa milisegundos

    llamada = genai_falso.llamadas[0]
    assert llamada["contents"][0] == "TEXTO"                           # texto primero, luego imagenes
    assert [parte[0] for parte in llamada["contents"][1:]] == ["imagen", "imagen"]
    assert llamada["contents"][1][2] == "image/jpeg"
    assert llamada["config"].kwargs["system_instruction"] == "CONTEXTO"
    assert llamada["config"].kwargs["temperature"] == 0


def test_llamar_gemini_prueba_el_siguiente_modelo(genai_falso, dos_imagenes):
    genai_falso.falla = {"malo"}
    modelo, _ = analizador.llamar_gemini("k", ["malo", "bueno"], "c", "t", dos_imagenes)
    assert modelo == "bueno"


def test_llamar_gemini_si_todos_fallan(genai_falso, dos_imagenes):
    genai_falso.falla = {"a"}
    with pytest.raises(RuntimeError, match="Ningun modelo de Gemini"):
        analizador.llamar_gemini("k", ["a"], "c", "t", dos_imagenes)


# ======================================================================
# analizar_hoja (de extremo a extremo con un proveedor falso)
# ======================================================================

@pytest.fixture
def carpeta_hoja(tmp_path):
    carpeta = tmp_path / "cab_1"
    carpeta.mkdir()
    Image.new("RGB", (1204, 1600), "white").save(carpeta / "cab_1_anotada.jpg")
    Image.new("RGB", (1204, 1600), "white").save(carpeta / "cab_1_vertical.jpg")
    (carpeta / "cab_1_coordenadas.txt").write_text(texto_coordenadas_anotador(), encoding="utf-8")
    (tmp_path / "contexto.md").write_text("Eres un experto en ampelografia.", encoding="utf-8")
    return carpeta


def _config(nombre="claude"):
    return {"proveedores": [{"nombre": nombre, "api_key": "k", "modelos": ["m1"]}],
            "timeout_s": 5, "automatico": False}


def test_analizar_hoja_flujo_completo(llamadores, carpeta_hoja):
    llamadores["claude"] = lambda *a: ("m1", BLOQUE_OK)
    datos = analizador.analizar_hoja(carpeta_hoja, carpeta_hoja.parent / "contexto.md",
                                     config=_config())

    assert datos["estado"] == "OK"
    assert datos["proveedor_usado"] == "claude"
    assert datos["modelo_usado"] == "claude/m1"
    # L y angulos se recalculan en Python con la escala que dio la IA (58.67)
    assert datos["_local"]["L1_cm"] == pytest.approx(767.74 / ESCALA_EJEMPLO, abs=0.01)
    assert datos["_local"]["A1_grados"] == pytest.approx(59.05, abs=0.01)
    assert set(datos["_puntos"]) == set(analizador.NOMBRES_PUNTOS)

    # y deja los archivos de la respuesta en la carpeta de la hoja
    assert (carpeta_hoja / "respuesta_llm_completa.txt").read_text(encoding="utf-8") == BLOQUE_OK
    assert (carpeta_hoja / "resultados_llm.txt").exists()
    guardado = json.loads((carpeta_hoja / "resultados_llm.json").read_text(encoding="utf-8"))
    assert guardado["modelo_usado"] == "claude/m1"


def test_analizar_hoja_manda_las_dos_imagenes_si_existe_la_limpia(llamadores, carpeta_hoja):
    recibido = {}
    llamadores["claude"] = lambda k, m, ctx, txt, imgs, t: recibido.update(imgs=imgs, ctx=ctx) or ("m1", BLOQUE_OK)
    analizador.analizar_hoja(carpeta_hoja, carpeta_hoja.parent / "contexto.md", config=_config())
    assert [p.name for p in recibido["imgs"]] == ["cab_1_anotada.jpg", "cab_1_vertical.jpg"]
    assert recibido["ctx"] == "Eres un experto en ampelografia."


def test_analizar_hoja_sin_limpia_solo_manda_la_anotada(llamadores, carpeta_hoja):
    recibido = {}
    llamadores["claude"] = lambda k, m, ctx, txt, imgs, t: recibido.update(imgs=imgs) or ("m1", BLOQUE_OK)
    analizador.analizar_hoja(carpeta_hoja, carpeta_hoja.parent / "contexto.md",
                             usar_limpia=False, config=_config())
    assert [p.name for p in recibido["imgs"]] == ["cab_1_anotada.jpg"]


def test_analizar_hoja_respuesta_de_error_no_calcula_local(llamadores, carpeta_hoja):
    bloque = ("INICIO_RESULTADOS\nimagen: cab_1.jpeg\nvariedad: v\nestado: ERROR\n"
              "motivo: no se ve la cuadricula\nFIN_RESULTADOS")
    llamadores["claude"] = lambda *a: ("m1", bloque)
    datos = analizador.analizar_hoja(carpeta_hoja, carpeta_hoja.parent / "contexto.md", config=_config())
    assert datos["estado"] == "ERROR"
    assert "_local" not in datos


def test_analizar_hoja_sin_escala_calcula_solo_px(llamadores, carpeta_hoja):
    bloque = "INICIO_RESULTADOS\nimagen: x\nvariedad: v\nestado: OK\nFIN_RESULTADOS"
    llamadores["claude"] = lambda *a: ("m1", bloque)
    datos = analizador.analizar_hoja(carpeta_hoja, carpeta_hoja.parent / "contexto.md", config=_config())
    assert "L1_px" in datos["_local"] and "L1_cm" not in datos["_local"]


def test_analizar_hoja_si_el_unico_proveedor_falla_lanza_error(llamadores, carpeta_hoja):
    llamadores["claude"] = _falla("sin cuota")
    with pytest.raises(RuntimeError, match="Ningun proveedor"):
        analizador.analizar_hoja(carpeta_hoja, carpeta_hoja.parent / "contexto.md", config=_config())


def test_analizar_hoja_con_carpeta_incompleta(tmp_path):
    (tmp_path / "contexto.md").write_text("x", encoding="utf-8")
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    with pytest.raises(FileNotFoundError):
        analizador.analizar_hoja(vacia, tmp_path / "contexto.md", config=_config())