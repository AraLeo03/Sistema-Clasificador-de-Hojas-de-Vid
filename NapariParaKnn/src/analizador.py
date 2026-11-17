"""
analizador.py
=============

Modulo de analisis con LLM (se usa desde el anotador y tambien por terminal): manda a la LLM la imagen anotada + las coordenadas + el
archivo de contexto, y guarda el bloque de resultados que devuelve.

Ademas recalcula L1-L4 y los angulos en Python con las mismas coordenadas y
los compara contra lo que respondio la LLM (para saber si se puede confiar).

Instalacion:
    pip install -U google-genai python-dotenv pillow

Configuracion: copia .env.example como .env (en la RAIZ del proyecto, junto a main.py) y escribe ahi tu API_KEY y las demas opciones.

Uso:
    python analizador.py resultados/para_gemini/cab_1
    python analizador.py resultados/para_gemini/cab_1 --modelo gemini-3.8-flash
    python analizador.py resultados/para_gemini/cab_1 --sin-llamar   (solo muestra el prompt)

La carpeta debe contener (el anotador los crea en una carpeta TEMPORAL al guardar;
para conservarlos pon CONSERVAR_ARCHIVOS_TEMPORALES = True en anotador_napari.py):
    <nombre>_anotada.jpg       imagen con los 5 puntos dibujados
    <nombre>_vertical.jpg      imagen limpia (opcional, ayuda a ver la cuadricula)
    <nombre>_coordenadas.txt   imagen, variedad, tamano y landmark, x, y
"""

import argparse
import base64
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

CARPETA_SCRIPT = Path(__file__).resolve().parent      # .../src
RAIZ_PROYECTO = CARPETA_SCRIPT.parent                  # carpeta de main.py, .env, prompts/, data/
# Prompt de sistema que se manda a la LLM (instrucciones, formato de salida, ejemplo).
CONTEXTO_POR_DEFECTO = RAIZ_PROYECTO / "prompts" / "contextFile.md"

try:
    from dotenv import load_dotenv
    load_dotenv(RAIZ_PROYECTO / ".env")
except ImportError:
    print("Aviso: no esta instalado python-dotenv; el archivo .env no se leera "
          "(pip install python-dotenv).")


# Proveedores de IA soportados (los nombres que se ponen en el .env).
PROVEEDORES_IA = ("gemini", "claude")


def cargar_config():
    """Lee la configuracion desde las variables de entorno (.env).

    Variables:
        IA_PROVEEDOR_PRINCIPAL    gemini | claude  (por defecto gemini). Se intenta primero.
        IA_PROVEEDOR_SECUNDARIO   gemini | claude | ninguno. Se usa si el principal falla.
                                  Vacio = el otro proveedor.
        IA_TIMEOUT_SEGUNDOS       espera maxima por intento (por defecto 60).
        IA_ANALISIS_AUTOMATICO    true/false: si el anotador lanza el analisis al guardar.

        GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MODELOS_RESPALDO
        ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MODELOS_RESPALDO

    Devuelve un diccionario con:
        proveedores  lista ORDENADA [{"nombre", "api_key", "modelos"}, ...]: principal primero
        timeout_s, automatico
    """
    def como_bool(valor):
        return str(valor).strip().lower() in ("1", "true", "si", "sí", "yes", "on")

    def modelos(var_modelo, defecto, var_respaldo):
        """Modelo principal + modelos de respaldo del MISMO proveedor (sin repetir)."""
        principal = os.environ.get(var_modelo, "").strip() or defecto
        respaldo = [m.strip() for m in os.environ.get(var_respaldo, "").split(",") if m.strip()]
        return [principal] + [m for m in respaldo if m != principal]

    # Si el nombre de un modelo falla, revisa los disponibles con:
    #   python src/analizador.py --listar-modelos      (lista los de Gemini)
    datos_proveedor = {
        "gemini": {
            "api_key": os.environ.get("GEMINI_API_KEY", "").strip(),
            "modelos": modelos("GEMINI_MODEL", "gemini-3.8-flash", "GEMINI_MODELOS_RESPALDO"),
        },
        "claude": {
            "api_key": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
            "modelos": modelos("CLAUDE_MODEL", "claude-sonnet-5-5", "CLAUDE_MODELOS_RESPALDO"),
        },
    }

    # Un valor mal escrito en el .env no debe tirar el programa: se avisa y se usa el valor por defecto.
    principal = os.environ.get("IA_PROVEEDOR_PRINCIPAL", "claude").strip().lower() or "claude"
    if principal not in PROVEEDORES_IA:
        print(f"Aviso: IA_PROVEEDOR_PRINCIPAL='{principal}' no es valido {PROVEEDORES_IA}; se usa claude.")
        principal = "claude"

    otro = [p for p in PROVEEDORES_IA if p != principal][0]
    secundario = os.environ.get("IA_PROVEEDOR_SECUNDARIO", "").strip().lower()
    if secundario == "":
        secundario = otro                                   # por defecto: el otro proveedor
    elif secundario in ("ninguno", "no", "none"):
        secundario = None                                   # sin respaldo
    elif secundario not in PROVEEDORES_IA:
        print(f"Aviso: IA_PROVEEDOR_SECUNDARIO='{secundario}' no es valido; se usa {otro}.")
        secundario = otro

    orden = [principal] + ([secundario] if secundario and secundario != principal else [])

    # GEMINI_TIMEOUT_SEGUNDOS se sigue aceptando por compatibilidad con .env antiguos.
    timeout = (os.environ.get("IA_TIMEOUT_SEGUNDOS") or os.environ.get("GEMINI_TIMEOUT_SEGUNDOS") or "60")
    return {
        "proveedores": [{"nombre": n, **datos_proveedor[n]} for n in orden],
        "timeout_s": float(timeout),
        "automatico": como_bool(os.environ.get("IA_ANALISIS_AUTOMATICO", "false")),
        "conservar_archivos": como_bool(os.environ.get("CONSERVAR_ARCHIVOS_IA", "true")),
    }

# Los 5 landmarks que se mandan a la IA (mismos nombres que en anotador_napari.PUNTOS).
NOMBRES_PUNTOS = ["seno_peciolar", "punta_L1", "punta_L2", "punta_L3", "punta_L4"]

# (nombre en la salida, linea de inicio, linea de fin)
ANGULOS = [
    ("A1_grados",   "L1", "L2"),
    ("A2_grados",   "L1", "L3"),
    ("A3_grados",   "L1", "L4"),
    ("A2_3_grados", "L2", "L3"),
    ("A2_4_grados", "L2", "L4"),
]


# ======================================================================
# 1. LECTURA DE ARCHIVOS
# ======================================================================

def leer_coordenadas(ruta_txt):
    """Lee el .txt del anotador. Devuelve (texto_completo, dict landmark -> (x, y))."""
    texto = Path(ruta_txt).read_text(encoding="utf-8")
    puntos = {}
    for linea in texto.splitlines():
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) == 3 and partes[0] in NOMBRES_PUNTOS:
            puntos[partes[0]] = (float(partes[1]), float(partes[2]))

    faltan = [n for n in NOMBRES_PUNTOS if n not in puntos]
    if faltan:
        raise ValueError(f"Faltan puntos en {ruta_txt}: {', '.join(faltan)}")
    return texto, puntos


def encontrar_archivos(carpeta):
    """Busca la imagen anotada, la limpia y el txt de coordenadas dentro de la carpeta."""
    carpeta = Path(carpeta)
    anotada = sorted(carpeta.glob("*_anotada.jpg"))
    vertical = sorted(carpeta.glob("*_vertical.jpg"))
    coords = sorted(carpeta.glob("*_coordenadas.txt"))

    if not anotada or not coords:
        raise FileNotFoundError(
            f"En {carpeta} hacen falta *_anotada.jpg y *_coordenadas.txt "
            "(se generan al guardar en el anotador)."
        )
    return anotada[0], (vertical[0] if vertical else None), coords[0]


# ======================================================================
# 2. CALCULO LOCAL (para comparar contra Gemini)
# ======================================================================

def vector(puntos, linea):
    """Vector (dx, dy) desde seno_peciolar hasta la punta de la linea ('L1'..'L4'), en pixeles."""
    xs, ys = puntos["seno_peciolar"]
    x, y = puntos[f"punta_{linea}"]
    return (x - xs, y - ys)


def angulo_horario(a, b):
    """Angulo en grados girando en sentido horario (en pantalla) de a hacia b."""
    cruz = a[0] * b[1] - a[1] * b[0]
    punto = a[0] * b[0] + a[1] * b[1]
    return math.degrees(math.atan2(cruz, punto)) % 360


def calcular_local(puntos, escala_px_cm=None):
    """Longitudes y angulos a partir de las coordenadas (precision completa)."""
    resultado = {}
    for linea in ("L1", "L2", "L3", "L4"):
        v = vector(puntos, linea)
        px = math.hypot(*v)
        resultado[f"{linea}_px"] = px
        if escala_px_cm:
            resultado[f"{linea}_cm"] = px / escala_px_cm
    for nombre, a, b in ANGULOS:
        resultado[nombre] = angulo_horario(vector(puntos, a), vector(puntos, b))
    return resultado


# ======================================================================
# 3. ARMADO DEL MENSAJE Y LLAMADA A GEMINI
# ======================================================================

def armar_texto_usuario(texto_coordenadas, hay_imagen_limpia):
    """Texto del mensaje a Gemini: describe las imagenes adjuntas, pega las coordenadas y exige el bloque de resultados."""
    imagenes = "La primera imagen es la hoja con los 5 puntos dibujados."
    if hay_imagen_limpia:
        imagenes += " La segunda es la misma imagen sin puntos (util para ver la cuadricula)."
    return (
        "Aplica el archivo de contexto a esta solicitud.\n\n"
        f"{imagenes}\n\n"
        "Coordenadas de los puntos:\n"
        f"{texto_coordenadas}\n"
        "Responde unicamente con el bloque INICIO_RESULTADOS ... FIN_RESULTADOS."
    )


def llamar_gemini(api_key, modelos, contexto, texto_usuario, rutas_imagenes, timeout_s=60):
    """Prueba cada modelo de Gemini en orden; si uno falla, pasa al siguiente.

    Devuelve (modelo_que_respondio, texto_de_la_respuesta).
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("Falta la libreria: pip install -U google-genai")

    # El timeout del SDK se da en milisegundos
    cliente = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )

    partes = [texto_usuario]
    for ruta in rutas_imagenes:
        partes.append(types.Part.from_bytes(data=Path(ruta).read_bytes(), mime_type="image/jpeg"))

    errores = []
    for modelo in modelos:
        try:
            respuesta = cliente.models.generate_content(
                model=modelo,
                contents=partes,
                config=types.GenerateContentConfig(system_instruction=contexto, temperature=0),
            )
            return modelo, (respuesta.text or "")
        except Exception as error:  # modelo inexistente, saturado, timeout, cuota...
            print(f"  Fallo el modelo {modelo}: {error}")
            errores.append(f"{modelo}: {error}")

    raise RuntimeError("Ningun modelo de Gemini respondio:\n  " + "\n  ".join(errores))


def llamar_claude(api_key, modelos, contexto, texto_usuario, rutas_imagenes, timeout_s=60):
    """Igual que llamar_gemini pero con la API de Claude (Messages API).

    Las imagenes van en base64 ANTES del texto (orden recomendado por la documentacion;
    limite de 5 MB por imagen). El contexto va como 'system'. No se envia 'temperature':
    los modelos recientes de Claude no aceptan valores distintos del predeterminado.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Falta la libreria: pip install -U anthropic")

    cliente = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)   # timeout en segundos

    contenido = []
    for ruta in rutas_imagenes:
        datos = base64.standard_b64encode(Path(ruta).read_bytes()).decode("ascii")
        contenido.append({"type": "image",
                          "source": {"type": "base64", "media_type": "image/jpeg", "data": datos}})
    contenido.append({"type": "text", "text": texto_usuario})

    errores = []
    for modelo in modelos:
        try:
            respuesta = cliente.messages.create(
                model=modelo,
                max_tokens=2000,
                system=contexto,
                thinking={"type": "disabled"},
                #output_config={"effort": "medium"},
                messages=[{"role": "user", "content": contenido}],
            )
            texto = "".join(b.text for b in respuesta.content if b.type == "text")
            print("stop_reason:", respuesta.stop_reason)
            print("tipos de bloque:", [b.type for b in respuesta.content])
            print("texto recibido:", texto[:500])
            return modelo, texto
        except Exception as error:  # modelo inexistente, saturado, timeout, cuota...
            print(f"  Fallo el modelo {modelo}: {error}")
            errores.append(f"{modelo}: {error}")

    raise RuntimeError("Ningun modelo de Claude respondio:\n  " + "\n  ".join(errores))


# Funcion que atiende a cada proveedor (los nombres coinciden con PROVEEDORES_IA).
LLAMADORES = {"gemini": llamar_gemini, "claude": llamar_claude}


def llamar_ia(proveedores, contexto, texto_usuario, rutas_imagenes, timeout_s=60):
    """Prueba los proveedores en orden (principal primero, luego el secundario).

    Pasa al siguiente si el proveedor: no tiene API key, falla con todos sus modelos
    o responde sin el bloque INICIO_RESULTADOS/FIN_RESULTADOS.
    Devuelve (proveedor, modelo, texto_de_la_respuesta).
    """
    errores = []
    for prov in proveedores:
        nombre = prov["nombre"]
        if not prov["api_key"]:
            print(f"  Se omite {nombre}: falta su API key en el .env")
            errores.append(f"[{nombre}] falta su API key en el .env (ver .env.example)")
            continue
        try:
            modelo, texto = LLAMADORES[nombre](prov["api_key"], prov["modelos"], contexto,
                                               texto_usuario, rutas_imagenes, timeout_s)
            if extraer_bloque(texto) is None:
                raise ValueError("la respuesta no trae el bloque INICIO_RESULTADOS/FIN_RESULTADOS")
            return nombre, modelo, texto
        except Exception as error:
            print(f"  Fallo el proveedor {nombre}: {error}")
            errores.append(f"[{nombre}] {error}")

    raise RuntimeError("Ningun proveedor de IA respondio:\n" + "\n".join(errores))


def listar_modelos(proveedor="gemini", api_key=""):
    """Imprime los modelos disponibles del proveedor indicado para tu API key."""
    if not api_key:
        raise RuntimeError(f"Falta la API key de {proveedor} en el .env (ver .env.example)")

    if proveedor == "gemini":
        try:
            from google import genai
        except ImportError:
            raise RuntimeError("Falta la libreria: pip install -U google-genai")
        cliente = genai.Client(api_key=api_key)
        for m in cliente.models.list():
            print(m.name)

    elif proveedor == "claude":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Falta la libreria: pip install -U anthropic")
        cliente = anthropic.Anthropic(api_key=api_key)
        for m in cliente.models.list():
            print(f"{m.id}  ({m.display_name})")

    else:
        raise ValueError(f"Proveedor no soportado: {proveedor}")


# ======================================================================
# 4. LECTURA DE LA RESPUESTA
# ======================================================================

def extraer_bloque(texto):
    """Devuelve unicamente el texto entre INICIO_RESULTADOS y FIN_RESULTADOS (sin etiquetas ni formato extra)."""
    if "INICIO_RESULTADOS" not in texto or "FIN_RESULTADOS" not in texto:
        return None
        
    # 1. Encontrar posiciones ignorando el texto de alrededor
    inicio = texto.index("INICIO_RESULTADOS") + len("INICIO_RESULTADOS")
    fin = texto.index("FIN_RESULTADOS")
    
    # 2. Recortar dejando solo el interior
    bloque = texto[inicio:fin].strip()
    
    # 3. Limpiar los bloques markdown que la IA suele añadir
    bloque = bloque.replace("```json", "").replace("```", "").strip()
    
    return bloque


def parsear_bloque(bloque):
    """Convierte las lineas 'campo: valor' en un diccionario (numeros como float)."""
    datos = {}
    for linea in bloque.splitlines():
        if ":" not in linea:
            continue
        clave, valor = linea.split(":", 1)
        clave, valor = clave.strip(), valor.strip()
        try:
            datos[clave] = float(valor)
        except ValueError:
            datos[clave] = valor
    return datos


# ======================================================================
# 5. COMPARACION GEMINI VS PYTHON
# ======================================================================

def comparar(datos_gemini, puntos):
    escala = datos_gemini.get("escala_promedio_px_cm")
    escala = escala if isinstance(escala, float) else None
    local = calcular_local(puntos, escala)

    print("\nComparacion Gemini vs calculo local en Python")
    print("-" * 62)
    print(f"{'campo':<16}{'Gemini':>12}{'Python':>12}{'diferencia':>14}")
    print("-" * 62)
    for campo, valor_local in local.items():
        valor_g = datos_gemini.get(campo)
        if isinstance(valor_g, float):
            print(f"{campo:<16}{valor_g:>12.2f}{valor_local:>12.2f}{valor_g - valor_local:>14.2f}")
        else:
            print(f"{campo:<16}{'(no vino)':>12}{valor_local:>12.2f}{'':>14}")
    print("-" * 62)
    if escala is None:
        print("Aviso: Gemini no devolvio la escala; se omitieron los cm.")
    return local


# ======================================================================
# 6. FUNCION REUTILIZABLE (la usa tambien el anotador)
# ======================================================================

def analizar_hoja(carpeta, ruta_contexto=CONTEXTO_POR_DEFECTO, usar_limpia=True, config=None):
    """Analiza una carpeta de hoja con la LLM y guarda los resultados en ella.

    Pasos: leer archivos de la carpeta -> llamar a Gemini -> extraer y parsear el
    bloque INICIO_RESULTADOS/FIN_RESULTADOS -> recalcular L y angulos en Python.

    Devuelve un diccionario con los datos (incluye 'modelo_usado' y 'estado').
    Claves principales:
        estado, advertencias, escala_*_px_cm ... lo que respondio Gemini (numeros como float)
        proveedor_usado  'gemini' o 'claude' (el que respondio)
        modelo_usado     'proveedor/modelo' que respondio
        _local         L1-L4 (px y cm) y angulos calculados en Python (los que se usan de verdad)
        _bloque        texto crudo del bloque de resultados
        _puntos        dict landmark -> (x, y)
    Ademas escribe en la carpeta: respuesta_llm_completa.txt,
    resultados_llm.txt y resultados_llm.json.
    Lanza una excepcion si algo falla, para que quien la llame decida que hacer.
    """
    config = config or cargar_config()
    carpeta = Path(carpeta)

    contexto = Path(ruta_contexto).read_text(encoding="utf-8")
    anotada, limpia, ruta_coords = encontrar_archivos(carpeta)
    if not usar_limpia:
        limpia = None

    texto_coords, puntos = leer_coordenadas(ruta_coords)
    texto_usuario = armar_texto_usuario(texto_coords, limpia is not None)
    imagenes = [anotada] + ([limpia] if limpia else [])

    proveedor, modelo, respuesta = llamar_ia(config["proveedores"], contexto, texto_usuario,
                                             imagenes, config["timeout_s"])
    (carpeta / "respuesta_llm_completa.txt").write_text(respuesta, encoding="utf-8")

    bloque = extraer_bloque(respuesta)
    if bloque is None:
        raise ValueError("La IA no devolvio el bloque INICIO_RESULTADOS/FIN_RESULTADOS. "
                         f"Revisa {carpeta / 'respuesta_llm_completa.txt'}")

    datos = parsear_bloque(bloque)
    datos["proveedor_usado"] = proveedor
    datos["modelo_usado"] = f"{proveedor}/{modelo}"   # se guarda asi en medidas_hojas.csv
    if datos.get("estado") == "OK":
        escala = datos.get("escala_promedio_px_cm")
        datos["_local"] = calcular_local(puntos, escala if isinstance(escala, float) else None)
    (carpeta / "resultados_llm.txt").write_text(bloque + "\n", encoding="utf-8")
    (carpeta / "resultados_llm.json").write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    datos["_bloque"] = bloque
    datos["_puntos"] = puntos
    return datos


# ======================================================================
# 6b. RESUMEN Y CSV DE MEDIDAS (los resultados SI se conservan)
# ======================================================================

# Columnas de medidas_hojas.csv (una fila por imagen analizada).
COLUMNAS_MEDIDAS = [
    "fecha", "imagen", "variedad", "clase_predicha", "confianza_pct", "modelo",
    "escala_horizontal_px_cm", "escala_vertical_px_cm", "escala_promedio_px_cm",
    "L1_px", "L1_cm", "L2_px", "L2_cm", "L3_px", "L3_cm", "L4_px", "L4_cm",
    "A1_grados", "A2_grados", "A3_grados", "A2_3_grados", "A2_4_grados",
    "advertencias",
]


def fila_medidas(datos):
    """Arma la fila del CSV. L y angulos vienen del calculo en Python (exacto);
    de Gemini solo se toma la escala."""
    local = datos["_local"]
    fila = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "imagen": datos.get("imagen", ""),
        "variedad": datos.get("variedad", ""),
        "modelo": datos.get("modelo_usado", ""),
        "advertencias": datos.get("advertencias", ""),
    }
    clasificacion = datos.get("clasificacion")
    fila["clase_predicha"] = clasificacion["clase"] if clasificacion else ""
    fila["confianza_pct"] = round(clasificacion["confianza"], 2) if clasificacion else ""
    for campo in ("escala_horizontal_px_cm", "escala_vertical_px_cm", "escala_promedio_px_cm"):
        fila[campo] = datos.get(campo, "")
    for campo, valor in local.items():
        fila[campo] = round(valor, 2)
    return fila


def guardar_medidas(datos, archivo_csv):
    """Agrega (o reemplaza, si la imagen ya estaba) la fila de esta hoja en el CSV."""
    archivo_csv = Path(archivo_csv)
    archivo_csv.parent.mkdir(parents=True, exist_ok=True)
    nueva = fila_medidas(datos)

    filas = []
    if archivo_csv.exists():
        with open(archivo_csv, newline="", encoding="utf-8") as f:
            filas = [r for r in csv.DictReader(f) if r.get("imagen") != nueva["imagen"]]
    filas.append(nueva)

    with open(archivo_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_MEDIDAS)
        escritor.writeheader()
        escritor.writerows(filas)
    return archivo_csv


def resumen_texto(datos):
    """Texto corto para mostrar en pantalla."""
    if datos.get("estado") != "OK":
        return f"La IA no pudo calcular: {datos.get('motivo', 'sin motivo')}"
    l = datos["_local"]
    cm = "cm" if "L1_cm" in l else "px"
    partes = [f"{k} = {l[f'{k}_{cm}']:.2f} {cm}" for k in ("L1", "L2", "L3", "L4")]
    angulos = [f"{n.replace('_grados', '')} = {l[n]:.2f}°" for n, _, _ in ANGULOS]
    texto = "  |  ".join(partes) + "\n" + "  |  ".join(angulos)
    if isinstance(datos.get("escala_promedio_px_cm"), float):
        texto += f"\nEscala: {datos['escala_promedio_px_cm']:.2f} px/cm"
    if datos.get("advertencias") not in (None, "", "ninguna"):
        texto += f"\nAdvertencias: {datos['advertencias']}"
    return texto


# ======================================================================
# 7. PROGRAMA PRINCIPAL (pruebas desde la terminal)
# ======================================================================

def main():
    config = cargar_config()

    ap = argparse.ArgumentParser(description="Prueba de llamada a Gemini para medir hojas de vid.")
    ap.add_argument("carpeta", nargs="?", help="Carpeta de la hoja (con _anotada.jpg y _coordenadas.txt)")
    ap.add_argument("--contexto", default=str(CONTEXTO_POR_DEFECTO), help="Ruta de contextFile.md")
    ap.add_argument("--proveedor", choices=PROVEEDORES_IA, help="Usa solo este proveedor (sin respaldo)")
    ap.add_argument("--modelo", help="Fuerza un modelo en el proveedor principal (ignora sus modelos de respaldo)")
    ap.add_argument("--sin-limpia", action="store_true", help="No enviar la imagen sin puntos")
    ap.add_argument("--sin-llamar", action="store_true", help="Solo mostrar el mensaje; no llama a Gemini")
    ap.add_argument("--listar-modelos", action="store_true", help="Lista los modelos disponibles y sale")
    args = ap.parse_args()

    if args.proveedor:
        config["proveedores"] = [p for p in config["proveedores"] if p["nombre"] == args.proveedor] \
            or [{"nombre": args.proveedor, "api_key": "", "modelos": []}]
    if args.modelo:
        config["proveedores"][0]["modelos"] = [args.modelo]

    if args.listar_modelos:
        prov = config["proveedores"][0]        # el elegido con --proveedor, o el principal del .env
        print(f"Modelos disponibles en {prov['nombre']}:\n")
        try:
            listar_modelos(prov["nombre"], prov["api_key"])
        except Exception as error:
            sys.exit(f"Error: {error}")
        return

    if not args.carpeta:
        ap.error("Indica la carpeta de la hoja, por ejemplo: resultados/para_gemini/cab_1")

    print("Proveedores: " + " -> ".join(
        f"{p['nombre']} ({', '.join(p['modelos'])})" for p in config["proveedores"]))
    print(f"Timeout:   {config['timeout_s']:.0f} s")
    print(f"Contexto:  {args.contexto}")

    if args.sin_llamar:
        contexto = Path(args.contexto).read_text(encoding="utf-8")
        anotada, limpia, ruta_coords = encontrar_archivos(args.carpeta)
        limpia = None if args.sin_limpia else limpia
        texto_coords, puntos = leer_coordenadas(ruta_coords)
        print(f"Imagenes:  {anotada.name}" + (f", {limpia.name}" if limpia else ""))
        print(f"Contexto:  {len(contexto)} caracteres")
        print("\n--- MENSAJE AL USUARIO (no se llamo a Gemini) ---")
        print(armar_texto_usuario(texto_coords, limpia is not None))
        print("\n--- Calculo local con las coordenadas (sin escala) ---")
        for k, v in calcular_local(puntos).items():
            print(f"  {k:<14}{v:>10.2f}")
        return

    print("\nLlamando a la IA...")
    try:
        datos = analizar_hoja(args.carpeta, args.contexto, not args.sin_limpia, config)
    except Exception as error:
        sys.exit(f"\nError: {error}")

    print(f"\nRespondio el modelo: {datos['modelo_usado']}\n")
    print(datos["_bloque"])
    if datos.get("estado") == "OK":
        comparar(datos, datos["_puntos"])
    print(f"\nGuardado en: {args.carpeta}")


if __name__ == "__main__":
    main()