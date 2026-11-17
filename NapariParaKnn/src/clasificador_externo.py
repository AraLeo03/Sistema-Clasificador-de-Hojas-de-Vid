"""
clasificador_externo.py
=======================

Clasifica una hoja nueva ("elemento externo") con KNN usando las medidas que
calculo el analizador (L1-L4 en cm, angulos, etc.) contra un dataset CSV.

Reutiliza las clases KNN, Lector, Objeto y Distancia de knnQt.py, asi que el
resultado es el mismo que el de la pestaña "Elemento externo" del panel KNN.

TODA la configuracion esta en las constantes de la seccion 1. Cambialas ahi;
no hace falta tocar el resto del archivo.

Uso desde codigo:
    import clasificador_externo
    resultado = clasificador_externo.clasificar(medidas)      # medidas = datos["_local"]
    print(clasificador_externo.formatear_resultado(resultado))
"""

import csv
from pathlib import Path

from knnQt import KNN, Distancia, Lector, Objeto

CARPETA_PROGRAMA = Path(__file__).resolve().parent      # .../src
RAIZ_PROYECTO = CARPETA_PROGRAMA.parent                  # carpeta de main.py


# ======================================================================
# 1. CONSTANTES
# ======================================================================

# ---------------------------- DATASET ---------------------------------

# CSV de entrenamiento. Primera fila = encabezados; TODAS las columnas son
# atributos numericos y la ULTIMA columna es la clase (igual que en el panel KNN).
# Consejo: evita caracteres como ´ o espacios en el nombre del archivo; dan problemas entre sistemas.
DATASET_RUTA = RAIZ_PROYECTO / "data" / "dataset.csv"
#DATASET_RUTA = RAIZ_PROYECTO / "data" / "dataset(sin hojas 1).csv"

# Valor que se le pone a la clase del elemento nuevo (solo informativo).
DATASET_CLASE_DESCONOCIDA = "Desconocida"

# Equivalencia entre el nombre de cada columna del dataset y la medida que
# produce el analizador. Si la columna se llama igual que la medida (por
# ejemplo "L1_cm"), no hace falta ponerla aqui.
#
# Ejemplo:  {"L1": "L1_cm", "Ang1": "A1_grados"}
DATASET_EQUIVALENCIAS = {
    "L1": "L1_cm",
    "L2": "L2_cm",
    "L3": "L3_cm",
    "L4": "L4_cm",
    "A1": "A1_grados",
    "A2": "A2_grados",
    "A3": "A3_grados",
    "A2-3": "A2_3_grados",
    "A2-4": "A2_4_grados",
}

# Medidas que el analizador entrega y que puedes usar como columnas del dataset:
MEDIDAS_DISPONIBLES = [
    "L1_px", "L2_px", "L3_px", "L4_px",
    "L1_cm", "L2_cm", "L3_cm", "L4_cm",
    "A1_grados", "A2_grados", "A3_grados", "A2_3_grados", "A2_4_grados",
]

# ------------------------------- KNN ----------------------------------

# Numero de vecinos.
KNN_K = 3

# Escalado de los datos: "ninguno", "normalizar" (0 a 1) o "estandarizar" (z-score).
# OJO: L esta en cm y los angulos en grados; sin escalar, los grados dominan
# la distancia. 
KNN_ESCALADO = "normalizar"

# Atributos que se usan. None = todas las columnas del dataset (menos la clase).
# O una lista de nombres de columna, por ejemplo: ["L1_cm", "L2_cm", "A1_grados"]
KNN_ATRIBUTOS_USADOS = None

# Peso de cada atributo en la distancia. Los que no aparezcan usan el peso por defecto.
# Ejemplo: {"L1_cm": 2.0, "A1_grados": 0.5}
KNN_PESOS = {}
KNN_PESO_POR_DEFECTO = 1.0


# ======================================================================
# 2. CARGA DEL DATASET
# ======================================================================

def cargar_dataset(ruta=None):
    """Devuelve (encabezados, lista_de_Objeto). Lanza FileNotFoundError si no existe."""
    ruta = Path(ruta or DATASET_RUTA)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset: {ruta}\n"
            "Revisa la constante DATASET_RUTA en clasificador_externo.py."
        )

    with open(ruta, mode="r", encoding="utf-8") as f:
        encabezados = [h.strip() for h in next(csv.reader(f), [])]

    datos = Lector.leer_csv(str(ruta))
    if len(encabezados) < 2 or not datos:
        raise ValueError(f"El dataset {ruta} esta vacio o no tiene atributos y clase.")
    return encabezados, datos


# ======================================================================
# 3. CLASIFICACION
# ======================================================================

def _configuracion(columnas):
    """Indices y pesos de los atributos usados, segun las constantes."""
    if KNN_ATRIBUTOS_USADOS is None:
        indices = list(range(len(columnas)))
    else:
        desconocidas = [c for c in KNN_ATRIBUTOS_USADOS if c not in columnas]
        if desconocidas:
            raise ValueError(
                f"KNN_ATRIBUTOS_USADOS tiene columnas que no estan en el dataset: {desconocidas}"
            )
        indices = [i for i, c in enumerate(columnas) if c in KNN_ATRIBUTOS_USADOS]

    if not indices:
        raise ValueError("No hay atributos seleccionados para el KNN.")

    pesos = [float(KNN_PESOS.get(columnas[i], KNN_PESO_POR_DEFECTO)) for i in indices]
    return indices, pesos


def clasificar(medidas, ruta_dataset=None):
    """Clasifica una hoja nueva.

    medidas: diccionario {nombre_medida: valor}, por ejemplo datos["_local"] del analizador.
    Devuelve un diccionario con:
        clase, confianza (% de votos del ganador), votos {clase: n}, k, escalado,
        dataset, atributos (valor original y escalado de cada uno) y vecinos
        (los k mas cercanos con su distancia).
    Lanza ValueError si el analizador no entrego alguna medida que pide el dataset.
    """
    encabezados, datos_train = cargar_dataset(ruta_dataset)
    columnas = encabezados[:-1]                 # la ultima columna es la clase
    indices, pesos = _configuracion(columnas)

    # Valor del elemento nuevo para cada atributo usado
    # Las columnas que no se usan se rellenan con 0.0 (no participan en la distancia).
    valores = ["0.0"] * len(columnas)
    faltan = []
    for i in indices:
        medida = DATASET_EQUIVALENCIAS.get(columnas[i], columnas[i])
        if medida not in medidas:
            faltan.append(f"{columnas[i]} (medida '{medida}')")
        else:
            valores[i] = f"{float(medidas[medida])}"
    if faltan:
        raise ValueError(
            "El analizador no entrego estas medidas que pide el dataset: "
            + ", ".join(faltan)
            + ".\nRevisa DATASET_EQUIVALENCIAS o los encabezados del dataset."
        )

    nuevo = Objeto(valores + [DATASET_CLASE_DESCONOCIDA])

    k = max(1, min(int(KNN_K), len(datos_train)))
    if KNN_ESCALADO == "normalizar":
        train, prueba = KNN.normalizar_datos(datos_train, [nuevo], indices)
    elif KNN_ESCALADO == "estandarizar":
        train, prueba = KNN.estandarizar_datos(datos_train, [nuevo], indices)
    elif KNN_ESCALADO == "ninguno":
        train, prueba = datos_train, [nuevo]
    else:
        raise ValueError(f"KNN_ESCALADO invalido: '{KNN_ESCALADO}' (usa ninguno, normalizar o estandarizar)")

    nuevo_esc = prueba[0]
    clase, conteo = KNN.clasificar_dato(nuevo_esc, train, k, indices, pesos)

    distancias = []
    for ent in train:
        d = KNN.calcular_distancia_euclidiana(nuevo_esc, ent, indices, pesos)
        distancias.append(Distancia(d, ent.clase, ent.atributos))
    distancias.sort(key=lambda x: x.valor)

    votos_ganador = conteo[clase]
    total_votos = sum(conteo.values())

    return {
        "clase": clase,
        "confianza": votos_ganador / total_votos * 100.0,
        "votos": conteo,
        "k": k,
        "escalado": KNN_ESCALADO,
        "dataset": str(ruta_dataset or DATASET_RUTA),
        "atributos": [
            {
                "nombre": columnas[i],
                "valor": nuevo.atributos[i],
                "escalado": nuevo_esc.atributos[i],
                "peso": pesos[n],
            }
            for n, i in enumerate(indices)
        ],
        "vecinos": [
            {
                "valores": [d.atributos[i].strip() for i in indices],
                "clase": d.clase,
                "distancia": d.valor,
            }
            for d in distancias[:k]
        ],
        "columnas_usadas": [columnas[i] for i in indices],
    }


# ======================================================================
# 4. TEXTO PARA MOSTRAR
# ======================================================================

def formatear_resultado(r):
    """Texto con el mismo estilo que la pestaña 'Elemento externo' del panel KNN."""
    sb = [
        "=== CLASIFICACION DE LA HOJA (KNN) ===",
        f"K = {r['k']} | Escalado = {r['escalado'].upper()}",
        f"Dataset: {r['dataset']}",
        "",
        "--- Valores usados ---",
    ]
    for a in r["atributos"]:
        sb.append(f"  * {a['nombre']:<14}: {float(a['valor']):>10.2f}  "
                  f"(escalado: {a['escalado']}, peso: {a['peso']})")

    total = sum(r["votos"].values())
    sb += [
        "",
        "--- Resultado ---",
        f"CLASE PREDICHA : {r['clase']}",
        f"Confianza      : {r['confianza']:.2f}% ({r['votos'][r['clase']]}/{total} votos)",
        f"Votos por clase: {r['votos']}",
        "",
        f"--- {r['k']} vecinos mas cercanos (valores segun el escalado) ---",
        "| " + " | ".join(r["columnas_usadas"]) + " | Clase | Distancia |",
    ]
    for v in r["vecinos"]:
        sb.append("| " + " | ".join(v["valores"]) + f" | {v['clase']} | {v['distancia']:.4f} |")
    return "\n".join(sb)