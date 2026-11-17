# Clasificador de hojas de vid (napari + LLM + KNN)

Programa de escritorio para identificar la **variedad de una hoja de vid** a partir de una foto. El flujo completo es:

1. Abres la foto y marcas a mano **5 puntos anatómicos** (landmarks).
2. Una LLM (Gemini) calcula la **escala** (px/cm) mirando la cuadrícula milimétrica de la foto.
3. Python calcula las longitudes **L1–L4** (cm) y los **ángulos** entre ellas.
4. Un **KNN** compara esas medidas contra un dataset y predice la variedad.

Además incluye un **Experimentador KNN** para probar el algoritmo (valor de K, escalado, pesos, métricas) sobre cualquier dataset CSV.

---

## 1. Estructura del proyecto

```
NapariParaKnn/
├── main.py                  ← EJECUTAR ESTE: ventana con menú
├── src/
│   ├── anotador_napari.py       ← pantalla "Clasificador Guiado" (napari)
│   ├── analizador.py            ← llamada a la LLM + cálculo de L y ángulos
│   ├── clasificador_externo.py  ← KNN sobre las medidas de la hoja (configuración en constantes)
│   ├── knnQt.py                 ← algoritmo KNN + panel "Experimentador KNN"
│   └── dialogo_ia.py             ← ventana de espera mientras la LLM responde
├── prompts
│   └── contextFile.md           ← prompt que se envía a la LLM
├── data/                        ← CSV de entrenamiento del KNN
│   ├── Dataset - L´s y Angulos 70_30 Shuffle.csv
├── resultados/                  ← CSV generados (se crea sola)
│   ├── hojas_landmarks.csv      ← puntos marcados (una fila por punto)
│   └── medidas_hojas.csv        ← medidas y clase predicha (una fila por hoja)
├── assets/                      ← Imagenes usadas para la GUI
│   ├── HojaMedidaCab1.jpg       ← fondo del menú
│   └── EjemploBase.png          ← imagen de referencia mostrada en el anotador
├── ejemplo-dataset/         ← fotos de prueba (una subcarpeta por variedad)
├── .env                     ← tu configuración y API key (NO se sube a git)
├── .env.example             ← plantilla del .env
├── .gitignore
└── requirements.txt
```

### Cómo se conectan los módulos

```
main.py ──► anotador_napari.py ──► analizador.py ──► Gemini (contextFile.md)
   │                │
   │                ├──► clasificador_externo.py ──► knnQt.py (KNN, Lector, Objeto)
   │                └──► dialogo_ia.py
   └──────► knnQt.py (PanelKNN)
```

---

## 2. Instalación (solo la primera vez)

Necesitas Python 3.10 o más reciente. Dentro de la carpeta del proyecto:

```bash
pip install -r requirements.txt
```

Con Anaconda es mejor un entorno aparte:

```bash
conda create -n hojas python=3.11
conda activate hojas
pip install -r requirements.txt
```

### Configurar la IA (opcional)

Copia `.env.example` como `.env` y rellénalo:

| Variable | Qué hace |
|---|---|
| `GEMINI_API_KEY` | Tu clave de la API de Gemini (obligatoria para usar la IA). |
| `GEMINI_MODEL` | Modelo principal. |
| `GEMINI_MODELOS_RESPALDO` | Modelos alternos separados por coma; se prueban en orden si el principal falla. |
| `GEMINI_TIMEOUT_SEGUNDOS` | Espera máxima por intento (por defecto 60). |
| `IA_ANALISIS_AUTOMATICO` | `true`: al guardar los puntos se lanza solo el análisis. `false`: solo se guardan los puntos. |

Si un nombre de modelo falla, lista los disponibles con
`python src/analizador.py --listar-modelos`.

> **Importante:** nunca subas el archivo `.env` a git ni lo compartas; contiene tu clave.

---

## 3. Uso

```bash
python main.py
```

Se abre el menú con tres botones:

- **Clasificador Guiado**: pide una imagen y abre el anotador.
- **Experimentador KNN**: panel para probar el KNN con un dataset.
- **Cerrar**.

### Clasificador Guiado

1. Elige la foto (jpg, jpeg o png). Se muestra **siempre vertical**.
2. Haz clic sobre los 5 puntos en el orden indicado en el panel derecho.
3. Presiona **G** (o el botón *Guardar*). Los puntos se guardan en
   `resultados/hojas_landmarks.csv`.
4. Si `IA_ANALISIS_AUTOMATICO=true`, aparece una ventana de espera con tres pasos
   (guardado → IA → KNN) y al final verás las medidas y la **clase predicha** con
   su confianza. Las medidas se guardan en `resultados/medidas_hojas.csv`.

> Mientras la IA responde, no cierres la ventana: si cancelas, el resultado se descarta.

---

## 4. Los 5 puntos

Se colocan en este orden. Los colores son los del anotador.

| # | Nombre | Color | Dónde va |
|---|---|---|---|
| 1 | `seno_peciolar` | rojo | Donde el pecíolo (el tallito) se une a la hoja. Es el origen de L1–L4. |
| 2 | `punta_L1`      | morado | Punta del lóbulo central (el de hasta arriba). |
| 3 | `punta_L2`      | azul | Punta del lóbulo L2. |
| 4 | `punta_L3`      | amarillo | Punta del lóbulo L3. |
| 5 | `punta_L4`      | cian | Punta del lóbulo L4 (el más cercano al pecíolo). |

Las líneas L1 → L4 siguen el **sentido de las manecillas del reloj**.
La imagen de referencia está en el panel del anotador (`EjemploBase.png`).

## 5. Controles

| Tecla / acción  | Qué hace |
|-----------------|----------|
| Clic izquierdo  | Coloca el siguiente punto |
| **G**           | Guardar (y lanzar el análisis) |
| **D**           | Deshacer el último punto |
| **Q**           | Quitar todos los puntos |
| **R**           | Imagen anterior (sin efecto con una sola imagen) |
| Rueda del ratón | Acercar / alejar |
| Espacio + arrastrar | Mover la imagen |
| **3**, luego arrastrar un punto | Corregir un punto (después **2** para volver a añadir) |

Solo se puede guardar cuando están los 5 puntos. Si te equivocas usa **D**; no borres un punto intermedio con Supr, porque los siguientes cambiarían de nombre.

---

## 6. Qué calcula

**Escala.** La foto se toma sobre una cuadrícula milimétrica; los cuadros grandes miden 1 cm. Gemini mide cuántos píxeles ocupa 1 cm y devuelve `px/cm`.

**Longitudes.** Distancia recta entre `seno_peciolar` y cada punta, dividida entre la escala: `L1_cm … L4_cm`. (Son ligeramente menores que una medición manual siguiendo el nervio, porque aquí se usa una línea recta.)

**Ángulos** (en grados, sentido horario, medidos en el seno peciolar):

| Medida |  Entre  |
|--------|---------|
| `A1`   | L1 → L2 |
| `A2`   | L1 → L3 |
| `A3`   | L1 → L4 |
| `A2_3` | L2 → L3 |
| `A2_4` | L2 → L4 |

Las longitudes y ángulos se **recalculan siempre en Python** con las coordenadas exactas; de Gemini solo se toma la escala.

---

## 7. Archivos de salida

**`resultados/hojas_landmarks.csv`**: una fila por punto, 5 por foto.

| folder_name | image_name | variedad | landmark | x | y |
|---|---|---|---|---|---|

`x`, `y` en píxeles de la foto (vertical); `(0, 0)` es la esquina superior izquierda y `y` crece hacia abajo. `variedad` es el nombre de la subcarpeta de la foto (o, si está suelta, lo que va antes del primer `_` del nombre).

**`resultados/medidas_hojas.csv`**: una fila por hoja analizada, con fecha, imagen, variedad, clase predicha, confianza, modelo usado, escalas, `L1–L4` (px y cm), los 5 ángulos y advertencias. Si analizas otra vez la misma imagen, su fila se reemplaza.

---

## 8. Clasificación KNN

La configuración está en las constantes (sección 1) de `clasificador_externo.py`:

| Constante | Qué controla |
|---|---|
| `DATASET_RUTA` | CSV de entrenamiento. Todas las columnas son atributos numéricos y la **última** es la clase. |
| `DATASET_EQUIVALENCIAS` | Relaciona el nombre de cada columna del dataset con la medida del analizador (p. ej. `"L1"` → `"L1_cm"`). |
| `KNN_K` | Número de vecinos. |
| `KNN_ESCALADO` | `"ninguno"`, `"normalizar"` (0–1) o `"estandarizar"` (z-score). |
| `KNN_ATRIBUTOS_USADOS` | `None` = todas las columnas, o una lista de nombres. |
| `KNN_PESOS` / `KNN_PESO_POR_DEFECTO` | Peso de cada atributo en la distancia. |

Conviene **normalizar**: las longitudes están en cm y los ángulos en grados; sin escalar, los grados dominan la distancia.

### Experimentador KNN (menú → "Experimentador KNN")

Elige un CSV, marca los atributos y sus pesos, fija K, el % de entrenamiento y el escalado, y presiona *Ejecutar KNN*. Cinco pestañas:

1. Salida del KNN (predicción de cada dato de prueba)
2. Reporte de métricas (precisión, recall, F1, matriz de confusión)
3. Detalle de votaciones (vecinos, distancias, empates)
4. Comparativa con la ejecución anterior
5. Elemento externo (clasificar un dato escrito a mano)

El split train/test es **estratificado por clase** y reproducible (semilla 42).

---

## 9. Uso por terminal (pruebas del analizador)

```bash
python src/analizador.py CARPETA_DE_LA_HOJA
python src/analizador.py CARPETA_DE_LA_HOJA --sin-llamar     # solo muestra el mensaje, no llama a Gemini
python src/analizador.py --listar-modelos
```

La carpeta debe tener `*_anotada.jpg`, `*_coordenadas.txt` y (opcional) `*_vertical.jpg`.
El anotador los crea en una carpeta temporal; para conservarlos pon
`CONSERVAR_ARCHIVOS_TEMPORALES = True` en `anotador_napari.py`.

---

## 10. Problemas comunes

| Síntoma | Solución |
|---|---|
| `Falta instalar o cargar el módulo: …` | `pip install -r requirements.txt` |
| `Falta GEMINI_API_KEY` | Crea el `.env` a partir de `.env.example` y escribe tu clave. |
| `Fallo el modelo …` | Cambia `GEMINI_MODEL` o agrega `GEMINI_MODELOS_RESPALDO`; lista modelos con `--listar-modelos`. |
| `No se encontro el dataset` | Revisa `DATASET_RUTA` en `clasificador_externo.py`. |
| `El analizador no entrego estas medidas…` | Los nombres de `DATASET_EQUIVALENCIAS` no coinciden con los encabezados del CSV. |
| Clic sin efecto | Asegúrate de que la ventana de napari esté activa y presiona **2**. |
| El menú no muestra el fondo | Falta `HojaMedidaCab1.jpg` junto a `main.py` (es solo estético). |