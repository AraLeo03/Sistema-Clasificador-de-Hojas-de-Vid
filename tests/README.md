# Pruebas unitarias

Desde la **raíz del proyecto** (donde está `main.py`):

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                      # todo
pytest -k claude            # solo las que mencionan "claude"
pytest tests/test_knn.py    # un archivo
pytest --cov=src            # cobertura (requiere pytest-cov)
```

Las pruebas **no usan la red ni tus API keys** (los SDK se reemplazan por dobles) y **no abren ventanas**
(Qt corre en modo `offscreen`). Si no tienes `data/dataset_70_30.csv` o `prompts/contextFile.md`, las pruebas
que dependen de ellos se omiten (SKIPPED); no es un error.

| Archivo | Qué prueba |
|---|---|
| `conftest.py` | Rutas, aislamiento del entorno, datos de ejemplo y doble de napari |
| `test_analizador_calculos.py` | Vectores, ángulos horarios, L1–L4, lectura de coordenadas y carpeta de la hoja |
| `test_analizador_respuesta.py` | Extraer y parsear el bloque de la IA, fila y CSV de medidas |
| `test_analizador_ia.py` | `.env` de proveedores, respaldo Gemini ⇄ Claude, peticiones a cada SDK y `analizar_hoja` completo |
| `test_knn.py` | Escalado, distancia, votación, empates, predicciones y métricas |
| `test_knn_panel.py` | Hilo del KNN (split 70/30 estratificado y reproducible) y limpieza de temporales |
| `test_clasificador_externo.py` | Dataset, constantes, equivalencias, K, escalado, pesos y dataset real |
| `test_anotador_apoyo.py` | Funciones de apoyo del anotador (sin napari) y contrato con el analizador |
| `test_dialogo_ia.py` | Pasos, resultado, error y cancelación del diálogo de espera |
| `test_proyecto.py` | Coherencia entre `contextFile.md` y el código |

**No cubierto** (necesita pantalla/napari): la clase `Anotador`, `main.py` y la interfaz del `PanelKNN`.