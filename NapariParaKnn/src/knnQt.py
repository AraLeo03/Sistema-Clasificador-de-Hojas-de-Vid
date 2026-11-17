"""
knnQt.py
========

Experimentador KNN (K vecinos mas cercanos) con interfaz Qt y sin librerias de
machine learning: toda la logica esta implementada a mano.

Contenido:
    Objeto, Distancia, Lector   modelos de datos y lectura de CSV
    KNN                         escalado, distancia, clasificacion y reportes de texto
    Estimaciones                metricas (precision, recall, F1...) y matriz de confusion
    HiloEjecucionKNN            QThread: split + KNN + metricas sin congelar la ventana
    PanelKNN                    widget Qt con 5 pestañas (lo usa main.py)

Formato del dataset CSV: primera fila = encabezados; TODAS las columnas son
atributos numericos y la ULTIMA es la clase.

Tambien lo reutiliza clasificador_externo.py (KNN, Lector, Objeto, Distancia).
"""

import os
import csv
import math
import random
import shutil
import tempfile

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QRadioButton, QButtonGroup, QTabWidget, QTextEdit,
    QFileDialog, QMessageBox, QScrollArea, QCheckBox, QGroupBox, QApplication
)

# Hojas de estilo (tema oscuro / claro). Hoy NO se aplican: el selector de tema
# esta comentado al final de PanelKNN (cambiar_tema).
ESTILO_OSCURO = """
    QWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    QGroupBox {
        border: 1px solid #333333;
        margin-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px;
    }
    QLineEdit, QSpinBox, QTextEdit {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 4px;
        padding: 4px;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        width: 16px;
    }
    QPushButton {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 5px 12px;
    }
    QPushButton:hover {
        background-color: #444444;
    }
    QTabWidget::pane {
        border: 1px solid #333333;
    }
    QTabBar::tab {
        background: #2d2d2d;
        color: #aaaaaa;
        padding: 6px 12px;
    }
    QTabBar::tab:selected {
        background: #3e3e3e;
        color: #ffffff;
        font-weight: bold;
    }
    QScrollArea {
        border: none;
    }
"""

ESTILO_CLARO = """
    QWidget {
        background-color: #f5f5f5;
        color: #2b2b2b;
    }
    QGroupBox {
        border: 1px solid #cccccc;
        margin-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px;
    }
    QLineEdit, QSpinBox, QTextEdit {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        width: 16px;
    }
    QPushButton {
        background-color: #e1e1e1;
        color: #000000;
        border: 1px solid #adadad;
        border-radius: 4px;
        padding: 5px 12px;
    }
    QPushButton:hover {
        background-color: #e5f1fb;
        border-color: #0078d7;
    }
    QTabWidget::pane {
        border: 1px solid #cccccc;
    }
    QTabBar::tab {
        background: #e1e1e1;
        color: #444444;
        padding: 6px 12px;
    }
    QTabBar::tab:selected {
        background: #ffffff;
        color: #000000;
        font-weight: bold;
    }
    QScrollArea {
        border: none;
    }
"""

# ==============================================================================
# CLASES DE SOPORTE (MODELOS DE DATOS)
# ==============================================================================

class Objeto:
    """Una fila del dataset: lista de atributos como TEXTO; el ultimo es la clase."""
    def __init__(self, atributos: list):
        self.atributos = [a.strip() for a in atributos]

    @property
    def clase(self) -> str:
        return self.atributos[-1]

    def valores_numericos(self, indices: list = None) -> list:
        """Atributos como float. indices=None -> todos menos la clase; si no, solo esas columnas."""
        if indices is None:
            return [float(x) for x in self.atributos[:-1]]
        return [float(self.atributos[i]) for i in indices]


class Distancia:
    """Resultado de comparar un dato contra uno de entrenamiento: distancia, su clase y sus atributos."""
    def __init__(self, valor: float, clase: str, atributos: list):
        self.valor = valor
        self.clase = clase
        self.atributos = atributos


class Lector:
    """Utilidades de lectura de archivos."""
    @staticmethod
    def leer_csv(ruta_csv: str) -> list:
        """Lee un CSV (omite el encabezado y filas vacias). Devuelve lista de Objeto; [] si no existe."""
        objetos = []
        if not os.path.exists(ruta_csv):
            return objetos
        with open(ruta_csv, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return objetos
            for row in reader:
                if row and any(field.strip() for field in row):
                    objetos.append(Objeto(row))
        return objetos


# ==============================================================================
# LÓGICA DEL ALGORITMO KNN
# ==============================================================================

class KNN:
    """Algoritmo KNN: escalado, distancia euclidiana ponderada, votacion y reportes."""
    @staticmethod
    def normalizar_datos(datos_train: list, datos_test: list, indices: list = None):
        """Escala a [0, 1] con min y max del ENTRENAMIENTO (se aplica igual a prueba).
        Devuelve copias (train, test); no modifica las listas originales.
        """
        train_norm = [Objeto(obj.atributos.copy()) for obj in datos_train]
        test_norm = [Objeto(obj.atributos.copy()) for obj in datos_test]

        if not train_norm:
            return train_norm, test_norm

        num_atributos = len(train_norm[0].atributos) - 1
        idx_list = indices if indices is not None else list(range(num_atributos))

        for idx in idx_list:
            vals_train = [float(obj.atributos[idx]) for obj in train_norm]
            min_val = min(vals_train)
            max_val = max(vals_train)
            rango = max_val - min_val

            for obj in train_norm:
                v = float(obj.atributos[idx])
                norm_v = (v - min_val) / rango if rango != 0 else 0.0
                obj.atributos[idx] = f"{norm_v:.6f}"

            for obj in test_norm:
                v = float(obj.atributos[idx])
                norm_v = (v - min_val) / rango if rango != 0 else 0.0
                obj.atributos[idx] = f"{norm_v:.6f}"

        return train_norm, test_norm

    @staticmethod
    def estandarizar_datos(datos_train: list, datos_test: list, indices: list = None):
        """Escala a z-score con media y desviacion del ENTRENAMIENTO (se aplica igual a prueba).
        Devuelve copias (train, test); no modifica las listas originales.
        """
        train_std = [Objeto(obj.atributos.copy()) for obj in datos_train]
        test_std = [Objeto(obj.atributos.copy()) for obj in datos_test]

        if not train_std:
            return train_std, test_std

        num_atributos = len(train_std[0].atributos) - 1
        idx_list = indices if indices is not None else list(range(num_atributos))

        for idx in idx_list:
            vals_train = [float(obj.atributos[idx]) for obj in train_std]
            n = len(vals_train)
            media = sum(vals_train) / n if n > 0 else 0.0
            varianza = sum((x - media) ** 2 for x in vals_train) / n if n > 0 else 0.0
            desv_std = math.sqrt(varianza)

            for obj in train_std:
                v = float(obj.atributos[idx])
                z_v = (v - media) / desv_std if desv_std != 0 else 0.0
                obj.atributos[idx] = f"{z_v:.6f}"

            for obj in test_std:
                v = float(obj.atributos[idx])
                z_v = (v - media) / desv_std if desv_std != 0 else 0.0
                obj.atributos[idx] = f"{z_v:.6f}"

        return train_std, test_std

    @staticmethod
    def calcular_distancia_euclidiana(a: Objeto, b: Objeto, indices: list = None, pesos: list = None) -> float:
        """sqrt(suma(peso * (a - b)^2)) sobre las columnas 'indices'. pesos=None -> todos 1.0."""
        val_a = a.valores_numericos(indices)
        val_b = b.valores_numericos(indices)
        if pesos is None:
            pesos = [1.0] * len(val_a)
        
        suma = sum(w * ((x - y) ** 2) for x, y, w in zip(val_a, val_b, pesos))
        return math.sqrt(suma)

    @classmethod
    def clasificar_dato(cls, dato_prueba: Objeto, datos_entrenamiento: list, k: int, indices: list = None, pesos: list = None):
        """Clasifica un dato por votacion de los k vecinos mas cercanos.

        Devuelve (clase_predicha, {clase: votos}). En caso de empate gana la clase
        empatada cuyo vecino esta mas cerca. Requiere k <= len(datos_entrenamiento).
        """
        distancias = []
        for ent in datos_entrenamiento:
            d = cls.calcular_distancia_euclidiana(dato_prueba, ent, indices, pesos)
            distancias.append(Distancia(d, ent.clase, ent.atributos))

        distancias.sort(key=lambda x: x.valor)

        conteo_clases = {}
        for i in range(k):
            clase = distancias[i].clase
            conteo_clases[clase] = conteo_clases.get(clase, 0) + 1

        max_votos = max(conteo_clases.values())
        clases_empatadas = [c for c, v in conteo_clases.items() if v == max_votos]

        # Desempate: el primer vecino (ya ordenados por distancia) que pertenezca a una clase empatada.
        if len(clases_empatadas) > 1:
            clase_predicha = ""
            for d in distancias:
                if d.clase in clases_empatadas:
                    clase_predicha = d.clase
                    break
        else:
            clase_predicha = clases_empatadas[0]

        return clase_predicha, conteo_clases

    @classmethod
    def ejecutar(cls, ruta_entrenamiento: str, ruta_prueba: str, k: int, ruta_salida: str, indices: list = None, tipo_escalado: str = "ninguno", pesos: list = None) -> str:
        """Clasifica todo el CSV de prueba y arma el reporte de texto (pestana 1).
        Tambien guarda las predicciones en ruta_salida (ver guardar_predicciones).
        """
        datos_entrenamiento = Lector.leer_csv(ruta_entrenamiento)
        datos_prueba = Lector.leer_csv(ruta_prueba)

        if not datos_entrenamiento or not datos_prueba:
            return "ERROR: No se pudieron cargar los datos."

        if tipo_escalado == "normalizar":
            datos_entrenamiento, datos_prueba = cls.normalizar_datos(datos_entrenamiento, datos_prueba, indices)
        elif tipo_escalado == "estandarizar":
            datos_entrenamiento, datos_prueba = cls.estandarizar_datos(datos_entrenamiento, datos_prueba, indices)

        num_atributos = len(indices) if indices is not None else (len(datos_entrenamiento[0].atributos) - 1)
        clases_unicas = sorted(list({obj.clase for obj in datos_entrenamiento}))

        conteo_train = {}
        for obj in datos_entrenamiento:
            conteo_train[obj.clase] = conteo_train.get(obj.clase, 0) + 1

        conteo_test = {}
        for obj in datos_prueba:
            conteo_test[obj.clase] = conteo_test.get(obj.clase, 0) + 1
        
        sb = []
        sb.append("=== INICIANDO PREDICCIONES (KNN) ===")
        sb.append(f"K = {k}")
        sb.append(f"Escalado de Datos: {tipo_escalado.upper()}")
        sb.append(f"Entrenamiento: {ruta_entrenamiento}")
        sb.append(f"Prueba: {ruta_prueba}")
        sb.append(f"Atributos seleccionados (total): {num_atributos}")
        if pesos is not None:
            sb.append(f"Pesos asignados: {pesos}")
        sb.append(f"Clases detectadas: {clases_unicas}\n")

        sb.append("\n--- distribucion por clases ---")
        sb.append(f"Train ({len(datos_entrenamiento)} elems total):")
        for cls_name, cant in conteo_train.items():
            sb.append(f"  - {cls_name}: {cant}")

        sb.append(f"Test ({len(datos_prueba)} elems total):")
        for cls_name, cant in conteo_test.items():
            sb.append(f"  - {cls_name}: {cant}")

        cls.guardar_predicciones(datos_prueba, datos_entrenamiento, k, ruta_salida, indices, pesos)
        sb.append(f"\nPredicciones guardadas en: {ruta_salida}\n")

        sb.append("--- Detalle por instancia ---")
        sb.append(f"{'inst#':<6} {'actual':<12} {'predicted':<12} {'error':<6} {'confianza':<10}")
        sb.append("-" * 52)

        for i, dato_prueba in enumerate(datos_prueba):
            clase_real = dato_prueba.clase
            clase_predicha, conteos = cls.clasificar_dato(dato_prueba, datos_entrenamiento, k, indices, pesos)
            error = "" if clase_real == clase_predicha else "*"

            votos_ganador = conteos[clase_predicha]
            total_votos = sum(conteos.values())
            porcentaje = (votos_ganador / total_votos) * 100.0

            sb.append(f"{i + 1:<6} {clase_real:<12} {clase_predicha:<12} {error:<6} {porcentaje:<10.2f}%")

        sb.append("\n=== CLASIFICACION COMPLETADA ===")
        return "\n".join(sb)

    @classmethod
    def guardar_predicciones(cls, datos_prueba: list, datos_entrenamiento: list, k: int, nombre_salida: str, indices: list = None, pesos: list = None):
        """Escribe un CSV con inst#, actual, predicted, error ('*' si fallo) y confianza."""
        try:
            with open(nombre_salida, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["inst#", "actual", "predicted", "error", "prediction"])

                for i, dato_prueba in enumerate(datos_prueba):
                    clase_real = dato_prueba.clase
                    clase_predicha, conteos = cls.clasificar_dato(dato_prueba, datos_entrenamiento, k, indices, pesos)
                    error = "" if clase_real == clase_predicha else "*"

                    votos_ganador = conteos[clase_predicha]
                    total_votos = sum(conteos.values())
                    porcentaje = (votos_ganador / total_votos) * 100.0

                    writer.writerow([i + 1, clase_real, clase_predicha, error, f"{porcentaje:.2f}%"])
        except Exception as e:
            print(f"Error al guardar predicciones: {e}")

    @classmethod
    def ejecutar_votaciones(cls, ruta_entrenamiento: str, ruta_prueba: str, k: int, encabezados: list, indices: list = None, tipo_escalado: str = "ninguno", pesos: list = None) -> str:
        """Reporte detallado (pestana 3): por cada dato de prueba, sus k vecinos con distancia,
        los votos y si hubo empate. Las clases se muestran como numeros (ver leyenda).
        """
        datos_entrenamiento = Lector.leer_csv(ruta_entrenamiento)
        datos_prueba = Lector.leer_csv(ruta_prueba)

        if not datos_entrenamiento or not datos_prueba:
            return "ERROR: No se pudieron cargar los datos."

        if tipo_escalado == "normalizar":
            datos_entrenamiento, datos_prueba = cls.normalizar_datos(datos_entrenamiento, datos_prueba, indices)
        elif tipo_escalado == "estandarizar":
            datos_entrenamiento, datos_prueba = cls.estandarizar_datos(datos_entrenamiento, datos_prueba, indices)

        mapa_clases = {}
        contador = 0
        for obj in datos_entrenamiento:
            clase = obj.clase
            if clase not in mapa_clases:
                mapa_clases[clase] = contador
                contador += 1

        sb = []
        sb.append("=== LEYENDA DE CLASES ===")
        for clase, num in mapa_clases.items():
            sb.append(f"  {num} = {clase}")
        sb.append("")

        if indices is not None:
            cabecera_cols = [encabezados[idx].strip() for idx in indices]
        else:
            cabecera_cols = [enc.strip() for enc in encabezados[:-1]]
            
        cabecera_str = "| " + " | ".join(cabecera_cols) + " | Clase |"

        total_empates = 0

        for i, dato_prueba in enumerate(datos_prueba):
            clase_real = dato_prueba.clase
            num_real = mapa_clases.get(clase_real, -1)

            distancias = []
            for ent in datos_entrenamiento:
                d = cls.calcular_distancia_euclidiana(dato_prueba, ent, indices, pesos)
                distancias.append(Distancia(d, ent.clase, ent.atributos))

            distancias.sort(key=lambda x: x.valor)

            conteo = {}
            for j in range(k):
                c = distancias[j].clase
                conteo[c] = conteo.get(c, 0) + 1

            max_votos = max(conteo.values())
            empatadas = [c for c, v in conteo.items() if v == max_votos]

            empate = len(empatadas) > 1
            if empate:
                clase_predicha = empatadas[0]
                for d in distancias:
                    if d.clase in empatadas:
                        clase_predicha = d.clase
                        break
                total_empates += 1
            else:
                clase_predicha = empatadas[0]

            num_predicho = mapa_clases.get(clase_predicha, -1)
            conteo_num = {str(mapa_clases.get(k_cls, -1)): v for k_cls, v in conteo.items()}

            sb.append(f"Dato de prueba {i + 1}:")
            sb.append(cabecera_str)

            vals_prueba = [dato_prueba.atributos[idx].strip() for idx in (indices if indices else range(len(dato_prueba.atributos)-1))]
            fila_prueba = "| " + " | ".join(vals_prueba) + f" | {num_real} |"
            sb.append(fila_prueba)

            sb.append("Datos de entrenamiento mas cercanos:")
            sb.append(cabecera_str)

            for j in range(k):
                d = distancias[j]
                num_vec = mapa_clases.get(d.clase, -1)
                vals_vec = [d.atributos[idx].strip() for idx in (indices if indices else range(len(d.atributos)-1))]
                fila_vec = "| " + " | ".join(vals_vec) + f" | {num_vec} |"
                sb.append(fila_vec)
                sb.append(f"Distancia Ponderada: {d.valor:.6f}")

            res_line = f"Prediccion = {num_predicho}, Realidad = {num_real}"
            if clase_predicha != clase_real:
                res_line += "  *"
            sb.append(res_line)

            votos_str = f"Votos: {conteo_num}"
            if empate:
                votos_str += "  EMPATE -> desempate por vecino mas cercano"
            sb.append(votos_str + "\n")

        sb.append("=" * 50)
        sb.append("=== RESUMEN DE VOTACIONES ===")
        sb.append(f"Total de instancias evaluadas: {len(datos_prueba)}")
        sb.append(f"Total de empates ocurridos   : {total_empates}")
        if total_empates > 0:
            sb.append("Los empates se resolvieron por vecino mas cercano.")
        else:
            sb.append("No hubo empates en esta ejecucion.")

        return "\n".join(sb)


# ==============================================================================
# REPORTES Y MÉTRICAS
# ==============================================================================

class Estimaciones:
    """Metricas de evaluacion por clase a partir del CSV de predicciones."""
    @classmethod
    def ejecutar(cls, ruta_predicciones: str, base_salida: str) -> tuple[str, dict]:
        """Calcula metricas, escribe _Evaluacion.csv, _Resumen.csv y _Matriz.csv (con base_salida
        como prefijo) y devuelve (texto_del_reporte, dict con promedios, clases y matriz).
        """
        resultados = cls._leer_resultados(ruta_predicciones)
        if not resultados:
            return "ERROR: No se pudieron leer resultados del archivo.", {}

        clases_set = []
        for r in resultados:
            if r['actual'] not in clases_set and r['actual']:
                clases_set.append(r['actual'])
            if r['predicted'] not in clases_set and r['predicted']:
                clases_set.append(r['predicted'])

        clases_set.sort()

        ruta_evaluacion = base_salida + "_Evaluacion.csv"
        ruta_resumen = base_salida + "_Resumen.csv"
        ruta_matriz = base_salida + "_Matriz.csv"

        metricas_por_clase = cls._calcular_metricas(resultados, clases_set)

        cls._guardar_evaluacion(resultados, clases_set, ruta_evaluacion)
        cls._guardar_resumen(metricas_por_clase, ruta_resumen)
        cls._guardar_matriz(resultados, clases_set, ruta_matriz)

        num_clases = len(clases_set) if clases_set else 1
        avg_prec = sum(m['Precision'] for m in metricas_por_clase.values()) / num_clases
        avg_rec = sum(m['Recall'] for m in metricas_por_clase.values()) / num_clases
        avg_acc = sum(m['Accuracy'] for m in metricas_por_clase.values()) / num_clases
        avg_f1 = sum(m['F1'] for m in metricas_por_clase.values()) / num_clases

        matriz = {}
        for real in clases_set:
            matriz[real] = {}
            for pred in clases_set:
                count = sum(1 for r in resultados if r['actual'] == real and r['predicted'] == pred)
                matriz[real][pred] = count

        resumen_dict = {
            "precision": avg_prec,
            "recall": avg_rec,
            "accuracy": avg_acc,
            "f1": avg_f1,
            "clases": clases_set,
            "matriz": matriz
        }

        sb = []
        sb.append(cls._generar_texto_resumen(metricas_por_clase, resultados, clases_set))
        sb.append("\nArchivos generados:")
        sb.append(f"- {ruta_evaluacion}")
        sb.append(f"- {ruta_resumen}")
        sb.append(f"- {ruta_matriz}")

        return "\n".join(sb), resumen_dict

    @staticmethod
    def _leer_resultados(ruta: str) -> list:
        """Lee el CSV de predicciones como lista de dicts {id, actual, predicted}."""
        resultados = []
        if not os.path.exists(ruta):
            return resultados
        with open(ruta, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                resultados.append({
                    'id': row['inst#'],
                    'actual': row['actual'].strip(),
                    'predicted': row['predicted'].strip()
                })
        return resultados

    @staticmethod
    def _calcular_metricas(resultados: list, clases: list) -> dict:
        """Por clase: TP, FP, FN, TN, precision, recall, F1 y accuracy (enfoque uno-contra-todos)."""
        metricas = {}
        total = len(resultados)

        for c in clases:
            tp = sum(1 for r in resultados if r['actual'] == c and r['predicted'] == c)
            fp = sum(1 for r in resultados if r['actual'] != c and r['predicted'] == c)
            fn = sum(1 for r in resultados if r['actual'] == c and r['predicted'] != c)
            tn = sum(1 for r in resultados if r['actual'] != c and r['predicted'] != c)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / total if total > 0 else 0.0

            metricas[c] = {
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy
            }
        return metricas

    @staticmethod
    def _guardar_evaluacion(resultados: list, clases: list, ruta: str):
        """CSV con la etiqueta TP/FP/FN/TN de cada instancia para cada clase."""
        with open(ruta, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["inst#", "actual", "predicted"] + [f"Eval_{c}" for c in clases]
            writer.writerow(header)
            for r in resultados:
                row = [r['id'], r['actual'], r['predicted']]
                for c in clases:
                    if r['actual'] == c and r['predicted'] == c:
                        row.append("TP")
                    elif r['actual'] != c and r['predicted'] == c:
                        row.append("FP")
                    elif r['actual'] == c and r['predicted'] != c:
                        row.append("FN")
                    else:
                        row.append("TN")
                writer.writerow(row)

    @staticmethod
    def _guardar_resumen(metricas: dict, ruta: str):
        """CSV con las metricas de cada clase."""
        with open(ruta, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Clase", "TP", "FP", "TN", "FN", "Precision", "Recall", "Accuracy", "F1-Score"])
            for c, m in metricas.items():
                writer.writerow([
                    c, m['TP'], m['FP'], m['TN'], m['FN'],
                    f"{m['Precision']:.3f}", f"{m['Recall']:.3f}",
                    f"{m['Accuracy']:.3f}", f"{m['F1']:.3f}"
                ])

    @staticmethod
    def _guardar_matriz(resultados: list, clases: list, ruta: str):
        """CSV con la matriz de confusion (filas = real, columnas = predicho)."""
        with open(ruta, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Real \\ Predicho"] + clases)
            for real in clases:
                fila = [real]
                for pred in clases:
                    count = sum(1 for r in resultados if r['actual'] == real and r['predicted'] == pred)
                    fila.append(count)
                writer.writerow(fila)

    @staticmethod
    def _generar_texto_resumen(metricas: dict, resultados: list, clases: list) -> str:
        """Texto para la pestana 2: tabla por clase, promedios y matriz de confusion."""
        sb = []
        sb.append("=== REPORTE DE METRICAS ===\n")
        
        sb.append("--- Resultados por clase ---")
        sb.append(f"{'Clase':<22} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}   {'Prec':>9} {'Rec':>9} {'Acc':>9} {'F1':>9}")
        
        sum_prec = sum_rec = sum_acc = sum_f1 = 0.0
        num_clases = len(clases) if clases else 1

        for c in clases:
            m = metricas[c]
            sb.append(
                f"{c:<22} {m['TP']:>4} {m['FP']:>4} {m['TN']:>4} {m['FN']:>4}   "
                f"{m['Precision']:>9.3f} {m['Recall']:>9.3f} {m['Accuracy']:>9.3f} {m['F1']:>9.3f}"
            )
            sum_prec += m['Precision']
            sum_rec += m['Recall']
            sum_acc += m['Accuracy']
            sum_f1 += m['F1']

        sb.append("\n--- Promedio General ---")
        avg_prec = sum_prec / num_clases
        avg_rec = sum_rec / num_clases
        avg_acc = sum_acc / num_clases
        avg_f1 = sum_f1 / num_clases
        
        sb.append(f"{'Media':<41} {avg_prec:>9.3f} {avg_rec:>9.3f} {avg_acc:>9.3f} {avg_f1:>9.3f}")

        sb.append("\n--- Matriz de confusion ---")
        letras = [chr(97 + i) for i in range(len(clases))]
        encabezado_letras = "  ".join(letras)
        sb.append(f"   {encabezado_letras}  <-- classified as")

        for i, real in enumerate(clases):
            letra_real = letras[i]
            fila_counts = []
            for pred in clases:
                count = sum(1 for r in resultados if r['actual'] == real and r['predicted'] == pred)
                fila_counts.append(f"{count:>2}")
            
            str_counts = " ".join(fila_counts)
            sb.append(f"  {str_counts}  | {letra_real} = {real}")

        return "\n".join(sb)


# ==============================================================================
# HILO DE EJECUCIÓN SECUNDARIO
# ==============================================================================

class HiloEjecucionKNN(QThread):
    """Hilo (QThread) que hace todo el trabajo pesado: split -> KNN -> metricas -> votaciones.
    Emite 'finalizado' con los textos de las pestanas o 'error' con el mensaje.
    """
    # Transmite (salida_knn, salida_est, salida_votaciones, config_actual, metricas_dict)
    finalizado = Signal(str, str, str, dict, dict)
    error = Signal(str)

    def __init__(self, ruta_csv, k, split_porcentaje, indices, pesos, tipo_escalado, temp_files):
        super().__init__()
        self.ruta_csv = ruta_csv
        self.k = k
        self.split_porcentaje = split_porcentaje
        self.indices = indices
        self.pesos = pesos
        self.tipo_escalado = tipo_escalado
        self.temp_files = temp_files

    def run(self):
        """1) Divide el dataset por clase (estratificado) segun el % de entrenamiento y escribe
        los CSV temporales; 2) corre KNN, Estimaciones y votaciones; 3) emite el resultado.
        """
        try:
            temp_train, temp_test, temp_pred, temp_met = self.temp_files
            
            # Split dataset
            # Semilla fija: el split es reproducible (mismos resultados en cada ejecucion).
            random.seed(42)
            with open(self.ruta_csv, mode='r', encoding='utf-8') as f:
                lineas = [l.strip() for l in f.readlines() if l.strip()]

            if len(lineas) < 2:
                raise Exception("El archivo está vacío o solo tiene encabezado.")

            encabezado = lineas[0]
            datos = lineas[1:]

            agrupado_por_clase = {}
            for fila in datos:
                clase = fila.split(",")[-1].strip()
                agrupado_por_clase.setdefault(clase, []).append(fila)

            train_datos, test_datos = [], []
            for clase, filas in agrupado_por_clase.items():
                random.shuffle(filas)
                limite_clase = int(round(len(filas) * (self.split_porcentaje / 100.0)))
                train_datos.extend(filas[:limite_clase])
                test_datos.extend(filas[limite_clase:])

            random.shuffle(train_datos)
            random.shuffle(test_datos)

            with open(temp_train, mode='w', encoding='utf-8') as f:
                for l in [encabezado] + train_datos: f.write(l + "\n")
            with open(temp_test, mode='w', encoding='utf-8') as f:
                for l in [encabezado] + test_datos: f.write(l + "\n")

            # 1. Salida del KNN (Pestaña 1)
            salida_knn = KNN.ejecutar(temp_train, temp_test, self.k, temp_pred, self.indices, self.tipo_escalado, self.pesos)
            
            # 2. Métricas y estimaciones (Pestaña 2)
            salida_est, metricas_dict = Estimaciones.ejecutar(temp_pred, temp_met)
            
            # 3. Detalle de votaciones (Pestaña 3)
            encabezados = []
            with open(self.ruta_csv, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                encabezados = next(reader, [])

            salida_votaciones = KNN.ejecutar_votaciones(temp_train, temp_test, self.k, encabezados, self.indices, self.tipo_escalado, self.pesos)

            config_actual = {'k': self.k, 'split': self.split_porcentaje, 'escalado': self.tipo_escalado}

            self.finalizado.emit(salida_knn, salida_est, salida_votaciones, config_actual, metricas_dict)

        except Exception as ex:
            self.error.emit(str(ex))


# ==============================================================================
# INTERFAZ GRÁFICA EN QT
# ==============================================================================

class PanelKNN(QWidget):
    """Panel completo del experimentador KNN.

    Secciones: 1) elegir dataset, atributos y pesos; 2) K, % de train, escalado y
    boton Ejecutar; 3) pestanas: Salida KNN, Metricas, Votaciones, Comparativa con la
    ejecucion anterior y Elemento Externo (clasificar un dato escrito a mano).
    Senal volver_al_menu: la conecta main.py para regresar al menu.
    """
    #Señal para volver al menu principal
    volver_al_menu = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Archivos temporales en una carpeta propia del sistema (no ensucian la carpeta del
        # proyecto); se borran al cerrar con limpiar_temporales().
        # TEMP_TRAIN tambien lo usa la pestana 5 para clasificar con el mismo 'train' de la ultima ejecucion.
        self.dir_temp = tempfile.mkdtemp(prefix="knn_")
        self.TEMP_TRAIN = os.path.join(self.dir_temp, "split_train.csv")
        self.TEMP_TEST = os.path.join(self.dir_temp, "split_test.csv")
        self.TEMP_PRED = os.path.join(self.dir_temp, "predicciones_knn.csv")
        self.TEMP_MET = os.path.join(self.dir_temp, "metricas_knn")   # prefijo de _Evaluacion/_Resumen/_Matriz.csv
        self.vars_atributos = []
        self.entries_pesos = []
        self.encabezados = []
        self.ultimo_resultado = None
        self.entries_externo = {}

        self.crear_componentes()

    def limpiar_temporales(self):
        """Borra la carpeta temporal del KNN (splits, predicciones y metricas)."""
        shutil.rmtree(self.dir_temp, ignore_errors=True)

    def crear_componentes(self):
        """Construye toda la interfaz (boton Volver, secciones 1 y 2 y las pestanas)."""
        layout_principal = QVBoxLayout(self)

        # -------------------------------------------------------------
        # BOTÓN VOLVER AL MENÚ PRINCIPAL
        # -------------------------------------------------------------
        top_layout = QHBoxLayout()
        btn_volver_menu = QPushButton("Volver")
        btn_volver_menu.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 5px 12px;")
        btn_volver_menu.clicked.connect(self.volver_al_menu.emit)
        top_layout.addWidget(btn_volver_menu)
        top_layout.addStretch()
        layout_principal.addLayout(top_layout)
        # -------------------------------------------------------------

        # 1. Dataset CSV
        gb_dataset = QGroupBox("1. Dataset CSV")
        l_dataset = QVBoxLayout(gb_dataset)

        l_archivo = QHBoxLayout()
        l_archivo.addWidget(QLabel("Archivo CSV:"))
        self.txt_ruta_dataset = QLineEdit()
        self.txt_ruta_dataset.setReadOnly(True)
        l_archivo.addWidget(self.txt_ruta_dataset)
        btn_buscar = QPushButton("Seleccionar")
        btn_buscar.clicked.connect(self.seleccionar_dataset)
        l_archivo.addWidget(btn_buscar)
        l_dataset.addLayout(l_archivo)

        l_dataset.addWidget(QLabel("Selecciona atributos e ingresa sus pesos:"))

        # Scroll área horizontal para lista de atributos
        self.scroll_chk = QScrollArea()
        self.scroll_chk.setWidgetResizable(True)
        self.scroll_chk.setFixedHeight(95)
        self.scroll_chk.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.widget_chk = QWidget()
        self.layout_chk = QHBoxLayout(self.widget_chk)
        self.layout_chk.setContentsMargins(5, 5, 5, 5)
        self.layout_chk.setSpacing(15)
        self.layout_chk.setAlignment(Qt.AlignLeft)
        self.scroll_chk.setWidget(self.widget_chk)
        l_dataset.addWidget(self.scroll_chk)

        lbl_nota = QLabel("Formato: columnas de atributos + columna Clase al final. El programa aplica split automaticamente.")
        lbl_nota.setStyleSheet("font-style: italic; color: gray; font-size: 11px;")
        l_dataset.addWidget(lbl_nota)

        layout_principal.addWidget(gb_dataset)

        # 2. Configuración y Ejecución KNN
        gb_config = QGroupBox("2. Configuracion y Ejecucion KNN")
        l_config = QHBoxLayout(gb_config)

        l_config.addWidget(QLabel("Valor de K:"))
        self.spinner_k = QSpinBox()
        self.spinner_k.setRange(1, 100)
        self.spinner_k.setValue(3)
        self.spinner_k.setFixedWidth(85)
        l_config.addWidget(self.spinner_k)

        l_config.addWidget(QLabel("Split Train (%):"))
        self.spinner_split = QSpinBox()
        self.spinner_split.setRange(10, 95)
        self.spinner_split.setValue(70)
        self.spinner_split.setFixedWidth(85)
        l_config.addWidget(self.spinner_split)

        self.btn_group_escalado = QButtonGroup(self)
        self.rb_ninguno = QRadioButton("Sin escalado")
        self.rb_normalizar = QRadioButton("Normalizar")
        self.rb_estandarizar = QRadioButton("Estandarizar")
        self.rb_normalizar.setChecked(True)

        self.btn_group_escalado.addButton(self.rb_ninguno, 0)
        self.btn_group_escalado.addButton(self.rb_normalizar, 1)
        self.btn_group_escalado.addButton(self.rb_estandarizar, 2)

        l_config.addWidget(self.rb_ninguno)
        l_config.addWidget(self.rb_normalizar)
        l_config.addWidget(self.rb_estandarizar)

        # self.chk_modo_oscuro = QCheckBox("🌙")
        # self.chk_modo_oscuro.toggled.connect(self.cambiar_tema)
        # l_config.addWidget(self.chk_modo_oscuro)

        l_config.addStretch()

        self.btn_ejecutar = QPushButton("Ejecutar KNN")
        self.btn_ejecutar.setStyleSheet("background-color: #2E86C1; color: white; font-weight: bold; border-radius: 3px; padding: 6px 14px;")
        self.btn_ejecutar.clicked.connect(self.ejecutar_knn)
        l_config.addWidget(self.btn_ejecutar)

        layout_principal.addWidget(gb_config)

        # 3. Pestañas de Resultados
        self.pestanas = QTabWidget()
        self.txt_salida_knn = self.crear_tab_texto("1. Salida del KNN")
        self.txt_salida_estimaciones = self.crear_tab_texto("2. Reporte de Metricas")
        self.txt_salida_votaciones = self.crear_tab_texto("3. Detalle de Votaciones")
        self.txt_salida_comparativa = self.crear_tab_texto("4. Comparativa con Ejecucion Anterior")

        # Pestaña 5: Elemento Externo
        self.tab_externo = QWidget()
        self.crear_interfaz_externo()
        self.pestanas.addTab(self.tab_externo, "5. Elemento Externo")

        layout_principal.addWidget(self.pestanas)

    def crear_tab_texto(self, titulo):
        """Agrega una pestana con un cuadro de texto de solo lectura (fuente monoespaciada) y lo devuelve."""
        txt = QTextEdit()
        txt.setFontFamily("Courier New")
        txt.setFontPointSize(9)
        txt.setLineWrapMode(QTextEdit.NoWrap)
        txt.setReadOnly(True)
        self.pestanas.addTab(txt, titulo)
        return txt

    def crear_interfaz_externo(self):
        """Construye la pestana 5: campos de valores, boton Clasificar y cuadro de resultado."""
        layout = QVBoxLayout(self.tab_externo)
        
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Ingresa los valores numéricos para cada atributo del nuevo elemento:"))
        btn_clasificar_ext = QPushButton("Clasificar Elemento")
        btn_clasificar_ext.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold; border-radius: 3px; padding: 6px 12px;")
        btn_clasificar_ext.clicked.connect(self.clasificar_elemento_externo)
        top_bar.addWidget(btn_clasificar_ext)
        layout.addLayout(top_bar)

        self.scroll_ext = QScrollArea()
        self.scroll_ext.setWidgetResizable(True)
        self.scroll_ext.setFixedHeight(95)
        self.scroll_ext.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.widget_ext = QWidget()
        self.layout_ext = QHBoxLayout(self.widget_ext)
        self.layout_ext.setContentsMargins(5, 5, 5, 5)
        self.layout_ext.setSpacing(15)
        self.layout_ext.setAlignment(Qt.AlignLeft)
        self.scroll_ext.setWidget(self.widget_ext)
        layout.addWidget(self.scroll_ext)

        self.txt_salida_externo = QTextEdit()
        self.txt_salida_externo.setFontFamily("Courier New")
        self.txt_salida_externo.setFontPointSize(9)
        self.txt_salida_externo.setLineWrapMode(QTextEdit.NoWrap)
        self.txt_salida_externo.setReadOnly(True)
        layout.addWidget(self.txt_salida_externo)

    def seleccionar_dataset(self):
        """Pide un CSV, lee sus encabezados y crea un checkbox + peso (seccion 1) y un campo de valor (pestana 5) por columna."""
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar Dataset CSV", "", "Archivos CSV (*.csv)")
        if ruta:
            self.txt_ruta_dataset.setText(ruta)

            # Limpiar elementos previos
            for i in reversed(range(self.layout_chk.count())):
                item = self.layout_chk.takeAt(i)
                if item.widget(): item.widget().deleteLater()

            for i in reversed(range(self.layout_ext.count())):
                item = self.layout_ext.takeAt(i)
                if item.widget(): item.widget().deleteLater()

            self.vars_atributos = []
            self.entries_pesos = []
            self.entries_externo = {}

            with open(ruta, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.encabezados = next(reader, [])

            for idx, col in enumerate(self.encabezados[:-1]):
                # Componente para Dataset CSV (Sección 1)
                item_w = QWidget()
                v_box = QVBoxLayout(item_w)
                v_box.setContentsMargins(0, 0, 0, 0)
                v_box.setSpacing(4)
                v_box.setAlignment(Qt.AlignCenter)

                chk = QCheckBox(col.strip())
                chk.setChecked(True)
                ent_peso = QLineEdit("1.0")
                ent_peso.setFixedWidth(50)
                ent_peso.setAlignment(Qt.AlignCenter)

                v_box.addWidget(chk, alignment=Qt.AlignCenter)
                v_box.addWidget(ent_peso, alignment=Qt.AlignCenter)

                self.layout_chk.addWidget(item_w)
                self.vars_atributos.append(chk)
                self.entries_pesos.append(ent_peso)

                # Componente para Elemento Externo (Pestaña 5)
                item_ext = QWidget()
                v_box_ext = QVBoxLayout(item_ext)
                v_box_ext.setContentsMargins(0, 0, 0, 0)
                v_box_ext.setSpacing(4)
                v_box_ext.setAlignment(Qt.AlignCenter)

                lbl_attr = QLabel(col.strip())
                lbl_attr.setStyleSheet("font-weight: bold;")
                ent_val = QLineEdit("0.0")
                ent_val.setFixedWidth(65)
                ent_val.setAlignment(Qt.AlignCenter)

                v_box_ext.addWidget(lbl_attr, alignment=Qt.AlignCenter)
                v_box_ext.addWidget(ent_val, alignment=Qt.AlignCenter)

                self.layout_ext.addWidget(item_ext)
                self.entries_externo[idx] = ent_val

            self.txt_salida_externo.setText("Ingresa los valores de las características en la parte superior y presiona 'Clasificar Elemento'.")

    def ejecutar_knn(self):
        """Valida atributos/pesos, lee el escalado elegido y lanza HiloEjecucionKNN."""
        ruta_csv = self.txt_ruta_dataset.text()
        if not ruta_csv:
            QMessageBox.warning(self, "Atención", "Selecciona el archivo CSV primero.")
            return

        indices_seleccionados = []
        pesos_seleccionados = []

        for i, (chk, ent_peso) in enumerate(zip(self.vars_atributos, self.entries_pesos)):
            if chk.isChecked():
                indices_seleccionados.append(i)
                try:
                    p = float(ent_peso.text())
                    if p < 0: raise ValueError()
                    pesos_seleccionados.append(p)
                except ValueError:
                    QMessageBox.critical(self, "Error", f"Peso inválido en el atributo {self.encabezados[i]}")
                    return

        if not indices_seleccionados:
            QMessageBox.warning(self, "Atención", "Selecciona al menos un atributo en el Checklist.")
            return

        tipo_escalado = "ninguno"
        if self.rb_normalizar.isChecked(): tipo_escalado = "normalizar"
        elif self.rb_estandarizar.isChecked(): tipo_escalado = "estandarizar"

        self.btn_ejecutar.setEnabled(False)

        temp_files = (self.TEMP_TRAIN, self.TEMP_TEST, self.TEMP_PRED, self.TEMP_MET)
        self.hilo = HiloEjecucionKNN(
            ruta_csv, self.spinner_k.value(), self.spinner_split.value(),
            indices_seleccionados, pesos_seleccionados, tipo_escalado, temp_files
        )
        self.hilo.finalizado.connect(self.finalizar_proceso)
        self.hilo.error.connect(lambda err: (QMessageBox.critical(self, "Error", err), self.btn_ejecutar.setEnabled(True)))
        self.hilo.start()

    def finalizar_proceso(self, salida_knn, salida_est, salida_votaciones, config_actual, metricas_actual):
        """Recibe los resultados del hilo, llena las 4 primeras pestanas y guarda la ejecucion para la comparativa."""
        # 1. Salida KNN -> Pestaña 1
        self.txt_salida_knn.setText(salida_knn)

        # 2. Métricas -> Pestaña 2
        self.txt_salida_estimaciones.setText(salida_est)

        # 3. Detalle de Votaciones -> Pestaña 3
        self.txt_salida_votaciones.setText(salida_votaciones)

        # 4. Comparativa -> Pestaña 4
        texto_comparativa = self._generar_texto_comparativa(config_actual, metricas_actual)
        self.txt_salida_comparativa.setText(texto_comparativa)

        # Guardar ejecución actual para posterior comparación
        self.ultimo_resultado = {
            'config': config_actual,
            'metricas': metricas_actual
        }

        self.btn_ejecutar.setEnabled(True)
        self.pestanas.setCurrentIndex(0)
        QMessageBox.information(self, "Listo", "Análisis completo finalizado. Revisa las pestañas.")

    def clasificar_elemento_externo(self):
        """Clasifica un dato escrito a mano (pestana 5) con los mismos ajustes (K, escalado, atributos, pesos)."""
        ruta_csv = self.txt_ruta_dataset.text()
        if not ruta_csv or not self.encabezados:
            QMessageBox.warning(self, "Atención", "Carga un archivo CSV primero.")
            return

        indices_seleccionados = []
        pesos_seleccionados = []

        for i, (chk, ent_peso) in enumerate(zip(self.vars_atributos, self.entries_pesos)):
            if chk.isChecked():
                indices_seleccionados.append(i)
                try:
                    p = float(ent_peso.text())
                    if p < 0: raise ValueError()
                    pesos_seleccionados.append(p)
                except ValueError:
                    QMessageBox.critical(self, "Error", f"Peso inválido en el atributo {self.encabezados[i]}")
                    return

        if not indices_seleccionados:
            QMessageBox.warning(self, "Atención", "Debes seleccionar al menos un atributo en el Checklist.")
            return

        valores_externo = []
        for i in range(len(self.encabezados) - 1):
            try:
                v = float(self.entries_externo[i].text())
                valores_externo.append(f"{v}")
            except ValueError:
                QMessageBox.critical(self, "Error", f"El valor para '{self.encabezados[i]}' debe ser numérico.")
                return

        valores_externo.append("Desconocida")
        obj_externo = Objeto(valores_externo)

        # Si ya se ejecuto el KNN usa el 'train' del split; si no, usa el CSV completo.
        ruta_train = self.TEMP_TRAIN if os.path.exists(self.TEMP_TRAIN) else ruta_csv
        datos_train = Lector.leer_csv(ruta_train)

        if not datos_train:
            QMessageBox.critical(self, "Error", "No se pudo cargar el dataset de entrenamiento.")
            return

        k = self.spinner_k.value()

        tipo_escalado = "ninguno"
        if self.rb_normalizar.isChecked(): tipo_escalado = "normalizar"
        elif self.rb_estandarizar.isChecked(): tipo_escalado = "estandarizar"

        if tipo_escalado == "normalizar":
            datos_train_esc, datos_test_esc = KNN.normalizar_datos(datos_train, [obj_externo], indices_seleccionados)
        elif tipo_escalado == "estandarizar":
            datos_train_esc, datos_test_esc = KNN.estandarizar_datos(datos_train, [obj_externo], indices_seleccionados)
        else:
            datos_train_esc, datos_test_esc = datos_train, [obj_externo]

        elem_externo_esc = datos_test_esc[0]
        clase_predicha, conteo = KNN.clasificar_dato(elem_externo_esc, datos_train_esc, k, indices_seleccionados, pesos_seleccionados)

        distancias = []
        for ent in datos_train_esc:
            d = KNN.calcular_distancia_euclidiana(elem_externo_esc, ent, indices_seleccionados, pesos_seleccionados)
            distancias.append(Distancia(d, ent.clase, ent.atributos))

        distancias.sort(key=lambda x: x.valor)

        sb = []
        sb.append("=== CLASIFICACIÓN DE ELEMENTO EXTERNO ===")
        sb.append(f"K = {k} | Escalado = {tipo_escalado.upper()}")
        sb.append(f"Fuente de entrenamiento: {ruta_train}\n")

        sb.append("--- Valores ingresados ---")
        for idx in indices_seleccionados:
            col_name = self.encabezados[idx].strip()
            val_ori = obj_externo.atributos[idx]
            val_esc = elem_externo_esc.atributos[idx]
            peso = pesos_seleccionados[indices_seleccionados.index(idx)]
            sb.append(f"  * {col_name:<15}: {val_ori:>10}  (Escalado: {val_esc}, Peso: {peso})")

        sb.append("\n--- Resultado ---")
        sb.append(f"CLASE PREDICHA : {clase_predicha}")
        
        votos_ganador = conteo[clase_predicha]
        total_votos = sum(conteo.values())
        sb.append(f"Confianza      : {(votos_ganador / total_votos) * 100:.2f}% ({votos_ganador}/{total_votos} votos)")
        sb.append(f"Votos por clase: {conteo}\n")

        sb.append(f"--- {k} Vecinos más cercanos ---")
        cabeceras_sel = [self.encabezados[i].strip() for i in indices_seleccionados]
        cabecera_str = "| " + " | ".join(cabeceras_sel) + " | Clase | Distancia |"
        sb.append(cabecera_str)

        for j in range(k):
            d = distancias[j]
            vals_vec = [d.atributos[idx].strip() for idx in indices_seleccionados]
            fila_vec = "| " + " | ".join(vals_vec) + f" | {d.clase} | {d.valor:.6f} |"
            sb.append(fila_vec)

        self.txt_salida_externo.setText("\n".join(sb))

    def _generar_texto_comparativa(self, config_actual: dict, metricas_actual: dict) -> str:
        """Tabla comparando configuracion y metricas promedio de la ejecucion anterior vs la actual."""
        if not self.ultimo_resultado:
            return "No existe una ejecución anterior guardada para comparar.\nEjecuta el modelo una segunda vez para ver la comparativa entre ejecuciones."

        ant_cfg = self.ultimo_resultado['config']
        ant_met = self.ultimo_resultado['metricas']

        sb = []
        sb.append("=== COMPARATIVA: EJECUCIÓN ANTERIOR vs ACTUAL ===\n")
        
        sb.append("--- Configuración ---")
        sb.append(f"{'Parámetro':<15} {'Anterior':<15} {'Actual':<15}")
        sb.append("-" * 45)
        sb.append(f"{'Valor K':<15} {ant_cfg['k']:<15} {config_actual['k']:<15}")
        sb.append(f"{'Split Train %':<15} {ant_cfg['split']:<15} {config_actual['split']:<15}")
        sb.append(f"{'Escalado':<15} {ant_cfg['escalado']:<15} {config_actual['escalado']:<15}")
        
        sb.append("\n--- Rendimiento Promedio ---")
        sb.append(f"{'Métrica':<15} {'Anterior':<12} {'Actual':<12} {'Diferencia (Delta)':<15}")
        sb.append("-" * 55)

        for m_key, m_nombre in [('accuracy', 'Accuracy'), ('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1-Score')]:
            val_ant = ant_met.get(m_key, 0.0)
            val_act = metricas_actual.get(m_key, 0.0)
            diff = val_act - val_ant
            signo = "+" if diff > 0 else ""
            
            sb.append(f"{m_nombre:<15} {val_ant:<12.4f} {val_act:<12.4f} {signo}{diff:<14.4f}")

        sb.append("\n--- Matriz de Confusión (Ejecución Anterior) ---")
        sb.extend(self._formatear_matriz_confusion(ant_met.get('clases', []), ant_met.get('matriz', {})))

        sb.append("\n--- Matriz de Confusión (Ejecución Actual) ---")
        sb.extend(self._formatear_matriz_confusion(metricas_actual.get('clases', []), metricas_actual.get('matriz', {})))

        return "\n".join(sb)

    def _formatear_matriz_confusion(self, clases: list, matriz: dict) -> list:
        """Lineas de texto de una matriz de confusion con letras a, b, c... por clase."""
        if not clases or not matriz:
            return ["  (No disponible)"]

        lineas = []
        letras = [chr(97 + i) for i in range(len(clases))]
        encabezado_letras = "  ".join(letras)
        lineas.append(f"   {encabezado_letras}  <-- classified as")

        for i, real in enumerate(clases):
            letra_real = letras[i]
            fila_counts = []
            for pred in clases:
                count = matriz.get(real, {}).get(pred, 0)
                fila_counts.append(f"{count:>2}")
            
            str_counts = " ".join(fila_counts)
            lineas.append(f"  {str_counts}  | {letra_real} = {real}")

        return lineas

    # def cambiar_tema(self, activo):
    #     app = QApplication.instance()
    #     if not app:
    #         return
        
    #     if activo:
    #         app.setStyleSheet(ESTILO_OSCURO)
    #     else:
    #         app.setStyleSheet(ESTILO_CLARO)


if __name__ == "__main__":
    app = QApplication([])
    panel = PanelKNN()
    panel.setWindowTitle("Sistema de Clasificacion KNN")
    panel.resize(900, 700)
    app.aboutToQuit.connect(panel.limpiar_temporales)
    panel.show()
    app.exec_()