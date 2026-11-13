# Detección de plagas y enfermedades del tomate en el Valle de Majes

Material de investigación del proyecto (Universidad Peruana Unión, Escuela de Ingeniería de Sistemas):
curación del dataset, entrenamiento y comparación de YOLOv8n / YOLO11n / YOLOv10n, conversión a
TensorFlow Lite y resultados. Autores: Fray Benjamin Kana Chullo, Hiram Jeremy Ccoto Cruz; asesor:
Angel Rosendo Condori Coaquira.

Clases (este orden lo usan la app y el backend): `mildiu`, `mosca_blanca`, `oidio`, `polilla`, `sana`.

## Repositorios y datos

| Qué | Dónde |
|---|---|
| Backend (FastAPI + YOLO) | https://github.com/fraykanachullo/backentTomate |
| App móvil AgriScan Pro (Flutter + TFLite, funciona sin internet) | https://github.com/fraykanachullo/frontendTomate |
| Dataset (Roboflow Universe, CC BY 4.0) | https://universe.roboflow.com/fray-hxtls/plagas-tomate-majes-v2 |
| Versión usada en la v5 (1966 imágenes; 1370 / 404 / 192; Fit within 1024; sin aumentos) | https://universe.roboflow.com/fray-hxtls/plagas-tomate-majes-v2/dataset/2 |
| Versión usada en la v4 (1021 imágenes; 715 / 204 / 102) | https://universe.roboflow.com/fray-hxtls/plagas-tomate-majes-v2/dataset/1 |

## Cómo reproducir los resultados de la v5

1. Descargar la versión 2 del dataset en formato **YOLOv8** (zip) desde el link de arriba.
2. Instalar `ultralytics==8.4.82` y `pandas` (se usó PyTorch 2.12 con CUDA 13.2 en una RTX 5070 Ti).
3. Poner el zip junto a `entrenamiento/entrenar_yolo.py` y ejecutar
   `python entrenar_yolo.py --tamanos n`. El script descomprime, verifica que no haya fuga de datos,
   mueve las 66 imágenes de `dataset/movimientos_split_v5.csv` (quedan 1405 / 381 / 180), entrena
   YOLOv8n, YOLO11n y YOLOv10n con las semillas 0, 1 y 2 (200 épocas, paciencia 50, batch 32,
   640 px) y evalúa en prueba, por franja horaria y en el test fijo (`dataset/test_fijo_v4.txt`).
4. Comparar con `resultados/v5-1966-completas/`. Como se entrena con `deterministic=False`, los
   valores pueden variar un poco entre ejecuciones; por eso se reportan 3 semillas.

## Contenido

| Carpeta | Contenido |
|---|---|
| `entrenamiento/` | `entrenar_yolo.py` (PC con GPU NVIDIA), `entrenar_supercomputadora.py` y `entrenamiento_colab.ipynb` (Colab). Verifican el dataset, entrenan los 3 modelos × 3 semillas y evalúan. |
| `dataset/` | División por planta (`agrupar_split.py`), corrección de fuga video-fotos (`revisar_fuga_videos.py`, `movimientos_split_v5.csv`), test fijo de comparación (`test_fijo_v4.txt`) y extracción de cuadros de video. |
| `conversion/` | Verificación ONNX vs TFLite y comparación servidor vs celular. |
| `documentacion/` | Bitácora del dataset, reglas de etiquetado, guías de entrenamiento y resultados de cada etapa (01–10). |
| `resultados/` | Métricas (CSV) y mejor modelo de la v4 (1021 imágenes) y la v5 (1966 imágenes). |
| `articulo/` | Avance en formato MDPI (LaTeX + PDF). |

## Resultados principales

| | v4 (1021 imágenes) | v5 (1966 imágenes) |
|---|---|---|
| mAP@0.5 en el test fijo (97 imágenes), YOLOv8n / YOLO11n / YOLOv10n | 23.8 / 24.2 / 21.5 % | 25.1 / 24.6 / 28.3 % |
| mAP@0.5:0.95 en el test fijo | 10.4 / 10.1 / 9.1 % | 12.3 / 12.3 / 13.3 % |
| Modelo elegido por validación | YOLO11n, semilla 1 | YOLOv8n, semilla 2 |

Los tres modelos quedan empatados dentro de la variación entre semillas. Las clases mejoran en
proporción a los datos nuevos que reciben; oídio y mosca blanca (objetos de ~35 y ~10 px a 640 px)
son las más difíciles. Detalle en `documentacion/10_resultados_entrenamiento_v5.md`.

En el celular (Samsung Galaxy A56, TFLite float16) el modelo procesa una foto en unos 0.5 s sin internet.

## Entrenar

Con el zip del dataset (formato YOLOv8) en la carpeta de trabajo:

```bash
pip install ultralytics==8.4.82 pandas
python entrenamiento/entrenar_yolo.py --solo-verificar
python entrenamiento/entrenar_yolo.py --tamanos n
```

Guía completa: `documentacion/09_entrenamiento_supercomputadora.md`.
