# Detección de plagas y enfermedades del tomate en el Valle de Majes

Material de investigación del proyecto (Universidad Peruana Unión, Escuela de Ingeniería de Sistemas):
curación del dataset, entrenamiento y comparación de YOLOv8n / YOLO11n / YOLOv10n, conversión a
TensorFlow Lite y resultados. Autores: Fray Benjamin Kana Chullo, Hiram Jeremy Ccoto Cruz; asesor:
Angel Rosendo Condori Coaquira.

Clases: la v6 (versión final) usa 4 clases `mildiu`, `mosca_blanca`, `oidio`, `polilla`. La v4 y la v5 tenían
además `sana` (la app y el backend actuales todavía usan el modelo v5 de 5 clases).

**Foco del trabajo:** cómo la alta radiación solar (sobreexposición) degrada al detector y cómo mitigarlo.

## Repositorios y datos

| Qué | Dónde |
|---|---|
| Backend (FastAPI + YOLO) | https://github.com/fraykanachullo/backentTomate |
| App móvil AgriScan Pro (Flutter + TFLite, funciona sin internet) | https://github.com/fraykanachullo/frontendTomate |
| Dataset (Roboflow Universe, CC BY 4.0) | https://universe.roboflow.com/fray-hxtls/plagas-tomate-majes-v2 |
| Versión usada en la v5 (1966 imágenes; 1370 / 404 / 192; Fit within 1024; sin aumentos) | https://universe.roboflow.com/fray-hxtls/plagas-tomate-majes-v2/dataset/2 |
| Versión usada en la v6 (1966 imágenes, etiquetas corregidas, 4 clases) | https://universe.roboflow.com/fray-hxtls/plagas-tomate-majes-v2/dataset/3 |
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

## Cómo reproducir la v6 (prueba de estrés y mitigación)

1. Descargar la versión 3 del dataset en formato **YOLOv8** (zip).
2. `python entrenamiento/entrenar_yolo_v6.py --solo-verificar --zip <zip>`: quita `sana` si viene, aplica los
   66 movimientos (1405 / 381 / 180) y verifica que no haya fuga.
3. `python entrenamiento/entrenar_yolo_v6.py --zip <zip>`: entrena 3 experimentos (base, fotométrico, CLAHE)
   × 3 modelos × 3 semillas y evalúa en prueba, prueba fija, imágenes nuevas y la prueba de estrés
   (−1 a +2 pasos de exposición). Unas 6 h en una RTX 5070 Ti.
4. Comparar con `resultados/v6-4clases/` y analizar con `analisis/a9_analisis_v6.py`.

El experimento de control de fuga (`entrenamiento/entrenar_control.py`) está listo pero aún no se ejecutó.

## Contenido

| Carpeta | Contenido |
|---|---|
| `entrenamiento/` | `entrenar_yolo.py` (v5), `entrenar_yolo_v6.py` (v6: 4 clases, fotométrico, CLAHE, prueba de estrés), `entrenar_control.py` (fuga y sesiones, pendiente), versiones para supercomputadora y Colab. |
| `dataset/` | División por grupo de captura (`agrupar_split.py`), corrección de fuga video-fotos (`movimientos_split_v5.csv`), test fijo (`test_fijo_v4.txt`) y extracción de cuadros. |
| `analisis/` | Scripts de la revisión (a1–a10): IC por bootstrap, métricas por imagen y falso sano, exposición, EXIF, duplicados pHash, auditoría de etiquetas, prueba de estrés y análisis v6, con sus salidas (CSV y figuras) y las listas de cada subconjunto de prueba. Incluye la validación del aviso de luz de la app (a11). Usan rutas locales que hay que ajustar. |
| `conversion/` | Verificación ONNX vs TFLite y comparación servidor vs celular. |
| `documentacion/` | Bitácora, reglas de etiquetado, guías y resultados de cada etapa (01–15), corrección de etiquetas y plan de radiación. |
| `resultados/` | Métricas y mejor modelo de la v4, la v5 y la v6 (en la v6, el mejor de cada experimento y el `results.csv` de las 27 corridas). |
| `articulo/` | Avance corregido en formato MDPI (LaTeX + PDF), respuesta a la revisión y matriz de respuesta. |

## Resultados principales

mAP@0.5 de cuatro clases (sin hoja sana), media ± DE de 3 semillas:

| | v4 (1021 img.) | v5 (1966 img.) | v6 base (etiquetas corregidas) |
|---|---|---|---|
| Prueba fija (97 imágenes de mayo, mismas etiquetas) | 19.2 | 17.6–21.3 | 20.2 ± 2.8 |
| Imágenes nuevas (83, julio) | 25.8 | 32.1–44.4 | 37.6 ± 7.0 |
| Prueba completa (180), YOLOv8n / YOLO11n / YOLOv10n | 21.9 (YOLO11n) | 25.8 / 24.0 / 29.8 | 29.8 / 26.4 / 23.4 |

Las comparaciones v4–v5 usan las mismas imágenes y las etiquetas de la v5. La v6 cambió también las etiquetas de prueba, así que su comparación con la v5 es aproximada.

**Sobreexposición (v6, misma foto con más luz):** con +1 paso, el mAP@0.5 cae 32 % (26.3 → 17.9) y la mosca blanca pierde dos tercios de su AP. La subexposición no afecta.
- El aumento fotométrico retiene el 42 % del mAP a +2 pasos, frente a 29 % de la base, y mejora en las 9 de 9 corridas. A cambio, pierde 2.4 puntos con luz normal.
- CLAHE no ayuda.
- Ninguna estrategia rescata la mosca blanca por encima de +1.5 pasos.

Detalle en `documentacion/15_resultados_v6.md`. En el celular (Samsung Galaxy A56, TFLite float16), el modelo procesa una foto en unos 0.6 s sin internet (modelo v6; detalle en `analisis/v6/celular/`).

## Entrenar

Con el zip del dataset (formato YOLOv8) en la carpeta de trabajo:

```bash
pip install ultralytics==8.4.82 pandas
python entrenamiento/entrenar_yolo.py --solo-verificar
python entrenamiento/entrenar_yolo.py --tamanos n
```

Guía completa: `documentacion/09_entrenamiento_supercomputadora.md`.
