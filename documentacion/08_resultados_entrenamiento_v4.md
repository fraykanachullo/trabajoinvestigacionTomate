# Resultados del entrenamiento v4 (1021 imágenes revisadas)

Fuente: `Resultados de los entrenamientos/v4-1021-revisadas` (zip del notebook `entrenamiento_colab.ipynb`).
Roboflow `plagas-tomate-majes-v2`, versión 1: 715 / 204 / 102 imágenes, todas de la sesión del 03/05/2026,
divididas por planta (sin fuga). YOLOv8n, YOLOv10n y YOLO11n, 3 semillas cada uno, `imgsz=640`,
hasta 150 épocas con `patience=30`. Valores: media ± desviación de las 3 semillas.

## 1. Comparación de modelos

| Modelo | Valid mAP50 | Valid mAP50-95 | Test mAP50 | Test mAP50-95 | Acierto clase principal (test) |
|---|---|---|---|---|---|
| **YOLO11n** | **30.4 ± 0.7** | **12.8 ± 0.5** | **24.2 ± 0.6** | 10.1 ± 0.8 | 58.8 ± 9.4 % |
| YOLOv8n | 28.7 ± 0.9 | 12.3 ± 0.1 | 23.8 ± 3.2 | 10.4 ± 1.9 | 60.8 ± 4.3 % |
| YOLOv10n | 27.7 ± 1.5 | 11.5 ± 0.5 | 21.5 ± 2.3 | 9.1 ± 1.0 | 55.9 ± 15.2 % |

- Elegido por validación: **YOLO11n, semilla 1** (valid mAP50 31.2 %; test mAP50 23.9 %, mAP50-95 9.7 %;
  acierto de clase principal 67.6 %, IC95 58–76 %, en 102 imágenes).
- YOLO11n es el mejor en validación y el más estable entre semillas; en test empata con YOLOv8n.
  YOLOv10n queda último y su salida además requeriría cambiar la app.
- La diferencia entre semillas del mismo modelo (hasta 6 puntos en test) es tan grande como entre
  modelos: con una sola corrida no se habría podido elegir.

## 2. Por clase (test, AP50)

| Clase | Cajas en test | YOLO11n | YOLOv8n | YOLOv10n | v1 (valid, con fuga) |
|---|---|---|---|---|---|
| sana | 85 | 46.0 | 44.6 | 38.7 | ~46 |
| mildiu | 198 | 24.9 | 20.0 | 17.8 | 17–23 |
| mosca_blanca | 18 | 24.1 | 20.2 | 23.7 | 9–14 |
| polilla | 11 | 14.8 | 25.4 | 19.3 | 63–66 |
| oidio | 87 | 11.4 | 8.6 | 7.7 | 7–8 |

- **Mosca blanca mejoró** (recall en test 22–30 % frente a 5–7 % en la v1), aunque con solo 18 cajas.
- **Polilla cayó de ~65 % a 15–25 %**: confirma que su buen resultado en la v1 venía de la fuga de datos
  (fotos de la misma planta en train y valid). Con 11 cajas en test el valor es muy inestable.
- **Oídio sigue siendo la clase más difícil** (8–11 %).
- Como en la v1, el modelo falla sobre todo por **no detectar** (matriz de confusión de YOLO11n s1 en test):
  queda como fondo el 84 % de oídio, 82 % de polilla, 78 % de mosca blanca y 61 % de mildiu; la confusión
  entre clases es ≤ 6 %. Además, el 62 % de sus falsas detecciones en el fondo son "mildiu".

## 3. Por franja horaria (test, mAP50)

| Modelo | Mañana (21 img.) | Mediodía (35 img.) | Tarde (46 img.) |
|---|---|---|---|
| YOLO11n | 25.3 ± 2.3 | 23.0 ± 1.7 | 33.6 ± 0.5 |
| YOLOv8n | 24.6 ± 1.5 | 19.3 ± 1.5 | 31.5 ± 6.8 |
| YOLOv10n | 21.5 ± 4.1 | 18.5 ± 2.5 | 31.1 ± 1.7 |

El mediodía sale más bajo en los tres modelos, pero **todavía no se puede atribuir a la radiación**:

- Las franjas tienen clases distintas: polilla solo aparece en la tarde (11 cajas) y mosca blanca no
  aparece al mediodía. El mediodía tiene sobre todo mildiu y oídio, que son las clases difíciles.
- Dentro de una clase las cifras dependen de muy pocas plantas. Oídio: mañana 2.4 % (15 cajas, 4 plantas),
  mediodía 9.7 % (65 cajas, 4 plantas, una sola con 35), tarde 33.8 % (7 cajas, 3 plantas).
- La señal más clara es **sana**: 31.7 % al mediodía frente a 53.9 % (mañana) y 50.0 % (tarde), con
  28 / 12 / 45 cajas. Es coherente con la hipótesis del brillo sobre la hoja, pero viene de 4 plantas.
- El acierto de la clase principal **no** baja al mediodía (59–74 % según el modelo).
- Todo el test es de un solo día. Para concluir sobre la radiación hacen falta las sesiones de julio y
  mediodía en más de una fecha.

**Nota (05/11/2026):** una revisión posterior encontró que 5 de las 102 imágenes de test eran
cuadros de un video de la misma planta que fotos de train o valid (bitácora §6b). Es una fuga
pequeña; para comparar con la v5 se evalúan ambas sobre el test sin esas imágenes.

## 4. Entrenamiento

- Cada corrida tardó entre 11 y 46 min en la T4; la parada temprana cortó entre las épocas 70 y 150 y
  el mejor modelo apareció entre las épocas 40 y 131.
- Como en la v1 hay **sobreajuste**: la pérdida de clasificación en validación sube al final
  (p. ej. YOLOv8n 2.29 → 2.51) mientras la de entrenamiento sigue bajando. Faltan datos, no épocas.
- La v1 (30 % de mAP50 en valid) no es comparable directamente: tenía fuga de datos, imágenes estiradas
  y etiquetas sin revisar. El test sin fuga (24 %) es la medida honesta de este avance.

## 5. Despliegue (05/11/2026)

- YOLO11n s1 instalado en el backend (`app/models/best.pt`) y en la app (`best_float16.tflite`,
  5.3 MB, antes 6.1 MB); `metadata.json` 2.0.0 en ambos. El modelo v1 quedó en
  `app/modelos_anteriores/v1_yolov8n_exp3/`.
- **La conversión es fiel:** con la misma imagen de entrada, ONNX y TFLite difieren como máximo en
  0.004 de score y coinciden en la detección principal en 96/96 imágenes.
- **Pero el modelo es sensible a cómo se reduce la foto.** Si la foto se redimensiona distinto
  (PIL con antialias en la PC, frente al reescalado del servidor), la clase principal coincide en el
  82.8 % de las 192 imágenes (el modelo v1 daba 91.7 % con la misma prueba). Muchas de sus detecciones
  están cerca del umbral de 0.30.
- **Pruebas en el Samsung A56 (192 imágenes de test, app en modo profile):**

  | | v1 YOLOv8n | v4 YOLO11n |
  |---|---|---|
  | Latencia por foto en el celular | 597 ± 195 ms | **501 ± 120 ms** (mediana 542) |
  | Misma clase principal que el servidor | 94.3 % | **83.9 %** |
  | Cajas del servidor reproducidas (IoU ≥ 0.5) | 78.3 % | **80.4 %** (634/789) |
  | Cajas del celular que coinciden con el servidor | 85.5 % | 84.1 % (634/754) |
  | Servidor en la CPU de la laptop | 526 ± 547 ms | 205 ± 38 ms |

  El modelo nuevo es más rápido y reproduce igual o mejor las cajas, pero la clase principal coincide
  menos: el nuevo da confianzas más bajas y la detección de mayor confianza cambia con pequeñas
  diferencias de la imagen. Resultados en `redaccionarticulo/sensorpeaper/resultados_movil_A56_v4.json`
  y `comparacion_servidor_movil_v4.json`.

## 6. Siguientes pasos

1. ~~Repetir las pruebas P1–P2 en el celular con el modelo nuevo.~~ Hecho.
2. ~~Corregir la fuga video-fotos (66 imágenes).~~ Hecho: tags en Roboflow y el notebook las mueve.
3. ~~Etiquetar las ~945 imágenes restantes~~ Hecho (12/11/2026, versión 2 de Roboflow), con prioridad en **oídio, mosca blanca y polilla**, y en las
   sesiones de julio (incluye mediodía de otra fecha).
4. Revisar en las imágenes con falsas detecciones de "mildiu" si son lesiones sin caja (etiqueta faltante).
5. Probar `imgsz=1024` para los objetos pequeños (mosca blanca, oídio), midiendo también la latencia en
   el celular.
