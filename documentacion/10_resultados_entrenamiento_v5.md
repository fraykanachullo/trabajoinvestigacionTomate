# Resultados del entrenamiento v5 (1966 imágenes, etiquetado completo)

Fuente: `Resultados de los entrenamientos/v5-1966-completas` (zip de la PC con RTX 5070 Ti,
`entrenar_yolo.py` sin cambios). Roboflow versión 2. Después de mover las 66 imágenes con fuga
video-fotos: **1405 train / 381 valid / 180 test**, 288 grupos de planta, sin fuga.
YOLOv8n, YOLO11n y YOLOv10n × semillas 0, 1 y 2; 640 px, batch 32, 200 épocas con `patience=50`,
`cos_lr`, `close_mosaic=15`, AMP. Valores: media ± desviación de las 3 semillas. Ninguna corrida
falló; cada una tardó 6–12 min en la GPU.

**Ojo al comparar con la v4:** el test general cambió (de 102 a 180 imágenes, ahora con las
sesiones de julio) y también la validación. La comparación justa es el **test fijo**: las mismas 97
imágenes de test de la v4, con etiquetas casi iguales (mildiu tiene 29 cajas más, las demás clases
±2). También cambiaron algunos hiperparámetros (batch 16 → 32, `cos_lr`, `close_mosaic`,
200/50 épocas), así que la diferencia no se debe solo a los datos.

## 1. Resultado general

| Modelo | Valid mAP50 v4 → v5 | **Test fijo mAP50** v4 → v5 | Test fijo mAP50-95 v4 → v5 | Acierto clase principal (test fijo) v4 → v5 |
|---|---|---|---|---|
| YOLOv8n | 28.7 → 38.5 | 23.8 → **25.1** | 10.4 → 12.3 | 60.8 → 58.1 % |
| YOLO11n | 30.4 → 34.7 | 24.2 → **24.6** | 10.1 → 12.3 | 58.8 → 60.8 % |
| YOLOv10n | 27.7 → 35.5 | 21.5 → **28.3** | 9.1 → 13.3 | 55.9 → 63.2 % |

- **En las mismas imágenes, la mejora es chica.** YOLOv8n y YOLO11n suben apenas 0.4–1.3 puntos
  de mAP50, dentro de la variación entre semillas. Solo YOLOv10n mejora con claridad (+6.8).
- **Las cajas son más precisas:** el mAP50-95 sube 2–4 puntos en los tres modelos.
- **El acierto de la clase principal no cambia** (58–63 %, con intervalos de ±10 puntos).
- La validación sube 4–10 puntos, pero es otro conjunto de imágenes (incluye julio).
- Test completo v5 (180 imágenes): YOLOv10n 36.3 ± 1.1, YOLOv8n 33.6 ± 2.4, YOLO11n 32.1 ± 4.0.

## 2. Por clase (AP50 en el test fijo, media de las 3 semillas)

| Clase | Cajas en train v4 → v5 | YOLOv8n v4 → v5 | YOLOv10n v4 → v5 | Recall (media de los 3) v4 → v5 |
|---|---|---|---|---|
| sana | 879 → 2344 (+167 %) | 44.6 → 55.0 | 38.7 → 56.0 | 58 → 63 % |
| mildiu | 977 → 1882 (+93 %) | 20.0 → 25.4 | 17.8 → 22.6 | 28 → 32 % |
| polilla | 96 → 374 (+290 %) | 25.4 → 25.4 | 19.3 → 33.5 | 25 → 33 % |
| mosca blanca | 205 → 253 (+23 %) | 20.2 → 13.4 | 23.7 → 24.8 | 25 → 15 % |
| **oídio** | 550 → 594 (**+8 %**) | 8.6 → **6.0** | 7.7 → **4.4** | 12 → **8 %** |

En el test completo (180 imágenes, 29 cajas de polilla en vez de 9), polilla llega a 39–62 de AP50.

## 3. Por qué no mejoró más

1. **Las clases mejoran en proporción a los datos nuevos que recibieron.** Sana, mildiu y polilla
   ganaron entre 93 y 290 % de cajas y mejoraron. Oídio (+8 %) y mosca blanca (+23 %) casi no
   recibieron datos nuevos y **empeoraron**: su proporción en el dataset bajó (oídio pasó de ~20 %
   a ~11 % de las cajas) y el modelo se inclina hacia las clases mayoritarias.
2. **Oídio y mosca blanca son objetos chicos o difusos.** A 640 px una colonia de mosca blanca o
   el polvo blanco del oídio quedan en pocos píxeles. Además, el oídio cuesta delimitar con una
   caja, así que sus etiquetas son menos consistentes.
3. **Sigue el sobreajuste.** El mejor modelo aparece entre las épocas 24 y 79; después la pérdida
   de clasificación en validación sube (p. ej. YOLO11n 1.94 → 2.2) mientras la de entrenamiento
   sigue bajando. Faltan datos variados: son solo 288 plantas de 3 salidas al campo.
4. **El test ahora es más difícil y más representativo.** Incluye 2 sesiones nuevas (julio). La
   franja mañana bajó (mAP50 20–23 frente a 25 en la v4) porque ahora incluye la sesión del 12/07
   a las 8 h; la tarde subió (37–42).

## 4. Franjas horarias (test completo, mAP50)

| Modelo | Mañana (49 img.) | Mediodía (35) | Tarde (96) |
|---|---|---|---|
| YOLOv8n | 19.9 ± 1.8 | 24.8 ± 3.3 | 37.0 ± 3.7 |
| YOLO11n | 20.5 ± 1.6 | 25.7 ± 3.4 | 37.3 ± 6.4 |
| YOLOv10n | 22.6 ± 0.3 | 22.6 ± 4.0 | 42.3 ± 1.1 |

El acierto de la clase principal muestra lo contrario: mañana 80 %, mediodía 53–65 % y tarde
47–52 %. Las franjas siguen confundidas con la sesión y la mezcla de clases (la tarde de julio
tiene mucha polilla), así que **todavía no permiten concluir sobre la radiación**.

## 5. Modelo elegido y comparación con el instalado

- **Elegido por validación: YOLOv8n, semilla 2** (valid mAP50-95 19.6 %), como manda el
  protocolo. En test, YOLOv10n sale algo mejor, pero no se elige con test y además su salida
  exigiría cambiar la app.
- **Evaluación propia en la PC, sobre el test nuevo de 180 imágenes y con las mismas reglas:**

  | | mAP50 | mAP50-95 | Acierto clase principal | AP50 oídio | AP50 sana | AP50 mosca blanca |
  |---|---|---|---|---|---|---|
  | v4 instalado (YOLO11n s1) | 27.8 | 12.0 | 66.1 % (119/180) | 11.4 | 52.1 | 16.9 |
  | v5 elegido (YOLOv8n s2) | **30.9** | **14.6** | 63.9 % (115/180) | **5.0** | **68.6** | **22.0** |

  El v5 detecta mejor en general (+3 de mAP50), pero el acierto del diagnóstico queda igual
  (la diferencia de 2 puntos está dentro del intervalo de ±7) y **oídio empeora a la mitad**.

**Instalado en el backend (13/11/2026):** `best.pt` = YOLOv8n s2, `metadata.json` 3.0.0. El modelo
v4 quedó respaldado en `app/modelos_anteriores/v4_yolo11n_s1/` (best.pt, metadata y TFLite). **En la app (13/11/2026):** convertido a `best_float16.tflite` (6.1 MB, entrada [1,640,640,3], salida
[1,9,8400]). Con la misma entrada, ONNX y TFLite coinciden en la detección principal en 96/96
imágenes (diferencia máxima de score 0.014). APK release instalado en el Samsung A56 encima del
anterior (se conservan sesión e historial). Pendiente: repetir las pruebas P1–P2 de latencia y
concordancia con el servidor.

Documento con las tres etapas: `redaccionarticulo/sensorpeaper/avance_final/Avance_final_MDPI_KanaChullo.pdf`.

**Corrección de la app (13/11/2026):** el login fallaba con "Error login" sin llegar al servidor.
Causa: Android restauraba por copia de seguridad automática los datos cifrados de
`flutter_secure_storage`; al no poder descifrarlos, la lectura del token lanzaba una excepción en el
interceptor de Dio y la petición se cortaba antes de salir. Arreglo: `auth_local_datasource.dart`
borra los datos ilegibles y sigue sin sesión; `allowBackup="false"` en el manifiesto; el mensaje de
error ahora muestra el tipo de fallo. Login verificado en el Samsung A56.

## 6. Siguientes pasos para mejorar

1. **Más oídio y mosca blanca etiquetados** (meta ≥ 1000 cajas de oídio en train) y revisar la
   consistencia de las cajas de oídio. Es la mejora con mayor efecto esperado según el punto 3.1.
2. **Probar `imgsz=1024`** (la versión de Roboflow ya está a 1024): ayuda a objetos chicos. En la
   RTX 5070 Ti tarda poco; luego medir la latencia en el celular.
3. **No seguir sumando cajas de `sana`**: ya son 2344, el 43 % de las cajas de train.
4. **Una salida al campo más** (plantas nuevas, mediodía en otra fecha) para el experimento de radiación.
5. Opcional: tamaño `s` con 1 semilla, para ver si la capacidad del modelo limita.

## 7. Tamaño de las cajas (análisis de las etiquetas v5, 13/11/2026)

Área de cada caja en % de la imagen (todas las cajas del dataset). A 640 px de lado mayor la
imagen tiene ~307 000 píxeles, así que 0.03 % ≈ una caja de 10 × 10 px.

| Clase | Cajas | Mediana | p10 | p90 | Cajas < 1 % | Lado típico a 640 px |
|---|---|---|---|---|---|---|
| mosca blanca | 349 | 0.03 % | 0.01 % | 0.22 % | 99 % | **~10 px** |
| oídio | 889 | 0.40 % | 0.09 % | 1.84 % | 79 % | ~35 px (p10 ~17 px) |
| polilla | 474 | 0.84 % | 0.20 % | 2.22 % | 55 % | ~50 px |
| mildiu | 2698 | 1.25 % | 0.24 % | 5.29 % | 43 % | ~60 px |
| sana | 3356 | 2.56 % | 0.75 % | 8.45 % | 16 % | ~90 px |

- **Mosca blanca queda por debajo del mínimo de 16 px** de la regla de etiquetado, y buena parte
  del oídio también: a 640 px el modelo casi no ve esos objetos. Es la explicación más directa de
  su bajo AP y respalda el experimento `imgsz=1024` (mosca blanca pasaría a ~16 px).
- **Mildiu no muestra mezcla de criterios hoja entera / mancha:** el 90 % de sus cajas mide menos
  del 5 % de la imagen (menos que una hoja sana típica) y solo 8 de 1052 imágenes (1 %) tienen a
  la vez cajas muy chicas y muy grandes. Sus cajas son de lesión, como dice la regla. Su bajo AP
  apunta más a **lesiones sin caja** (en la v4, el 62 % de las falsas detecciones en el fondo
  eran "mildiu") y a confusión con oídio, que hay que revisar a mano.
