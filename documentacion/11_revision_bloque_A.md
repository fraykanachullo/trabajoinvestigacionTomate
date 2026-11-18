# Revisión del artículo: bloque A (análisis sin entrenar), 13/11/2026

Responde a las observaciones 1, 3, 5, 6 y 7 del reporte de revisión
(`reporte_revision_g6_dpcaikafth.docx`) usando los modelos y datos que ya existían.
Scripts y salidas: `D:\ProyectoInvestigacion\analisis_revision\` (`a1`–`a4`, carpetas `metricas/`,
`exposicion/`, `duplicados/`). Todas las evaluaciones usan **las etiquetas de la v5** como verdad única.

Modelos: los 9 de la v5 (YOLOv8n, YOLO11n, YOLOv10n × semillas 0–2) y **solo uno de la v4**
(YOLO11n semilla 1, el elegido; los pesos de las otras 8 corridas v4 están en Google Drive y no en la PC).

Subconjuntos de prueba:
- **fijo97**: las 97 imágenes del test de la v4 que siguen en el test de la v5 (todas del 3 de mayo).
- **nuevas83**: las otras 83 imágenes del test de la v5 (29 del 12/07, 51 del 29/07, 3 del 03/05).
- **test180**: las dos juntas.

## 1. Error confirmado en la Tabla 4 del artículo (obs. 3-i)

En el artículo, los valores v4 de "mAP@0.5 test fijo" (23.8 / 24.2 / 21.5) son los de la prueba de la
v4: **102 imágenes con etiquetas de la v4**, no las 97 con etiquetas de la v5. Re-evaluado
correctamente (Ultralytics `val`, mismas 97 imágenes y mismas etiquetas):

| Modelo | fijo97 mAP@0.5 | fijo97 mAP@0.5:0.95 | nuevas83 mAP@0.5 | test180 mAP@0.5 |
|---|---|---|---|---|
| v4 YOLO11n s1 | **26.6** | 11.4 | **32.1** | 27.9 |
| v5 YOLO11n (3 semillas) | 24.6 ± 0.8 | 12.3 ± 0.8 | 41.7 ± 8.8 | 32.2 ± 3.9 |
| v5 YOLOv8n (3 semillas) | 25.1 ± 0.6 | 12.3 ± 0.3 | 46.8 ± 5.3 | 33.6 ± 2.5 |
| v5 YOLOv10n (3 semillas) | 28.3 ± 0.9 | 13.3 ± 1.0 | 49.6 ± 5.3 | 36.3 ± 1.1 |

**Conclusión nueva (reemplaza la del artículo):**
- En las imágenes de mayo (fijo97) **la v5 no mejora a la v4**. Diferencias pareadas por bootstrap
  agrupado (1000 réplicas): todas con IC 95 % que incluye 0 (p. ej. v5 YOLOv8n s2 − v4: −2.5 puntos
  de mAP@0.5, IC −7.5 a +2.3).
- En las sesiones de julio (nuevas83), que la v4 nunca vio, **la v5 mejora de 32 a 42–50**.
- Por lo tanto, lo que aportó la v5 fue **cobertura de sesiones nuevas (generalización temporal)**, no
  más imágenes de las mismas condiciones. La frase "el mAP@0.5:0.95 sube 2–4 puntos en el test fijo"
  del artículo **es incorrecta** y debe retirarse.

Por clase en fijo97 (AP@0.5, media de semillas): mosca blanca v4 29.0 → v5 13–25; oídio 12.6 → 4–9;
polilla 6.1 → 22–34; sana y mildiu sin cambio claro. Con 9 cajas de polilla y 18 de mosca blanca en
fijo97 estas cifras por clase son muy inestables.

## 2. Incertidumbre (obs. 6)

IC 95 % por bootstrap agrupado por grupo de captura (implementación propia de mAP de 101 puntos; da
valores ~3 puntos menores que Ultralytics, por eso se usa solo para IC y diferencias pareadas):
el IC de mAP@0.5 de un modelo en fijo97 mide unos **±6–8 puntos** (p. ej. v4: 23.9 [19.0–31.9]).
Con este tamaño de prueba, diferencias de menos de ~8 puntos entre modelos no son concluyentes.

Cajas por clase en prueba: fijo97 — mildiu 227, oídio 89, sana 72, mosca blanca 18, polilla 9;
test180 — mildiu 273, sana 266, oídio 89, mosca blanca 34, polilla 29.

## 3. Métricas por imagen y falso sano (obs. 7)

Reglas de la app (conf 0.30, IoU 0.45, NMS agnóstica). "Falso sano" = imagen con alguna plaga o
enfermedad etiquetada donde el modelo no detecta ninguna.

| Modelo | Acierto clase principal fijo97 / test180 | Falso sano test180 | Sens. mildiu | Sens. mosca | Sens. oídio | Sens. polilla |
|---|---|---|---|---|---|---|
| v4 YOLO11n s1 | 70 % / 66 % | **22 %** | 71 % | 37 % | 59 % | 55 % |
| v5 YOLOv8n s2 (instalado) | 64 % / 64 % | **28 %** | 64 % | 37 % | 41 % | 52 % |
| v5 rango 9 modelos | 46–72 % / 51–67 % | 27–52 % | 40–73 % | 13–43 % | 6–59 % | 21–59 % |

**Líneas base del acierto de la clase principal:**
- fijo97: "siempre mildiu" acierta el **60.8 %**, igual que la mayoría de los modelos. En imágenes de
  mayo, el acierto de clase principal no supera a la línea base trivial.
- test180: "siempre mildiu" 41.1 %, "siempre sana" 26.7 %; los modelos (51–67 %) sí la superan.

**McNemar (acierto, v4 vs v5 en fijo97):** la v4 acierta significativamente más que 3 de los 9
modelos v5 (YOLO11n s1, YOLOv10n s2, YOLOv8n s0; p < 0.01); con el instalado (YOLOv8n s2) no hay
diferencia significativa (14 vs 8 discordantes, p = 0.29).

**Implicación:** el error más costoso para el agricultor (falso sano) ocurre en 1 de cada 4–5
imágenes con plaga incluso con el mejor modelo. La métrica "acierto de clase principal" usada hasta
ahora favorece a mildiu y debe reemplazarse por métricas por imagen y por plaga.

## 4. Exposición solar medida (obs. 1, ruta a)

Por imagen: brillo medio, % de píxeles saturados (algún canal ≥ 250), % especular (V ≥ 230 y S ≤ 40),
contraste.

- **El mediodía sí produce más sobreexposición:** mediana de píxeles saturados 3.6 % al mediodía frente
  a 0.8 % (mañana) y 1.0 % (tarde); contraste 58.8 frente a 48.7 y 51.0. Es la primera medición
  objetiva del efecto de la radiación sobre las fotos.
- **Efecto sobre el modelo, no concluyente:** el acierto baja de 67 % (tercil bajo de saturación) a
  57 % y 55 % (medio y alto), pero los IC se superponen. Regresión logística del acierto con
  sesión y modelo como controles y errores agrupados por grupo: OR = 0.82 por desviación estándar de
  saturación (IC 95 % 0.65–1.04, p = 0.11). Brillo y especular: sin efecto (OR 1.00 y 0.98).
- **Confusión:** el tercil alto de saturación es casi todo de la sesión del 3 de mayo (55 de 60), y no
  hay mediodía en julio. Con 180 imágenes de prueba no se puede separar exposición de sesión.

Para el artículo: o se presenta como resultado exploratorio ("la sobreexposición del mediodía es
medible; su efecto en el detector apunta a una caída que no alcanza significancia"), o se diseña la
próxima salida al campo para cubrir mediodía en otra fecha.

## 5. Auditoría de casi-duplicados (obs. 5)

pHash de 63 bits sobre las 1966 imágenes, todos los pares entre subconjuntos (856 785):
- Distancia mínima entre subconjuntos: **12**; pares con distancia ≤ 10: **0**. El único par a 12
  (valid–test) es de escenas distintas (revisado a ojo).
- Referencia: fotos de la misma ráfaga están a una mediana de 30, así que el pHash detecta fotos casi
  idénticas pero **no** la misma planta fotografiada desde otro ángulo.
- Conclusión honesta: no hay imágenes duplicadas o casi idénticas entre subconjuntos; la fuga residual
  por la misma planta en dos ráfagas no se puede descartar con este método. Para eso hace falta una
  auditoría por embeddings (DINOv2/CLIP) o la prueba temporal (que en parte ya da `nuevas83`).
- Observación del reporte sobre personas: en el par revisado aparece una mano (sin rostro); conviene
  revisar el dataset completo antes de citarlo como público.

## 5b. mAP sin `sana` (apreciación del revisor, sección 8B) — opción C, 16/11/2026

El revisor pide reportar también el mAP de las 4 plagas/enfermedades, porque `sana` (43 % de las
cajas) no es un objeto y sube el promedio. Ultralytics `val`, etiquetas v5, media ± DE de 3 semillas
(`analisis_revision/a6_map_sin_sana.py`, `metricas/map_4_vs_5.csv`):

| Modelo | test180 mAP@0.5 (5 clases) | **test180 mAP@0.5 (4 plagas)** | fijo97 (5) | **fijo97 (4)** | nuevas83 (5) | **nuevas83 (4)** |
|---|---|---|---|---|---|---|
| v4 YOLO11n s1 | 27.9 | **21.9** | 26.6 | **19.2** | 32.1 | **25.8** |
| v5 YOLO11n | 32.2 ± 3.9 | **24.0 ± 4.9** | 24.6 ± 0.8 | **18.4 ± 1.0** | 41.7 ± 8.7 | **32.1 ± 11.7** |
| v5 YOLOv8n | 33.6 ± 2.5 | **25.8 ± 3.4** | 25.1 ± 0.6 | **17.6 ± 2.9** | 46.8 ± 5.4 | **39.5 ± 7.4** |
| v5 YOLOv10n | 36.3 ± 1.1 | **29.8 ± 1.3** | 28.3 ± 0.9 | **21.3 ± 0.9** | 49.6 ± 5.3 | **44.4 ± 6.4** |

mAP@0.5:0.95 de las 4 plagas en test180: v4 8.0; v5 9.0–11.0. AP@0.5 de `sana` sola: 52–65.

- **Sin `sana` el mAP baja 6–9 puntos**: `sana` es la clase más fácil y sostenía el promedio. El
  desempeño real en plagas y enfermedades es de **22–30 % de mAP@0.5** en la prueba completa.
- Las conclusiones no cambian: en mayo (fijo97) la v5 no supera a la v4 (17.6–21.3 frente a 19.2);
  en julio (nuevas83) sí (32–44 frente a 25.8).
- **En nuevas83 no hay ninguna caja de oídio**: el mAP de "4 plagas" ahí es sobre 3 clases. Confirma
  que el oídio casi no se etiquetó en julio (ver `12_correccion_etiquetas.md`).
- Por clase en test180 (v5): polilla 38–62, mosca blanca 24–33, mildiu 20–26, oídio 5–8.
- Para el artículo: reportar las dos columnas (5 clases y 4 plagas) y justificar `sana`: la app la usa
  para mostrar al agricultor qué hojas revisó y están sanas; la versión siguiente probará sin ella.

## 6. Qué cambia en el artículo

1. Corregir la Tabla 4 y la Fig. 4 con los valores de la sección 1 (fijo97 con etiquetas v5).
2. Reescribir la conclusión: la v5 mejora en sesiones nuevas (generalización temporal), no en las
   mismas condiciones; las arquitecturas empatan dentro de la incertidumbre.
3. Añadir IC a mAP y acierto; tabla por imagen con falso sano, sensibilidad/especificidad y líneas base.
4. Añadir la medición de exposición por franja como resultado; el efecto sobre el detector, como
   exploratorio con su IC.
5. Cambiar "planta" por "grupo de captura" y declarar el alcance de la auditoría de duplicados.

## 7. Pendiente (otros bloques)

- Pesos de las otras 8 corridas de la v4 (Google Drive) para repetir la sección 1 con 3 semillas.
- Bloque B (GPU): experimento controlado de fuga, v4 reentrenada con la configuración de la v5,
  modelo `s` / 1024 px.
- Bloque C: modelo v5 en el celular con P1–P5 y prueba en campo.
- Bloque D: validación de etiquetas por especialista, permiso del fundo.
