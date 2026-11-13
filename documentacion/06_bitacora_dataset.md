# Bitácora del dataset: qué se hizo y cómo

Registro paso a paso de la preparación de las imágenes de campo para etiquetar y entrenar.
Sirve como respaldo de la metodología (sección de datos del PPI/artículo).

---

## 1. Qué imágenes se usaron

**Se trabajó sobre `D:\ProyectoInvestigacion\dataset_procesado`**, no sobre el crudo, porque el
procesado ya contiene **todas** las fotos del crudo más los cuadros extraídos de videos.

Verificación (comparando nombres de archivo, 29/10/2026):

| Carpeta | Fotos | Videos | Comentario |
|---|---|---|---|
| `datasetcrudo` | 1697 | 58 | Originales del celular |
| `dataset_procesado` | 1970 | — | Las **1697 fotos del crudo** (todas, ninguna falta) **+ 273 cuadros de video** |

- Fotos del crudo que no están en el procesado: **0**.
- Cuadros de video en el procesado: 120 en `frames_extraidosv1` (sesión 03/05) y 153 en
  `version2/V3/frames_videosv2` (sesión 29/07), salidos de **47 de los 58 videos**.
- **11 videos no tienen cuadros extraídos** (p. ej. `20260503_161912`, `20260503_161933`,
  `20260503_162207`, de la carpeta de las 16 h). Se pueden extraer después si hacen falta más
  ejemplos de esa hora.
- **4 fotos de WhatsApp** (`IMG-20260503-WA0001.jpg`, `…WA0014`, `…WA0018`, `…WA0023`) no tienen la
  hora en el nombre, así que no se pudieron agrupar por planta y **quedaron fuera** (1970 − 4 = 1966).
  Revisarlas a mano y agregarlas si son útiles.

Sesiones de campo (El Pedregal, Majes):

| Fecha | Carpetas | Imágenes |
|---|---|---|
| 03/05/2026 | `8`, `12`, `16` (hora de captura) + `frames_extraidosv1` | 209 + 558 + 261 + 120 |
| 12/07/2026 | `version2/V2/8`, `version2/V2/12` | 293 + 50 |
| 29/07/2026 | `version2/V3/4` + `frames_videosv2` | 326 + 153 |

---

## 2. Revisión del "Plan de Trabajo y Hoja de Ruta"

**Qué:** se leyó el documento completo (`PLAN DE TRABAJO Y HOJA DE RUTA (1).docx`) y se verificó
cada cita buscando el artículo real en internet (PubMed Central, Frontiers, MDPI, ScienceDirect, ECCV).

**Cómo:** se extrajo el texto del `.docx` con Python (leyendo `word/document.xml`) y se buscó cada
autor + título + año.

**Resultado:** de 14 citas, 2 correctas, 5 reales pero mal citadas o usadas para afirmar algo que
no dicen, 7 no encontradas. Detalle y correcciones en [05_etiquetado.md](05_etiquetado.md) §1–3.

---

## 3. Análisis del dataset actual de Roboflow (`fray-hxtls/plagas-tomate-majes`, v2)

**Qué:** revisar el dataset con el que se entrenó el modelo actual
(`D:\ProyectoInvestigacion\plagas_tomate_majes\datasets`, 1028 imágenes ya etiquetadas).

**Cómo y resultados:**

1. **Copias por aumento de datos:** se comparó el nombre original de cada imagen
   (lo que está antes de `.rf.`). Resultado: 0 copias; cada imagen es una foto distinta.
2. **Misma foto en dos subconjuntos:** 0.
3. **Fuga por planta (ráfagas):** cada imagen se asignó a su grupo de planta (paso 4) y se vio en
   qué subconjunto estaba. Resultado: **65 de 189 plantas** tienen fotos repartidas entre
   train/valid/test y **82 de las 103 imágenes de test** tienen una foto "hermana" en
   train o valid. ⇒ El 68 % de acierto del modelo actual está **inflado**.
4. **Cómo se redimensionaron:** se comparó cada imagen de 640×640 con su original reducido de dos
   formas (estirado y con bordes), midiendo la correlación de píxeles. Resultado: correlación
   **1.000 con "estirado"** ⇒ se exportaron con *Stretch to 640×640* (fotos 3:4 deformadas a 1:1).
5. **Consecuencia útil:** al estirar, las coordenadas normalizadas de las cajas no cambian, así que
   **las 1024 etiquetas existentes valen tal cual sobre las fotos originales** y se reutilizaron.

---

## 4. Agrupación por planta y división train/valid/test

**Qué:** evitar la fuga de datos (Regla 5): todas las fotos de una misma planta en un solo
subconjunto.

**Cómo** (script `backend/scripts/agrupar_split.py`):

1. Se lee la fecha y hora del nombre de cada archivo (`AAAAMMDD_HHMMSS.jpg`).
2. Dentro de cada carpeta se ordenan las fotos por hora. Si entre una foto y la siguiente pasan
   **más de 30 segundos**, se considera otra planta (nuevo grupo). Con 90 s se juntaban varias plantas
   fotografiadas caminando por el surco.
3. Los cuadros de un mismo video forman un solo grupo.
4. Los grupos se reparten 70 / 20 / 10 **por número de imágenes** dentro de cada carpeta (así cada
   hora del día queda en los tres subconjuntos). Se usa semilla fija (42): siempre da el mismo
   resultado.

**Resultado:** **288 grupos** → **1370 train / 404 valid / 192 test**. La lista completa (archivo,
grupo, subconjunto, fecha) está en `dataset_procesado/split_por_grupo.csv`.

---

## 5. Carpeta para subir a Roboflow

**Qué:** dejar todo listo para arrastrar a Roboflow sin perder el split por planta ni las etiquetas
previas.

**Cómo** (mismo script, opción `--copiar`):

```bash
cd D:\ProyectoInvestigacion\app\backend
.\venv311\Scripts\python.exe scripts\agrupar_split.py --src D:\ProyectoInvestigacion\dataset_procesado --copiar D:\ProyectoInvestigacion\para_roboflow --etiquetas D:\ProyectoInvestigacion\plagas_tomate_majes\datasets
```

- Cada foto se reduce a **2048 px** de lado mayor **sin deformarla** (~0.6 MB por imagen; deja
  margen para entrenar a 640 o 1024).
- El nombre de cada archivo lleva su grupo, p. ej. `12__g015__20260503_120027.jpg`: carpeta 12 h,
  planta 15. Así se puede rastrear la planta dentro de Roboflow.
- Si la foto ya tenía etiqueta en el dataset viejo, se copia su `.txt` (formato YOLO).
- Se escribe `data.yaml` con las 5 clases en el orden de la app:
  `mildiu, mosca_blanca, oidio, polilla, sana`.

**Resultado** (`D:\ProyectoInvestigacion\para_roboflow`, 1.2 GB):

| | train | valid | test |
|---|---|---|---|
| Imágenes | 1370 | 404 | 192 |
| Con etiquetas previas | 716 | 205 | 103 |
| Por etiquetar | 654 | 199 | 89 |
| Cajas mildiu | 797 | 268 | 153 |
| Cajas mosca_blanca | 206 | 31 | 17 |
| Cajas oidio | 412 | 162 | 94 |
| Cajas polilla | **87** | **26** | **9** |
| Cajas sana | 1017 | 297 | 77 |

Tiempo de generación: ~50 min (las fotos de 50 MP son lentas de abrir).

---

## 5b. Subida a Roboflow (`fray-hxtls/plagas-tomate-majes-v2`, privado)

Subida hecha por Fray desde la web (carpeta `para_roboflow`, método **"Use Existing Values"**).
Verificación con el MCP de Roboflow (29/10/2026):

| | Esperado | En Roboflow |
|---|---|---|
| Imágenes totales | 1966 | 1966 ✅ |
| Etiquetadas (en Dataset) train / valid / test | 715 / 204 / 102 | 715 / 204 / 102 ✅ |
| Cajas mildiu / oidio / sana / mosca_blanca / polilla | 1218 / 668 / 1391 / 254 / 122 | iguales ✅ |
| Sin etiquetar (lote "Folder: para_roboflow") train / valid / test | 655 / 200 / 90 | 655 / 200 / 88 (2 aún indexándose) |

- Las imágenes sin etiquetar **conservan su split**: al etiquetarlas entran al Dataset en el
  subconjunto correcto.
- 3 imágenes tenían el `.txt` vacío y quedaron como "sin etiquetar":
  `16__g001__20260503_160410` (test), `16__g044__20260503_172609` (train),
  `16__g051__20260503_174405` (valid). Revisarlas y marcarlas *Null* si son solo fondo.

## 5c. Etiquetas (tags) por hora real de captura y por sesión

**Por qué:** el nombre de la carpeta (`8`, `12`, `16`, `V3/4`) no coincide siempre con la hora real
(p. ej. `V2/12` se tomó a las 14:49; `16` llega hasta las 17:44; `V3/4` va de 14:22 a 16:24).
Para el experimento de radiación solar del PPI se usa la **hora del nombre del archivo**
(`AAAAMMDD_HHMMSS`).

**Cómo:** con el MCP de Roboflow (`images_batch_update_metadata`, 29/10/2026), buscando por patrón de
nombre (`filename:"..._11*"`, etc.). Solo se agregaron tags; no se tocaron imágenes, etiquetas ni split.

| Tag | Rango | Imágenes | De ellas en test |
|---|---|---|---|
| `hora-manana` | antes de 11:00 | 555 | 53 |
| `hora-mediodia` | 11:00 – 13:59 | 397 | 35 |
| `hora-tarde` | 14:00 en adelante | 1014 | 104 |
| `sesion-20260503` | 03/05/2026 (carpetas 8, 12, 16 + frames v1) | 1144 | |
| `sesion-20260712` | 12/07/2026 (V2/8, V2/12) | 343 | |
| `sesion-20260729` | 29/07/2026 (V3/4 + frames v2) | 479 | |

Total de cada grupo de tags: 1966, sin imágenes sin tag. Un cuadro de `frames_extraidosv1`
proviene de un video de las 14:31, así que lleva `hora-tarde`, no `hora-manana`.

**Cómo filtrar en Roboflow** (buscador de imágenes del proyecto):
`tag:hora-mediodia`, `tag:hora-tarde split:test`, `tag:sesion-20260729`.
Uso previsto: después de entrenar, medir el acierto en **test** por separado para mediodía (35)
frente a mañana (53) y tarde (104). Ojo: 35 imágenes de mediodía en test es una muestra chica; reportar
el intervalo de confianza o al menos el número de imágenes junto al porcentaje.

## 5d. Etiquetado completo (revisión con el MCP, 12/11/2026)

- **1966 / 1966 imágenes etiquetadas.** 1022 ya en el Dataset y 944 en 3 trabajos de anotación
  (Job 2: 422, compañero; Job 3: 422; "Selected images": 100), todos al 100 % pero **todavía en
  "Annotating"**. Pasados al Dataset con el MCP (`split_method: preset`, es decir, cada imagen
  con su subconjunto ya asignado): Dataset final **1370 / 404 / 192**, 0 imágenes fuera.
- Las 944 conservan su subconjunto (655 / 200 / 89-90) y 0 imágenes con el tag `dudosa`.
- Cajas totales: sana 3356, mildiu 2698, oídio 889, **polilla 474** (antes 122), **mosca blanca
  349** (antes 254).
- Imágenes que contienen cada clase (train / valid / test): polilla 357 / 80 / 29;
  mosca blanca 139 / 51 / 31; en test, oídio 105 y mildiu 64 imágenes.
- **7 imágenes sin cajas** (fondo): 5 son paisajes o cuadros de video borrosos (correcto), pero
  `version2__V3__4__g020__20260729_154232` (hoja grande en primer plano con capa blanquecina) y
  `16__g044__20260503_172609` (hojas blanquecinas de cerca) conviene revisarlas: si es oídio,
  dejarlas sin caja enseña al modelo que eso es fondo.

**Versión 2 en Roboflow (12/11/2026):** 1966 imágenes, 1370 / 404 / 192, auto-orient +
*Fit within* 1024 (igual que la versión 1), sin aumentos. El notebook quedó con `NOMBRE =
'v5-1966-completas'`, `VERSION = 2` y ese `ESPERADO`; después de mover las 66 imágenes de la fuga
video-fotos entrena con 1405 / 381 / 180. El test fijo de la v4 (102 imágenes) se guardó en
`tareas/test_fijo_v4.txt` y va dentro del notebook, así ya no depende de Drive.

## 6b. Fuga entre videos y fotos de la misma planta (revisión del 05/11/2026)

**Qué:** el script del §4 agrupa las fotos por ráfaga y cada video por separado. Un video grabado
a segundos de una ráfaga de fotos es casi seguro la misma planta, pero pudo caer en otro subconjunto.

**Cómo** (script `backend/scripts/revisar_fuga_videos.py`, sobre `split_por_grupo.csv`):

1. Cada video ocupa desde su primer hasta su último cuadro extraído (a 30 cuadros/s).
2. Si un video y una ráfaga de fotos están a menos de **30 s** (la misma regla que en el §4), se
   consideran la misma planta. Los grupos unidos forman un componente.
3. Cada componente con imágenes en más de un subconjunto se lleva entero al subconjunto donde ya
   tiene más imágenes, para mover lo mínimo.

**Resultado:** **18 componentes con fuga, 66 imágenes a mover** (3.4 % del total):

| De → a | Ya etiquetadas | Sin etiquetar |
|---|---|---|
| test → train | 4 | 11 |
| test → valid | 1 | 0 |
| valid → test | 0 | 4 |
| valid → train | 18 | 15 |
| train → valid | 0 | 13 |

Antes 1370 / 404 / 192 → después **1405 / 381 / 180**. La lista exacta (nombre en Roboflow, de,
a) está en `tareas/movimientos_split_v5.csv`.

**Consecuencias:**

- El test de la v4 tenía **5 de 102 imágenes** con una "hermana" de video en train o valid. Es
  poco y el 24 % de mAP50 sigue siendo válido, pero para comparar la v5 con la v4 hay que evaluar
  las dos sobre las mismas imágenes de test limpias.
- **Aplicado (05/11/2026):** el MCP de Roboflow no permite cambiar el subconjunto de una imagen
  (solo rebalancear al azar, que rompería la partición). Por eso:
  1. En Roboflow las 66 imágenes tienen el tag `mover-a-train` (48), `mover-a-valid` (14) o
     `mover-a-test` (4). No se cambió nada más.
  2. El notebook trae la lista y **mueve esas imágenes solo después de descargar** el dataset.
     Si ya se movieron en Roboflow, no hace nada. Probado sobre una copia de la partición:
     1370 / 404 / 192 → 1405 / 381 / 180 y pasan los dos controles de fuga.
  3. `ESPERADO` se compara con lo que viene de Roboflow (antes de mover).
- El notebook ahora también revisa esta fuga (celda de verificación): si un video y una ráfaga
  cercana están en distinto subconjunto, se detiene. Probado: con la partición actual detecta los
  25 pares; con los movimientos aplicados pasa.
- No se puede detectar si la **misma planta** se fotografió en mayo y en julio (no hay
  identificador de planta). Por eso, además del test general, conviene reportar el test de cada
  sesión por separado.

## 6. Lo que NO se tocó

- `datasetcrudo` y `dataset_procesado`: **solo lectura**. No se modificó, movió ni borró ninguna foto.
- El proyecto anterior de Roboflow (`plagas-tomate-majes`): solo se leyó, no se cambió nada.
- El único archivo nuevo dentro de `dataset_procesado` es `split_por_grupo.csv`.

---

## 7. Pendientes

- [ ] Revisar en `split_por_grupo.csv` 3–4 grupos al azar (¿cada grupo es una sola planta?).
- [ ] Revisar las 4 fotos de WhatsApp y, si sirven, renombrarlas con su hora y volver a generar.
- [ ] (Opcional) Extraer cuadros de los 11 videos que faltan, **sin deformar** y 1 cada 2–3 s.
- [x] Crear proyecto nuevo en Roboflow y subir `para_roboflow` (ver §5b).
- [x] Tags de hora real y sesión en Roboflow (ver §5c).
- [ ] Lote piloto de 50–100 imágenes con las reglas corregidas.
- [x] Etiquetar las 942 restantes (12/11/2026; polilla 474 cajas, mosca blanca 349).
- [x] Versión 2 en Roboflow (1966 imágenes).
- [x] Corregir la fuga video-fotos: tags en Roboflow + movimiento automático en el notebook (ver §6b).
- [ ] Volver a generar la tabla de cajas por clase al terminar y anotarla aquí.
