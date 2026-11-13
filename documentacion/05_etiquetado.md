# Etiquetado del dataset propio (revisión del "Plan de Trabajo y Hoja de Ruta")

Documento revisado: `PLAN DE TRABAJO Y HOJA DE RUTA (1).docx` (29/10/2026).

**Resumen:** la guía de etiquetado está bien planteada (cajas ajustadas, fondos negativos, casos
dudosos, split por planta), pero **varias citas no existen o no dicen lo que el documento afirma**.
Hay que corregirlas antes de ponerlas en el PPI/artículo: un jurado o revisor que busque el DOI lo
nota de inmediato. Abajo: verificación de citas, correcciones a las reglas y el flujo para Roboflow.

---

## 1. Verificación de citas

| Cita en el documento | Estado | Qué hacer |
|---|---|---|
| Dong & Lee (2022), "Data-centric annotation analysis…", *CEA / IEEE Access* | ⚠️ **Existe, pero mal citada y mal usada** | Es Dong, J., Lee, J., Fuentes, A., Xu, M., Yoon, S., Lee, M. H., & Park, D. S. (2022). *Frontiers in Plant Science*, 13, 1037655. DOI 10.3389/fpls.2022.1037655. **No** dice "cajas ajustadas +10–20 %" ni "<10 % de tejido sano": el 10–20 % es de otro trabajo citado en su introducción. Su resultado: la anotación **sintoma-adaptativa** fue la mejor (90.4 mAP@0.5) y la **más ajustada ("local") la peor** (82.2). |
| Dong et al. (2023), "Spatial localization noise…", *CEA* | ⚠️ **Existe con otro título** | Es Dong, J., Fuentes, A., Yoon, S., Kim, H., & Park, D. S. (2023). *An iterative noisy annotation correction model for robust plant disease detection*. *Frontiers in Plant Science*, 14, 1238722. El "+26 %" es de **su modelo de corrección** (Faster R-CNN) sobre datos ruidosos, no de "un protocolo estricto". |
| Gerz et al. (2026), "Evaluating structural data leakage…", *J. of Agricultural AI & CV* | ⚠️ **Existe con otro título y otra conclusión** | Es Gerz, F. et al., *Towards reliable data augmentation in machine learning: Practices to prevent data leakage* (ScienceDirect, 2026). El 98.70 % → 32.77 % mAP50 ocurre por **aumentar datos ANTES de dividir** train/test, no por el split por planta. El "86 %" no aparece; el repositorio habla de hasta **66 puntos** de mAP50. Úsalo para justificar "aumento solo en train". |
| Joeres et al. (2025), "Spatiotemporal information leakage… agricultural" | ⚠️ **Existe con otro título y otro tema** | Es Joeres, R., Blumenthal, D. B., & Kalinina, O. V. (2025). *Data splitting to avoid information leakage with DataSAIL*. *Nature Communications*, 16, 3337. Es sobre datos biológicos en general (no agrícola). Sirve como respaldo general de split sin fuga. |
| Xiang et al. (2023), "Yolo-Pest: Multi-class small pest detection with background enhancement…", *CEA* | ⚠️ **Existe con otro título** | Es Xiang, Q. et al. (2023). *Yolo-Pest: An insect pest object detection algorithm via CAC3 module*. *Sensors*, 23(6), 3221. Usa **aumento con fondos aleatorios**, no imágenes negativas; respalda la idea solo parcialmente. |
| Jelali (2024), *Frontiers in Plant Science* | ✅ Correcta | Ya está en el PPI. |
| Yang et al. (2024), SRNet-YOLO | ✅ Existe | Título real: *SRNet-YOLO: A model for detecting tiny and very tiny pests in cotton fields based on super-resolution reconstruction*. *Frontiers in Plant Science*, 15, 1416940. Define "tiny" < 32×32 px y "very tiny" < 16×16 px. Respalda la Regla 2. |
| Patil et al. (2026) | ❌ **No encontrada** | Quitar. |
| Luo et al. (2024), Pest-YOLO random background | ❌ **No encontrada** | Quitar. (Existe un *Pest-YOLO* de Wen et al., 2022, pero trata otra cosa.) |
| Tang et al. (2026), EP-YOLO | ❌ **No encontrada** | Quitar. |
| Gálvez Jiménez et al. (2024) | ❌ **No encontrada** | Quitar. |
| Chen et al. (2022), Harada et al. (2025), Gupta et al. (2026), Li et al. (2026) | ❌ **No verificadas** (no se hallaron con esos títulos) | Quitar salvo que tengas el PDF y el DOI. |
| **Falta:** Woo et al. (2018), CBAM | ➕ Agregar | Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module*. ECCV 2018, pp. 3–19. Es la cita base de tu mecanismo de atención. |

**Regla de oro:** solo citar artículos de los que tengas el PDF o el DOI abierto. Las frases
"listas para copiar" de Consensus/ChatGPT inventan títulos, revistas y porcentajes.

---

## 2. Correcciones a la guía de etiquetado

1. **Regla 1 (cajas ajustadas):** mantenerla, pero sin el "+10–20 %" ni el "<10 % de tejido sano"
   atribuidos a Dong. Redacción correcta: *cajas que abarquen el síntoma completo con el mínimo
   fondo, de forma consistente entre anotadores; Dong et al. (2022) mostraron que la estrategia y la
   consistencia de anotación cambian el mAP hasta en ~8 puntos, y que la mejor fue la
   sintoma-adaptativa*. En la práctica: la caja abarca la lesión **y su halo**, no solo el centro.
2. **Contradicción con la clase `sana`:** la Regla 3 dice que el follaje sano va **sin cajas**
   (fondo negativo), pero la matriz dice que `sana` se **etiqueta con caja**. Hay que elegir uno.
   Recomendación (compatible con la app actual de 5 clases):
   - `sana` = caja sobre **folíolos completamente sanos y bien visibles**. Solo hojas: los
     **frutos, flores y tallos sanos van sin caja** (el modelo diagnostica hojas; si `sana` incluye
     frutos, la app diría "sana" al apuntar a un tomate).
   - Fondos negativos (sin cajas) = **solo** suelo, tuberías, manos, cielo, sombras, malla.
3. **Regla 2 (>16×16 px):** aclarar que se mide **en la imagen redimensionada a 640 px**. Tus fotos
   son de hasta 8160×6120: 16 px a 640 equivalen a ~200 px en la foto original. Una mosca blanca que
   en la foto mide 100 px desaparece a 640. Para mosca blanca conviene fotos macro (15–20 cm, como
   dice tu PPI) o entrenar con `imgsz=1024`.
4. **Regla 4 (casos dudosos):** "no etiquetar lo dudoso" deja una lesión real **sin caja**, y el
   modelo aprende que eso es fondo (falso negativo enseñado). Mejor: si una imagen tiene una lesión
   que no puedes clasificar, **excluye la imagen completa** (etiqueta Roboflow `dudosa`), no la dejes
   a medias.
5. **Mosca blanca "grupo denso":** definir una sola regla: caja individual por insecto si se
   distingue; caja de grupo solo si se tocan y no se pueden separar. Siempre igual.
6. **Polilla:** incluye dos cosas muy distintas (minas en la hoja y la polilla adulta). Está bien
   como una clase, pero etiquetar **las minas** (es lo que se ve en campo) y anotar en el informe
   cuántas cajas son de adulto.
7. **Orden de clases:** Roboflow numera en orden alfabético: `mildiu (0), mosca_blanca (1),
   oidio (2), polilla (3), sana (4)`. Es el mismo orden que usa la app y el backend: **no renombrar**.
8. **Hojas superpuestas (una sana encima de una enferma):** una caja **por hoja**, cada una con su
   clase; las cajas pueden encimarse (YOLO lo admite). La hoja de abajo se encierra **solo en su
   parte visible** con síntoma, no donde se supone que sigue. Nunca una caja que junte las dos.
   Si de la hoja tapada no se reconoce el síntoma, aplicar el punto 4 (imagen `dudosa`).
   Hojas amarillentas o con clorosis **no** son `sana`.
9. **Síntomas en tallos y peciolos** (p. ej. oídio sobre el tallo): una caja por **tramo continuo
   afectado** (de nudo a nudo o hasta donde cambia de dirección), no varios parches pegados del mismo
   tallo. Si el tallo va en diagonal y la caja quedaría casi toda de fondo, partirlo en 2–3 tramos.
   Cada caja debe medir al menos 16 px a 640 (≈ 50 px en la foto de 2048 de `para_roboflow`).
10. **Etiquetar todo lo reconocible del primer plano:** lo que se deja sin caja el modelo lo aprende
    como fondo. El follaje lejano o borroso no se etiqueta. Si una foto tiene demasiadas lesiones
    para etiquetarlas todas, marcarla `dudosa` en vez de dejarla a medias.

---

## 3. Correcciones al plan de entrenamiento (fases 2–4 del documento)

1. **CBAM en el YAML:** con la escala `n` (ancho 0.25) las capas C2f de 512 y 256 canales quedan en
   **128 y 64**. Escribir `CBAM, [512]` y `CBAM, [256]` da error de dimensiones. Usar `[128]` y `[64]`
   (o registrar CBAM para que tome los canales de la capa anterior). Se ajusta al entrenar.
2. **Comparación justa (ablación):** en el plan, el modelo base usa `yolov8n.pt` (preentrenado) y el
   CBAM se entrena **desde cero** (`YOLO('yolov8n-cbam.yaml')`) y con otros hiperparámetros. Así la
   comparación no vale. Ambos deben usar **los mismos** hiperparámetros, aumentos, épocas y semilla,
   y el CBAM debe cargar pesos: `YOLO('yolov8n-cbam.yaml').load('yolov8n.pt')`.
3. `augment=True` en `train()` no activa aumentos (es para TTA en predicción): quitarlo.
4. **Incluir YOLO11n** (el PPI promete comparar v8/v10/v11). YOLO11 ya trae atención (C2PSA): sirve
   como segundo punto de comparación.
5. **Umbrales por clase (Fase 4):** el diccionario usa clases que **no existen** ("Arañita Roja",
   "Minador") y un orden equivocado. El correcto es `0 mildiu, 1 mosca_blanca, 2 oidio, 3 polilla,
   4 sana`. Los umbrales se eligen con **validación**, nunca con test.
6. **Exportación:** usar 640 px y float16 (ver `03_entrenamiento_modelo.md`). El INT8 del plan da
   menos acierto y requiere datos de calibración.
7. **Cuadros de video:** `extraer_frames_version2.py` usa `RESIZE = (640, 640)` y **deforma** la
   imagen. Volver a extraerlos sin redimensionar (Roboflow redimensiona después manteniendo
   proporción) y tomar 1 cuadro cada 2–3 s para evitar casi-duplicados.

---

## 4. Tus imágenes (dataset_procesado)

| Sesión | Carpeta | Imágenes |
|---|---|---|
| 03/05/2026 | `8`, `12`, `16` (horas) + `frames_extraidosv1` | 209 + 558 + 261 + 120 |
| 12/07/2026 | `version2/V2/8`, `version2/V2/12` | 293 + 50 |
| 29/07/2026 | `version2/V3/4` + `frames_videosv2` | 326 + 153 |
| **Total** | | **1970** (+4 de WhatsApp sin fecha) |

Las carpetas por hora (8 h, 12 h, 16 h) permiten el experimento de saturación lumínica del PPI
(comparar el acierto en fotos de 12 h contra 8 h y 16 h).

**Ojo:** el dataset actual (`plagas_tomate_majes`) es de la sesión del 03/05. Si se reetiqueta, la
nueva partición debe reemplazar a la anterior, no mezclarse con ella.

### Hallazgos sobre el dataset actual (Roboflow `fray-hxtls/plagas-tomate-majes`, versión 2)

- **Fuga de datos medida:** 82 de las 103 imágenes de *test* tienen una foto "hermana" (misma
  planta, tomada segundos antes o después) en *train* o *valid*: 65 de 189 plantas quedaron
  repartidas entre subconjuntos. El 68 % del modelo actual está **inflado**. Dato útil para la
  tesis: justifica el split por planta con evidencia propia.
- **Imágenes estiradas:** se exportaron con "Stretch to 640×640" (fotos 3:4 deformadas a 1:1),
  mientras que la app analiza sin deformar. La nueva versión debe usar **Fit** (sin estirar).
- **Etiquetas reutilizables:** 1024 de las 1970 fotos ya están etiquetadas. Como el estirado no
  cambia las coordenadas normalizadas, esas cajas valen tal cual sobre la foto original. Solo hay
  que revisarlas con las reglas nuevas y etiquetar las ~950 restantes.

---

## 5. Split por planta (Regla 5) — script listo

`backend/scripts/agrupar_split.py` agrupa por ráfaga (fotos con menos de 30 s entre sí = misma
planta; cuadros de un mismo video = mismo grupo) y reparte los grupos 70/20/10 por número de
imágenes, dentro de cada carpeta/hora:

```bash
cd D:\ProyectoInvestigacion\app\backend
.\venv311\Scripts\python.exe scripts\agrupar_split.py --src D:\ProyectoInvestigacion\dataset_procesado
```

Resultado actual: **288 grupos → 1370 train / 404 valid / 192 test**, y un CSV
`dataset_procesado/split_por_grupo.csv` para revisar. Con `--copiar D:\ProyectoInvestigacion\para_roboflow`
copia las fotos a `train/`, `valid/` y `test/` con el grupo en el nombre.

- [ ] Revisar en el CSV 3–4 grupos al azar: ¿son de la misma planta? Ajustar `--gap` si hace falta.
- [ ] Revisar a mano las 4 imágenes `IMG-...-WA....jpg` (sin fecha).

---

## 6. Flujo en Roboflow

Carpeta lista para subir (formato YOLO, con las etiquetas previas y el split por planta):

```bash
cd D:\ProyectoInvestigacion\app\backend
.\venv311\Scripts\python.exe scripts\agrupar_split.py --src D:\ProyectoInvestigacion\dataset_procesado --copiar D:\ProyectoInvestigacion\para_roboflow --etiquetas D:\ProyectoInvestigacion\plagas_tomate_majes\datasets
```

Genera `para_roboflow/{train,valid,test}/{images,labels}` + `data.yaml`, con las fotos reducidas a
2048 px de lado mayor (sin deformar) y el grupo/planta en el nombre del archivo.

**Generada el 29/10/2026** (1.2 GB): 1966 imágenes, 1024 con etiquetas previas y 942 por etiquetar.

| Cajas previas | train (1370 img) | valid (404) | test (192) |
|---|---|---|---|
| mildiu | 797 | 268 | 153 |
| mosca_blanca | 206 | 31 | 17 |
| oidio | 412 | 162 | 94 |
| polilla | **87** | **26** | **9** |
| sana | 1017 | 297 | 77 |

Polilla y mosca blanca siguen muy escasas: al etiquetar las 942 nuevas, priorizar esas dos clases
(meta ≥ 300 cajas cada una en train). Solo 9 cajas de polilla en test es poco para medirla bien.

1. [ ] **Proyecto nuevo** tipo **Object Detection** (p. ej. `plagas-tomate-majes-v2`); dejar el
       actual intacto como referencia del modelo base de la tesis.
2. [ ] *Upload* → arrastrar la carpeta `para_roboflow` completa. Roboflow detecta el formato YOLO,
       importa las cajas previas y **respeta train/valid/test** por las carpetas. (No usar el
       split aleatorio de Roboflow: rompe el split por planta.)
3. [ ] **Lote piloto** de 50–100 imágenes, revisarlo juntos y recién después etiquetar el resto.
4. [ ] Marcar con la etiqueta `dudosa` las imágenes ambiguas para excluirlas.
5. [ ] Al generar la versión: preprocesado **Resize 640 (fit, sin estirar)**; **sin aumentos** en
       Roboflow (los hace Ultralytics solo sobre train).
6. [ ] Exportar en formato **YOLOv8** y comparar con el baseline usando `evaluar_modelo.py`.

---

## 7. Cómo etiquetar las imágenes restantes sin romper la partición (05/11/2026)

La partición por planta **ya está hecha para las 1966 imágenes**, también para las que faltan
etiquetar: en Roboflow las 945 pendientes ya tienen su subconjunto (655 train / 200 valid / 90 test,
verificado con el MCP). Al etiquetarlas entran solas al subconjunto correcto si se respetan estas
reglas:

1. **Al pasar las imágenes al Dataset** ("Add approved to Dataset") elegir **Use existing values**.
   Nunca "Split images between train/valid/test": reparte al azar y vuelve la fuga.
2. **No usar Rebalance** al generar la versión. El notebook lo detecta (el número de imágenes por
   subconjunto no coincide con `ESPERADO`).
3. **No mover imágenes de subconjunto a mano.** Las 66 imágenes con fuga video-fotos (tags
   `mover-a-train/valid/test`, bitácora §6b) las mueve el notebook solo al descargar; no hace
   falta tocarlas en Roboflow. Etiquétalas normal.
4. **Fotos nuevas** (otra salida al campo): no subirlas sueltas. Pasarlas primero por
   `agrupar_split.py` para que reciban su grupo `__gNNN__` y su subconjunto, y después por
   `revisar_fuga_videos.py`.
5. **Imagen dudosa:** marcarla con el tag `dudosa` y no borrarla ni cambiarla de subconjunto.
6. **Test fijo:** las 102 imágenes de test de la v4 se siguen usando (`test_fijo` en el notebook)
   para comparar versiones, menos las 5 que el notebook saca del test por el punto 3.
7. Al generar la versión: **Resize 640 Fit (sin estirar)**, sin aumentos, y actualizar `NOMBRE`,
   `VERSION` y `ESPERADO` en el notebook con los números que muestre Roboflow.
