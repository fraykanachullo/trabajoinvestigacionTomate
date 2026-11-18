# Entrenamiento v6: 4 clases y experimentos contra el brillo (17/11/2026)

Script: [entrenar_yolo_v6.py](entrenar_yolo_v6.py). Plan de fondo: [13_plan_radiacion.md](13_plan_radiacion.md).

## Estado del dataset en Roboflow (revisado con el MCP)
- 1966 imágenes, misma partición (1370 / 404 / 192). No se eliminó ninguna imagen.
- Correcciones hechas: mildiu 2698 → **2933** cajas; `sana` 3356 → 2846 (sigue en 631 imágenes).
- Las correcciones están en el proyecto, **no en la versión 2**: hay que generar la versión 3.

## Paso 1: generar la versión 3 en Roboflow (lo haces tú)
1. Proyecto → **Versions → Create New Version**.
2. Preprocessing: **Auto-Orient**, **Resize: Fit within 1024×1024**. Opcional: **Modify Classes →
   quitar `sana`** (si no lo haces, el script la quita solo).
3. Augmentation: **ninguna**. No usar Rebalance.
4. Generate → **Download Dataset → YOLOv8 → Download zip to computer**.
5. Copiar el zip a `~/PYTHON/entrenamientotomte` en la PC de la RTX, junto con `entrenar_yolo_v6.py`.
   Si en esa carpeta hay otros zip de datasets, se indica cuál con `--zip nombre.zip`.

## Paso 2: comandos (fish)

```fish
cd ~/PYTHON/entrenamientotomte
source ~/PYTHON/envia/bin/activate.fish
python entrenar_yolo_v6.py --solo-verificar --zip plagas-tomate-majes-v2.v3i.yolov8.zip
```

Debe decir `Clase sana quitada` (si venía), `Imagenes para entrenar: {'train': 1405, 'valid': 381,
'test': 180}`, `sin fuga` y `Prueba fija: 97 de 102`. Si las imágenes no coinciden, revisar que no se
haya usado Rebalance.

Prueba corta (minutos, una corrida de cada experimento):

```fish
python entrenar_yolo_v6.py --prueba
```

Entrenamiento completo dentro de tmux:

```fish
tmux new -s v6
```

```fish
python entrenar_yolo_v6.py 2>&1 | tee entrenamiento_v6.log
```

Salir sin cortarlo: `Ctrl+b` y luego `d`. Para volver: `tmux attach -t v6`. Si se corta, el mismo comando
retoma donde quedó.

## Qué entrena
| Experimento | Qué cambia | Corridas |
|---|---|---|
| base | 4 clases, etiquetas corregidas, aumento por defecto | 3 modelos × 3 semillas |
| fotometrico | `hsv_v=0.7` + una copia sobreexpuesta (+0.5 a +1.5 pasos de luz) de cada imagen de entrenamiento | 3 × 3 |
| clahe | CLAHE en todas las imágenes (y en la prueba de estrés) | 3 × 3 |

Cada corrida se evalúa en la prueba completa, la prueba fija (97, mayo), las imágenes nuevas (83, julio)
y la **prueba de estrés** (−1 a +2 pasos de luz): mAP, AP por clase y falso sano.

**Tiempo estimado en la RTX 5070 Ti:** base ~1.2 h, fotométrico ~2.5 h (el doble de imágenes), CLAHE ~1.2 h,
evaluación ~1–1.5 h. Total **~6 h**. Si hay apuro, primero solo YOLOv8n (el de la app):

```fish
python entrenar_yolo_v6.py --modelos yolov8n 2>&1 | tee entrenamiento_v6.log
```

## Al terminar
Traer `resultados_v6-4clases.zip` (incluye `comparacion_v6.csv`, `estres_v6.csv`, `resumen_v6.txt` y el
mejor modelo de cada experimento).

**Ojo con la app:** el modelo v6 tiene 4 clases (sin `sana`). Para instalarlo en la app y el backend hay que
ajustar la lista de clases (hoy esperan 5). Lo hago cuando lleguen los resultados.

---

# Segundo script: experimentos de control (obs. 2 y 3 del revisor)

Script: [entrenar_control.py](entrenar_control.py). Lanzarlo **cuando termine** `entrenar_yolo_v6.py`
(reutiliza `./dataset_v6`). Mismas etiquetas (versión 3), YOLOv8n, 3 semillas, misma configuración.

| Experimento | Qué cambia | Fuga | Responde |
|---|---|---|---|
| grupo | nada (referencia) | 0 % | — |
| aleatorio_pool | entrenamiento/validación al azar por imagen; prueba por grupo | 95 % en validación | ¿cuánto se infla la validación con fuga? (obs. 2) |
| aleatorio_total | todo al azar 70/20/10 | 99 % en prueba | ¿cuánto se infla la cifra reportada? (obs. 2) |
| solo_mayo | solo fotos del 3 de mayo (834 de entrenamiento) | 0 % | efecto de agregar las sesiones de julio (obs. 3) |
| mixto_reducido | mismo tamaño (881) con grupos de todas las sesiones | 0 % | separa "más sesiones" de "más imágenes" (obs. 3) |

```fish
python entrenar_control.py --solo-verificar --zip plagas-tomate-majes-v2.v3-v6-2026-11-17.yolov8.zip
```

```fish
python entrenar_control.py --prueba --zip plagas-tomate-majes-v2.v3-v6-2026-11-17.yolov8.zip
```

```fish
tmux new -s control
```

```fish
python entrenar_control.py --zip plagas-tomate-majes-v2.v3-v6-2026-11-17.yolov8.zip 2>&1 | tee entrenamiento_control.log
```

Tiempo estimado: 15 corridas (5 experimentos × 3 semillas), las de solo_mayo y mixto_reducido más
cortas: **~2–2.5 h**. Traer `resultados_control.zip` (incluye `control.csv`, las particiones usadas y los
`best.pt` para calcular IC por bootstrap en la PC).
