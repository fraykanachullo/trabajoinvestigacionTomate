# Entrenamiento del modelo (responsable: Fray)

Guía para etiquetar, entrenar y reemplazar el modelo sin romper la app.

## 1. Situación actual

- Modelo: **YOLOv8n**, entrenado en Colab, `imgsz=640`, 100 épocas, 720 imágenes de train.
- Clases (el orden **no se puede cambiar**, la app y el backend dependen de él):

  | id | clase |
  |---|---|
  | 0 | mildiu |
  | 1 | mosca_blanca |
  | 2 | oidio |
  | 3 | polilla |
  | 4 | sana |

- Cajas por clase en el dataset actual (`plagas_tomate_majes/datasets`):

  | Clase | train | valid | test |
  |---|---|---|---|
  | sana | 1032 | 232 | 128 |
  | mildiu | 759 | 319 | 149 |
  | oídio | 459 | 137 | 72 |
  | mosca_blanca | 145 | 81 | 28 |
  | **polilla** | **84** | 25 | 14 |

- **Baseline a superar:** 68.0 % de acierto en clase principal (test, 103 imágenes).

## 2. Etiquetado (Roboflow)

- [ ] Llegar a **≥ 300–500 cajas por clase en train**, prioridad polilla y mosca blanca.
- [ ] Mantener exactamente los mismos nombres e índices de clase de la tabla de arriba.
- [ ] Criterio consistente: la caja encierra la lesión / mina / insecto, no la hoja entera
      (salvo que se decida lo contrario para "sana").
- [ ] Partición por planta o sesión de captura (no aleatoria por imagen) para evitar fuga de datos.
- [ ] Aumentos solo en train. En Roboflow: no aplicar aumentos al exportar valid/test.
- [ ] Exportar en formato **YOLOv8 / YOLO11** (carpetas `images/` y `labels/` + `data.yaml`).

## 3. Entrenamiento (Google Colab con GPU)

**Usar [entrenamiento_colab.ipynb](entrenamiento_colab.ipynb)** (subirlo a Colab). Compara YOLOv8n,
YOLOv10n y YOLO11n con 3 semillas, verifica el dataset, guarda todo en Drive, continúa si Colab se
desconecta, mide por clase y por franja horaria, y elige el modelo con validación. En cada versión
del dataset cambiar `NOMBRE`, `VERSION` y `ESPERADO` en la celda de configuración.

Referencia de las opciones básicas de entrenamiento:

```python
!pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolo11s.pt")      # o yolo11n.pt para comparar
model.train(
    data="/content/dataset/data.yaml",
    imgsz=640,
    epochs=200,
    patience=40,                 # early stopping
    batch=16,
    hsv_v=0.5, hsv_s=0.6,        # variación de brillo/saturación (radiación solar)
    fliplr=0.5, degrees=10,
    close_mosaic=15,
)
metrics = model.val(split="test")
print(metrics.box.map50, metrics.box.map)
```

Comparar al menos: `yolov8n` (actual), `yolo11n`, `yolo11s`. Guardar `results.csv` y las
matrices de confusión de cada uno para la tesis.

## 4. Exportar para app y backend

**Exportar en Colab (Linux)**, no en Windows: las versiones nuevas de Ultralytics solo exportan
TFLite en Linux/macOS.

```python
best = YOLO("runs/detect/train/weights/best.pt")
best.export(format="tflite", imgsz=640)   # revisar opciones de cuantización de la versión instalada
```

Si la versión instalada no ofrece float16, usar la ruta que ya funcionó con el modelo actual:
`best.pt → ONNX (imgsz=640, opset 17) → onnx2tf 1.26.3 con TensorFlow 2.18`, que genera
`best_float16.tflite` (~6 MB). La app detecta sola si las coordenadas vienen normalizadas o en píxeles.

Ruta usada con el modelo v4 (YOLO11n, 05/11/2026) en Windows:

1. ONNX con Ultralytics: `YOLO('best.pt').export(format='onnx', imgsz=640, opset=17, simplify=True)`
   (con `onnx==1.17.0` y `onnxslim`, instalados aparte para no tocar `venv311`).
2. Entorno aparte con Python 3.11: `tensorflow==2.18.*`, `tf_keras==2.18.*`, `onnx2tf==1.26.3`,
   `onnx==1.17.0`, `onnx_graphsurgeon`, `sng4onnx`, `psutil`, `numpy<2.1`, `pillow`.
3. onnx2tf intenta descargar `calibration_image_sample_data_20x128x128x3_float32.npy` y falla.
   Crear ese archivo en la carpeta de trabajo con 20 fotos del test redimensionadas a 128×128
   (float32, 0..1) y luego: `onnx2tf -i best.onnx -o tflite_out -osd`.
   El error final de `onnxsim` no importa: el ONNX ya viene simplificado.
4. Verificar: entrada `[1, 640, 640, 3]`, salida `[1, 9, 8400]`. Con la misma entrada, ONNX y TFLite
   deben dar casi el mismo score (en la v4: diferencia máxima 0.004).

Entorno fijo (no en temporales, que Windows limpia): `D:\ProyectoInvestigacion\herramientas\conversion`
(`venv_tf` con TensorFlow 2.18 y onnx2tf 1.26.3; `libs_onnx` para exportar ONNX desde `venv311` con
`PYTHONPATH`). `comparar_onnx_tflite.py <carpeta>` verifica la conversión. Usado con el v5 el 13/11/2026.

Entregables:
- `best.pt` → `backend/app/models/best.pt`
- `best_float16.tflite` → `frontend/assets/models/best_float16.tflite` (mismo nombre, lo carga `LocalDetector`)

**Importante:** exportar SIEMPRE a 640 px. El modelo anterior estaba a 320 px y perdía ~10 puntos
de acierto (57 % vs 67 %).

Ver en `02_app_arquitectura.md` (Fase 3) cómo se registra la versión del modelo.

## 5. Evaluar y comparar con el baseline

Desde la PC (con el backend instalado):

```bash
cd D:\ProyectoInvestigacion\app\backend
.\venv311\Scripts\python.exe scripts\evaluar_modelo.py --model app\models\best.pt --data D:\ProyectoInvestigacion\plagas_tomate_majes\datasets\test
```

Reporta % de acierto en clase principal y cajas por imagen para varios umbrales de confianza.
Correr el mismo script con el modelo nuevo y el actual sobre **el mismo test**.

## 6. metadata.json (obligatorio al cambiar de modelo)

Actualizar el mismo archivo en `backend/app/models/metadata.json` y `frontend/assets/models/metadata.json`:
`version` (subirla: 1.0.0 → 2.0.0), `architecture` (ej. `YOLO11s`), `trained_at`, `conf_threshold`,
y en `metrics` los aciertos medidos con `evaluar_modelo.py`. No cambiar el orden de `classes`.

## 7. Checklist para cambiar de modelo

- [ ] El modelo nuevo supera el baseline en el mismo test.
- [ ] Clases en el mismo orden (el backend valida esto al arrancar y falla si no coinciden).
- [ ] Copiar `best.pt` y `metadata.json` al backend y reiniciar.
- [ ] Copiar `best_float16.tflite` y `metadata.json` a `frontend/assets/models/`, recompilar e instalar.
- [ ] Ajustar `CONFIDENCE_THRESHOLD` en `backend/.env` según el barrido del script.
