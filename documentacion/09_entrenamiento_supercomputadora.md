# Entrenamiento v5 en la supercomputadora

Script: [entrenar_supercomputadora.py](entrenar_supercomputadora.py). Hace lo mismo que
`entrenamiento_colab.ipynb`, sin Colab, sin Drive y sin API key:

1. **Verifica el dataset**: orden de clases, conteos, imágenes estiradas, fuga por planta y fuga
   video-fotos. Además **mueve las 66 imágenes con fuga video-fotos** (bitácora §6b). Si algo falla,
   se detiene con `ERROR: ...` antes de entrenar.
2. **Entrena** YOLOv8n, YOLOv10n y YOLO11n con las semillas 0, 1 y 2: 9 corridas, 150 épocas,
   `patience=30`, `imgsz=640`. Si se corta, al volver a lanzarlo retoma donde quedó.
3. **Evalúa** valid, test, cada franja horaria y el **test fijo de la v4** (97 de sus 102 imágenes
   siguen en test), más el acierto de la clase principal con las reglas de la app.
4. **Elige el modelo con validación** y arma `resultados_v5-1966-completas.zip` con el mejor
   `best.pt`, los CSV, las curvas y `resumen.txt`.

## Pasos

1. **Copiar al servidor** el script y el zip del dataset (versión 2, formato YOLOv8) y
   descomprimir el zip, por ejemplo en `~/tomate/dataset` (debe quedar `data.yaml` dentro).
2. **Instalar** en el entorno: `pip install ultralytics==8.4.82 pandas` (misma versión que la v4).
3. **Solo verificar** (segundos):
   ```bash
   python entrenar_supercomputadora.py --dataset ~/tomate/dataset --salida ~/tomate/resultados --etapa verificar
   ```
   Debe mostrar `Imagenes para entrenar: {'train': 1405, 'valid': 381, 'test': 180}` y
   `Test fijo: 102 imagenes, 97 presentes en test`.
4. **Prueba corta** (minutos: 1 modelo, 1 época, 10 % de train; guarda en `..._prueba`):
   ```bash
   python entrenar_supercomputadora.py --dataset ~/tomate/dataset --salida ~/tomate/resultados --prueba
   ```
   Confirma que usa la GPU (`device: 0` y CUDA en el registro) y que termina con `Listo: ...zip`.
5. **Entrenamiento completo**, de forma que siga si cierras la sesión:
   ```bash
   nohup python entrenar_supercomputadora.py --dataset ~/tomate/dataset --salida ~/tomate/resultados > entrenamiento.log 2>&1 &
   tail -f entrenamiento.log
   ```
   Si el servidor usa **SLURM**, ver la plantilla de abajo. Si se corta, lanzar **el mismo
   comando**: las corridas terminadas se saltan y la que quedó a medias continúa.
6. **Traer** `~/tomate/resultados/v5-1966-completas/resultados_v5-1966-completas.zip` a la PC
   (`D:\ProyectoInvestigacion\Resultados de los entrenamientos\`) para analizarlo y desplegarlo.

## Opciones útiles

| Opción | Para qué |
|---|---|
| `--batch 32` | GPU con mucha memoria (≥ 24 GB). **Mantener el mismo valor** en las 9 corridas. |
| `--device 0,1` | Usar dos GPU. |
| `--workers 8` | Hilos que cargan imágenes; bajar si el servidor da error de memoria compartida. |
| `--etapa evaluar` | Repetir solo la evaluación y el zip, sin entrenar. |
| `--modelos yolo11s.pt --nombre v5-yolo11s` | Experimentos extra del PPI (un factor a la vez y en otra carpeta). |
| `--imgsz 1024 --nombre v5-1024` | Experimento de resolución. **La app usa 640**: cambiarlo también exige ajustar la app. |

Para comparar con la v4, la corrida base debe quedar con los valores por defecto (los 3 modelos,
3 semillas, 640, batch 16). Si se cambia el batch, anotarlo.

## Plantilla SLURM (si el ingeniero la usa)

Pedirle al ingeniero el nombre de la partición y cómo se activa el entorno; lo demás puede quedar así:

```bash
#!/bin/bash
#SBATCH --job-name=tomate-v5
#SBATCH --partition=<particion-gpu>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=entrenamiento_%j.log

source <activar el entorno>   # p. ej. conda activate yolo
python entrenar_supercomputadora.py --dataset ~/tomate/dataset --salida ~/tomate/resultados --workers 8
```

Se lanza con `sbatch entrenar.sh` y se mira con `squeue -u $USER` y `tail -f entrenamiento_<id>.log`.
Si se acaba el tiempo del trabajo, volver a lanzar `sbatch`: retoma solo.

## Prueba en la PC (12/11/2026)

Probado sobre una copia de `para_roboflow` (misma partición que la versión 2):
la verificación movió las 66 imágenes, pasó los dos controles de fuga y encontró 97 del test fijo;
al correrla dos veces no movió nada. La prueba corta (`--prueba`, en CPU con `--imgsz 320`)
entrenó 1 época, evaluó las 6 pruebas y el acierto, eligió el modelo y armó el zip (17 MB): todo el
flujo funciona. Sus métricas salen casi en cero, lo esperado con 1 época.

---

# PC con RTX 5070 Ti (CachyOS + fish): `entrenar_yolo.py`

Script: [entrenar_yolo.py](entrenar_yolo.py). Es el que se le entrega al ingeniero para esa PC.
Mismas verificaciones y evaluación que el script de arriba, más lo que pidió el ingeniero:
descomprime solo el `.zip`, genera `data_fixed.yaml`, entrena **yolov8, yolo11 y yolov10** con los
mismos hiperparámetros y semillas, tamaño configurable (`--tamanos n,s`), sigue si un modelo
falla, reintenta con la mitad del batch si falta memoria, guarda `comparacion_modelos.csv` y copia
el mejor como `mejor_modelo_<corrida>.pt`. No usa API key.

## Decisiones de configuración

| Opción | Valor | Por qué |
|---|---|---|
| Tamaños | `n,s` | `n` es el que usa la app y se compara con la v4; `s` es el experimento extra (más preciso, más lento en el celular). |
| Semillas | `0,1,2` | En la v4 la diferencia entre semillas fue tan grande como entre modelos: con una sola no se puede elegir. |
| batch | **32 fijo** | Con `batch=0.85` cada modelo elige un batch distinto y la comparación deja de ser justa. 32 entra holgado en 16 GB con n y s a 640; si no entra, baja solo a 16. |
| epochs / patience | 200 / 50 | En la v4 el mejor modelo salió entre las épocas 40 y 131. |
| imgsz | 640 | La app usa 640. Subir a 1024 ayudaría a la mosca blanca y al oídio (objetos chicos), pero es otro experimento y exige ajustar la app. |
| amp, TF32, cudnn.benchmark | sí | Velocidad en la RTX 50xx. |
| deterministic | False | Más rápido; las semillas fijas siguen dando resultados comparables, aunque no idénticos bit a bit. |
| cos_lr / close_mosaic | True / 15 | Pedido del ingeniero; igual para todos los modelos. |
| cache | `ram` si el dataset ocupa < 12 GB | El dataset (1024 px) ocupa ~1 GB. |
| Elección del mejor | por **validación** | Si se elige con test, el test deja de ser una medida honesta. |

## Paso 1: verificar el entorno (en fish)

```fish
cd ~/PYTHON/entrenamientotomte
source ~/PYTHON/envia/bin/activate.fish
ls -la
unzip -l *.zip | tail -3
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.zeros(1).cuda() + 1)"
pip show ultralytics pandas
df -h ~
```

Debe salir CUDA **12.8 o más** y `tensor([1.], device='cuda:0')`. Si no (la RTX 5070 Ti es
Blackwell, sm_120):

```fish
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.4.82 pandas
```

## Paso 2: verificar el dataset y probar

Copiar `entrenar_yolo.py` a `~/PYTHON/entrenamientotomte` (junto al `.zip`).

```fish
python entrenar_yolo.py --solo-verificar
```

Debe mostrar `Imagenes para entrenar: {'train': 1405, 'valid': 381, 'test': 180}` y
`Test fijo: 102 imagenes, 97 presentes en test`. Luego la prueba corta (minutos):

```fish
python entrenar_yolo.py --prueba
```

## Paso 3: entrenamiento completo dentro de tmux

```fish
tmux new -s tomate
```

Dentro de tmux:

```fish
cd ~/PYTHON/entrenamientotomte; and source ~/PYTHON/envia/bin/activate.fish
python entrenar_yolo.py 2>&1 | tee entrenamiento.log
```

Salir de tmux sin cortarlo: `Ctrl+b` y luego `d`. Se puede cerrar Konsole. Para volver:
`tmux attach -t tomate`. Desde otra terminal:

```fish
tail -f ~/PYTHON/entrenamientotomte/entrenamiento.log
watch -n 2 nvidia-smi
```

Sin tmux (alternativa):

```fish
cd ~/PYTHON/entrenamientotomte
nohup ~/PYTHON/envia/bin/python entrenar_yolo.py > entrenamiento.log 2>&1 &
disown
```

Si se corta (apagón, reinicio), lanzar **el mismo comando**: las corridas terminadas se saltan y
la que quedó a medias continúa.

## Tiempo estimado (RTX 5070 Ti, 1405 imágenes de train a 640, batch 32)

Estimación, no medición; la primera época de cada corrida en el registro da el valor real.

| | Por época | Por corrida (parada temprana ~100–150 épocas) | 9 corridas (3 modelos × 3 semillas) |
|---|---|---|---|
| tamaño n | ~10–15 s | ~20–35 min | **~4–5 h** |
| tamaño s | ~20–30 s | ~40–70 min | **~7–10 h** |

Las dos juntas (18 corridas): ~11–15 h, más ~30 min de evaluación. Si hay apuro: primero
`--tamanos n` (comparable con la v4 y con la app) y después `--tamanos s`.

## Prueba en la PC (12/11/2026)

Con un zip armado desde `para_roboflow` (misma partición que la versión 2): descomprimió, verificó
(66 movidas, sin fuga, 97 del test fijo), entrenó 1 época en CPU, evaluó, escribió
`comparacion_modelos.csv`, copió `mejor_modelo_yolo11n_s0.pt` y armó el zip. No se pudo probar en
la PC el reintento por falta de memoria de GPU (no hay GPU NVIDIA aquí).

## Al terminar

Traer a la PC `resultados_v5-1966-completas.zip` (incluye `comparacion_modelos.csv`, el
mejor `.pt`, las curvas, las métricas por clase y franja, y `resumen.txt`).
