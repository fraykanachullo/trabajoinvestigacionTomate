#!/usr/bin/env python3
"""Entrenamiento YOLO para la deteccion de plagas del tomate (Majes), version para supercomputadora.

Misma logica que tareas/entrenamiento_colab.ipynb, sin Colab, sin Drive y sin API key:
  1. verifica el dataset (orden de clases, conteos, imagenes estiradas, fuga por planta y video-fotos)
     y mueve las 66 imagenes con fuga video-fotos al subconjunto correcto;
  2. entrena YOLOv8n, YOLOv10n y YOLO11n con 3 semillas (retoma solo si se corta);
  3. evalua valid, test, cada franja horaria y el test fijo de la v4, mas el acierto de la clase
     principal con las reglas de la app;
  4. elige el modelo con validacion y arma un zip con los resultados.

Uso (el dataset es el zip "YOLOv8" de la version 2 de Roboflow, ya descomprimido):
  python entrenar_supercomputadora.py --dataset /ruta/dataset --salida /ruta/resultados
  python entrenar_supercomputadora.py --dataset ... --salida ... --prueba       # prueba corta (minutos)
  python entrenar_supercomputadora.py --dataset ... --salida ... --etapa verificar

Requisitos: pip install ultralytics==8.4.82 pandas
"""
import argparse
import collections
import datetime as dt
import gc
import json
import math
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--dataset', required=True, type=Path, help='carpeta con data.yaml, train/, valid/ y test/')
ap.add_argument('--salida', required=True, type=Path, help='carpeta donde se guardan las corridas y resultados')
ap.add_argument('--nombre', default='v5-1966-completas', help='nombre de esta tanda (subcarpeta de --salida)')
ap.add_argument('--version-roboflow', type=int, default=2)
ap.add_argument('--esperado', default='1370,404,192', help='imagenes train,valid,test tal como vienen de Roboflow')
ap.add_argument('--modelos', default='yolov8n.pt,yolov10n.pt,yolo11n.pt')
ap.add_argument('--semillas', default='0,1,2')
ap.add_argument('--epochs', type=int, default=150)
ap.add_argument('--patience', type=int, default=30)
ap.add_argument('--imgsz', type=int, default=640, help='la app usa 640: no cambiar sin ajustar la app')
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--device', default='0', help="'0' = primera GPU, '0,1' = dos GPU, 'cpu'")
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--etapa', default='todo', choices=['verificar', 'entrenar', 'evaluar', 'todo'])
ap.add_argument('--prueba', action='store_true',
                help='prueba corta: 1 modelo, 1 semilla, 1 epoca, 10%% de train; guarda en <nombre>_prueba')
a = ap.parse_args()

CLASES = ['mildiu', 'mosca_blanca', 'oidio', 'polilla', 'sana']   # mismo orden que la app y el backend
DATASET = a.dataset.resolve()
NOMBRE = a.nombre + ('_prueba' if a.prueba else '')
RES = a.salida.resolve() / NOMBRE
TRABAJO = RES / 'trabajo'
MODELOS = a.modelos.split(',')
SEMILLAS = [int(s) for s in a.semillas.split(',')]
EPOCHS, FRACCION = a.epochs, 1.0
if a.prueba:
    MODELOS, SEMILLAS, EPOCHS, FRACCION = MODELOS[-1:], SEMILLAS[:1], 1, 0.1
ESPERADO = dict(zip(('train', 'valid', 'test'), (int(x) for x in a.esperado.split(','))))
DEVICE = a.device if a.device == 'cpu' or ',' in a.device else int(a.device)
EXTS = {'.jpg', '.jpeg', '.png'}

# Correccion de fuga video <-> fotos (tareas/06_bitacora_dataset.md, seccion 6b): 66 imagenes que van a otro
# subconjunto. En Roboflow tienen el tag mover-a-train / mover-a-valid / mover-a-test. Si ya estan bien, no hace nada.
MOVER = {   # nombre: (subconjunto en Roboflow, subconjunto correcto)
    'frames_extraidosv1__video_20260503_143122__20260503_143122_frame_002430': ('train', 'valid'),
    'frames_extraidosv1__video_20260503_174007__20260503_174007_frame_000330': ('valid', 'train'),
    'frames_extraidosv1__video_20260503_174007__20260503_174007_frame_001770': ('valid', 'train'),
    'frames_extraidosv1__video_20260503_173624__20260503_173624_frame_000360': ('test', 'train'),
    'frames_extraidosv1__video_20260503_173624__20260503_173624_frame_002940': ('test', 'train'),
    'frames_extraidosv1__video_20260503_173645__20260503_173645_frame_002940': ('test', 'train'),
    '16__g041__20260503_171906': ('valid', 'train'),
    '16__g041__20260503_171924': ('valid', 'train'),
    '16__g041__20260503_171943': ('valid', 'train'),
    '16__g041__20260503_172003': ('valid', 'train'),
    '16__g041__20260503_172016': ('valid', 'train'),
    '16__g041__20260503_172025': ('valid', 'train'),
    'frames_extraidosv1__video_20260503_171654__20260503_171654_frame_000000': ('valid', 'train'),
    '16__g029__20260503_165433': ('test', 'train'),
    '16__g029__20260503_165439': ('test', 'train'),
    '16__g031__20260503_165815': ('valid', 'train'),
    '16__g031__20260503_165821': ('valid', 'train'),
    '16__g031__20260503_165848': ('valid', 'train'),
    '16__g031__20260503_165858': ('valid', 'train'),
    '16__g031__20260503_165903': ('valid', 'train'),
    '16__g031__20260503_165907': ('valid', 'train'),
    'frames_extraidosv1__video_20260503_165721__20260503_165721_frame_000000': ('test', 'train'),
    'frames_extraidosv1__video_20260503_165721__20260503_165721_frame_000060': ('test', 'train'),
    'frames_extraidosv1__video_20260503_165721__20260503_165721_frame_000660': ('test', 'train'),
    'frames_extraidosv1__video_20260503_165721__20260503_165721_frame_001230': ('test', 'train'),
    '16__g008__20260503_161618': ('valid', 'train'),
    '16__g008__20260503_161621': ('valid', 'train'),
    '16__g025__20260503_164855': ('valid', 'train'),
    '16__g025__20260503_164858': ('valid', 'train'),
    '16__g025__20260503_164906': ('valid', 'train'),
    '16__g025__20260503_164915': ('valid', 'train'),
    '16__g026__20260503_165015': ('test', 'train'),
    '16__g026__20260503_165030': ('test', 'train'),
    'frames_extraidosv1__video_20260503_164830__20260503_164830_frame_000000': ('valid', 'train'),
    'frames_extraidosv1__video_20260503_162752__20260503_162752_frame_007290': ('train', 'valid'),
    'frames_extraidosv1__video_20260503_100125__20260503_100125_frame_000000': ('test', 'train'),
    'frames_extraidosv1__video_20260503_100125__20260503_100125_frame_000600': ('test', 'train'),
    'frames_extraidosv1__video_20260503_100125__20260503_100125_frame_000810': ('test', 'train'),
    '8__g045__20260503_091145': ('test', 'valid'),
    'version2__V3__4__g016__20260729_151950': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_151912__20260729_151912_frame_000390': ('test', 'train'),
    'version2__V3__frames_videosv2__video_20260729_153942__20260729_153942_frame_000210': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_153942__20260729_153942_frame_000570': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_144721__20260729_144721_frame_000000': ('train', 'valid'),
    'version2__V3__frames_videosv2__video_20260729_144721__20260729_144721_frame_000030': ('train', 'valid'),
    'version2__V3__frames_videosv2__video_20260729_144721__20260729_144721_frame_000060': ('train', 'valid'),
    'version2__V3__frames_videosv2__video_20260729_144721__20260729_144721_frame_000090': ('train', 'valid'),
    'version2__V3__frames_videosv2__video_20260729_144721__20260729_144721_frame_000150': ('train', 'valid'),
    'version2__V3__frames_videosv2__video_20260729_150335__20260729_150335_frame_000060': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_150335__20260729_150335_frame_000240': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_150335__20260729_150335_frame_000600': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_150335__20260729_150335_frame_004200': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_150335__20260729_150335_frame_004980': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_150335__20260729_150335_frame_005430': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_150335__20260729_150335_frame_005550': ('valid', 'train'),
    'version2__V3__4__g014__20260729_151159': ('train', 'valid'),
    'version2__V3__4__g014__20260729_151205': ('train', 'valid'),
    'version2__V3__4__g014__20260729_151206': ('train', 'valid'),
    'version2__V3__4__g014__20260729_151207': ('train', 'valid'),
    'version2__V3__4__g014__20260729_151208': ('train', 'valid'),
    'version2__V3__4__g014__20260729_151209': ('train', 'valid'),
    'version2__V3__4__g004__20260729_143539': ('valid', 'test'),
    'version2__V3__4__g004__20260729_143543': ('valid', 'test'),
    'version2__V3__4__g004__20260729_143547': ('valid', 'test'),
    'version2__V3__4__g010__20260729_145436': ('valid', 'train'),
    'version2__V3__frames_videosv2__video_20260729_151053__20260729_151053_frame_000060': ('valid', 'test'),
}

# Test fijo = las 102 imagenes de test de la v4 (tareas/test_fijo_v4.txt): todas las versiones se comparan con
# estas mismas imagenes. Las 5 que salen de test por la fuga video-fotos quedan fuera solas.
FIJOS = set([
    '12__g008__20260503_121120',
    '12__g008__20260503_121130',
    '12__g008__20260503_121158',
    '12__g008__20260503_121218',
    '12__g008__20260503_121243',
    '12__g010__20260503_121522',
    '12__g010__20260503_121526',
    '12__g010__20260503_121543',
    '12__g012__20260503_121831',
    '12__g013__20260503_121918',
    '12__g016__20260503_122602',
    '12__g020__20260503_124012',
    '12__g020__20260503_124030',
    '12__g020__20260503_124051',
    '12__g021__20260503_124926',
    '12__g021__20260503_124930',
    '12__g021__20260503_124938',
    '12__g021__20260503_124953',
    '12__g021__20260503_125002',
    '12__g021__20260503_125006',
    '12__g021__20260503_125007',
    '12__g027__20260503_130848',
    '12__g027__20260503_130908',
    '12__g035__20260503_133643',
    '12__g035__20260503_133705',
    '12__g035__20260503_133726',
    '12__g035__20260503_133731',
    '12__g035__20260503_133734',
    '12__g035__20260503_133738',
    '12__g035__20260503_133743',
    '12__g043__20260503_135920',
    '12__g043__20260503_135926',
    '12__g043__20260503_135940',
    '12__g043__20260503_135947',
    '12__g043__20260503_135955',
    '12__g043__20260503_140015',
    '12__g043__20260503_140022',
    '12__g043__20260503_140038',
    '12__g044__20260503_140120',
    '12__g044__20260503_140137',
    '12__g044__20260503_140143',
    '12__g052__20260503_142832',
    '12__g052__20260503_142836',
    '12__g052__20260503_142840',
    '12__g052__20260503_142900',
    '12__g053__20260503_142931',
    '12__g053__20260503_142936',
    '12__g053__20260503_142946',
    '12__g057__20260503_143722',
    '12__g057__20260503_143725',
    '12__g057__20260503_143728',
    '12__g057__20260503_143737',
    '12__g057__20260503_143741',
    '12__g060__20260503_144459',
    '12__g060__20260503_144505',
    '12__g060__20260503_144514',
    '16__g001__20260503_160359',
    '16__g001__20260503_160406',
    '16__g004__20260503_160943',
    '16__g004__20260503_160946',
    '16__g014__20260503_162651',
    '16__g015__20260503_162828',
    '16__g015__20260503_162845',
    '16__g015__20260503_162856',
    '16__g015__20260503_162858',
    '16__g018__20260503_163323',
    '16__g018__20260503_163338',
    '16__g022__20260503_164137',
    '16__g022__20260503_164139',
    '16__g022__20260503_164151',
    '16__g022__20260503_164209',
    '16__g023__20260503_164307',
    '16__g023__20260503_164325',
    '16__g023__20260503_164345',
    '16__g023__20260503_164348',
    '16__g026__20260503_165015',
    '16__g026__20260503_165030',
    '16__g029__20260503_165433',
    '16__g029__20260503_165439',
    '16__g037__20260503_171052',
    '16__g037__20260503_171054',
    '8__g001__20260503_080603',
    '8__g013__20260503_082304',
    '8__g015__20260503_082606',
    '8__g016__20260503_082703',
    '8__g017__20260503_082751',
    '8__g017__20260503_082808',
    '8__g028__20260503_084215',
    '8__g028__20260503_084228',
    '8__g032__20260503_084748',
    '8__g032__20260503_084807',
    '8__g044__20260503_091011_001',
    '8__g045__20260503_091145',
    '8__g049__20260503_091559',
    '8__g049__20260503_091616',
    '8__g055__20260503_092220',
    '8__g058__20260503_092629',
    '8__g058__20260503_092658',
    '8__g062__20260503_093426',
    '8__g063__20260503_093730',
    '8__g067__20260503_094210',
    '8__g070__20260503_094708',
])


def log(*x):
    print(*x, flush=True)


def nombre_base(p):
    # Roboflow agrega _jpg.rf.<hash> al nombre original
    return re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', Path(p).stem, flags=re.I)


def grupo(p):
    # <carpeta>__g015__20260503_120027 -> <carpeta>__g015 (planta o video)
    return nombre_base(p).rsplit('__', 1)[0]


def periodo(p):
    # Hora real de captura, leida del nombre: AAAAMMDD_HHMMSS
    hora = int(re.findall(r'\d{8}_(\d{2})\d{4}', nombre_base(p))[-1])
    return 'manana' if hora < 11 else 'mediodia' if hora < 14 else 'tarde'


def imagenes(split):
    return sorted(p for p in (DATASET / split / 'images').iterdir() if p.suffix.lower() in EXTS)


def etiqueta(p):
    return p.parent.parent / 'labels' / (p.stem + '.txt')


def clases_en(p):
    t = etiqueta(p)
    return [int(l.split()[0]) for l in t.read_text().splitlines() if l.strip()] if t.exists() else []


def rango_grupo(rutas):
    nb = [nombre_base(p) for p in rutas]
    if '__video_' in nb[0]:
        ini = dt.datetime.strptime(re.search(r'video_(\d{8}_\d{6})', nb[0]).group(1), '%Y%m%d_%H%M%S')
        cu = [int(re.search(r'frame_(\d+)', n).group(1)) for n in nb]
        return True, ini + dt.timedelta(seconds=min(cu) / 30), ini + dt.timedelta(seconds=max(cu) / 30)
    t = sorted(dt.datetime.strptime(re.findall(r'\d{8}_\d{6}', n)[-1], '%Y%m%d_%H%M%S') for n in nb)
    return False, t[0], t[-1]


# ---------------------------------------------------------------- 1. Verificar el dataset
def verificar():
    import yaml
    from PIL import Image

    info = yaml.safe_load((DATASET / 'data.yaml').read_text())
    assert list(info['names']) == CLASES, f"Orden de clases distinto al de la app: {info['names']}"

    conteo_rf = {s: len(imagenes(s)) for s in ('train', 'valid', 'test')}
    log('Imagenes (como vienen de Roboflow):', conteo_rf)
    if conteo_rf != ESPERADO:
        # Si ya se movieron en una ejecucion anterior, se cuentan de nuevo en su subconjunto de Roboflow
        original = collections.Counter()
        for s in ('train', 'valid', 'test'):
            for p in imagenes(s):
                de, hacia = MOVER.get(nombre_base(p), (s, s))
                original[de if s == hacia else s] += 1
        assert dict(original) == ESPERADO, f'Se esperaba {ESPERADO}. ¿Se uso Rebalance o es otra version?'
        log('  (las imagenes con fuga video-fotos ya estaban movidas de una ejecucion anterior)')

    movidas = 0
    for s in ('train', 'valid', 'test'):
        for p in imagenes(s):
            destino = MOVER.get(nombre_base(p), (s, s))[1]
            if destino != s:
                for origen, carpeta in ((p, 'images'), (etiqueta(p), 'labels')):
                    if origen.exists():
                        (DATASET / destino / carpeta).mkdir(parents=True, exist_ok=True)
                        shutil.move(str(origen), str(DATASET / destino / carpeta / origen.name))
                movidas += 1
    log(f'Imagenes movidas de subconjunto por fuga video-fotos: {movidas}')

    splits = {s: imagenes(s) for s in ('train', 'valid', 'test')}
    log('Imagenes para entrenar:', {s: len(v) for s, v in splits.items()})

    donde = collections.defaultdict(set)
    por_grupo = collections.defaultdict(list)
    for s, v in splits.items():
        for p in v:
            donde[grupo(p)].add(s)
            por_grupo[grupo(p)].append(p)
    fuga = {g: sorted(s) for g, s in donde.items() if len(s) > 1}
    assert not fuga, f'Fuga de datos: {len(fuga)} grupos en mas de un subconjunto, ej. {list(fuga.items())[:3]}'
    log(f'Grupos (plantas/videos): {len(donde)}, ninguno repartido entre subconjuntos')

    # Un video grabado a < 30 s de una rafaga de fotos es casi seguro la misma planta
    rg = {g: (*rango_grupo(v), next(iter(donde[g]))) for g, v in por_grupo.items()}
    gap = dt.timedelta(seconds=30)
    cerca = [(g, h) for g, (vid, a0, a1, s) in rg.items() if vid
             for h, (vid2, b0, b1, t) in rg.items() if not vid2 and t != s and b0 - gap <= a1 and b1 + gap >= a0]
    assert not cerca, f'Fuga video-fotos: {len(cerca)} pares en distinto subconjunto, ej. {cerca[:3]}'
    log('Videos y rafagas cercanas: todos en el mismo subconjunto')

    muestra = splits['train'][:: max(1, len(splits['train']) // 50)]
    cuadradas = sum(Image.open(p).width == Image.open(p).height for p in muestra)
    assert cuadradas < len(muestra) / 2, 'Imagenes cuadradas: la version parece hecha con Stretch. Generala con Fit within.'

    log('\nCajas por clase:')
    for s, v in splits.items():
        c = collections.Counter(k for p in v for k in clases_en(p))
        log(f'  {s:6}', {CLASES[k]: c[k] for k in range(len(CLASES))})
    log('Imagenes de test por franja:', dict(collections.Counter(periodo(p) for p in splits['test'])))
    log(f'Test fijo: {len(FIJOS)} imagenes, {sum(nombre_base(p) in FIJOS for p in splits["test"])} presentes en test')
    return splits


def escribir_yaml(nombre, test='test/images'):
    import yaml
    # Rutas absolutas: las de Roboflow son relativas y fallan segun la carpeta actual
    f = TRABAJO / f'{nombre}.yaml'
    f.write_text(yaml.safe_dump({
        'path': str(DATASET), 'train': 'train/images', 'val': 'valid/images', 'test': test,
        'names': dict(enumerate(CLASES)),
    }))
    return f


def subconjunto(nombre, rutas):
    # test = lista de imagenes: asi se evalua cada franja por separado
    lista = TRABAJO / f'{nombre}.txt'
    lista.write_text('\n'.join(str(p) for p in rutas) + '\n')
    return escribir_yaml(nombre, test=str(lista))


def liberar():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- 2. Entrenar
def carpeta_corrida(modelo, semilla):
    return RES / 'corridas' / f'{Path(modelo).stem}_s{semilla}'


def entrenar(data, modelo, semilla):
    from ultralytics import YOLO

    d = carpeta_corrida(modelo, semilla)
    if (d / 'terminado.json').exists():
        log(f'[listo] {d.name}')
        return
    t0 = time.time()
    ultimo = d / 'weights' / 'last.pt'
    try:
        if ultimo.exists():
            log(f'[continua] {d.name}')
            YOLO(str(ultimo)).train(resume=True)
        else:
            log(f'[entrena] {d.name}')
            YOLO(modelo).train(
                data=str(data), epochs=EPOCHS, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
                seed=semilla, deterministic=True, device=DEVICE, workers=a.workers, fraction=FRACCION,
                project=str(d.parent), name=d.name, exist_ok=True,
            )
    except AssertionError as e:
        # Ultralytics no reanuda una corrida que ya llego al final
        if 'finished' not in str(e) and 'nothing to resume' not in str(e):
            raise
    (d / 'terminado.json').write_text(json.dumps({'modelo': modelo, 'semilla': semilla, 'segundos': time.time() - t0}))
    liberar()


# ---------------------------------------------------------------- 3. Evaluar
def wilson(ok, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = ok / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - m, c + m)


def evaluar(splits, data):
    import pandas as pd
    import torch
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    sana = CLASES.index('sana')
    dispositivo = DEVICE if DEVICE == 'cpu' or torch.cuda.is_available() else 'cpu'
    pruebas = {'valid': (data, 'val'), 'test': (data, 'test')}
    for f in ('manana', 'mediodia', 'tarde'):
        rutas = [p for p in splits['test'] if periodo(p) == f]
        if rutas:
            pruebas[f'test_{f}'] = (subconjunto(f'test_{f}', rutas), 'test')
    pruebas['test_fijo'] = (subconjunto('test_fijo', [p for p in splits['test'] if nombre_base(p) in FIJOS]), 'test')
    rutas_acierto = {
        'test': splits['test'],
        'test_fijo': [p for p in splits['test'] if nombre_base(p) in FIJOS],
        **{f'test_{f}': [p for p in splits['test'] if periodo(p) == f] for f in ('manana', 'mediodia', 'tarde')},
    }

    def acierto_principal(model):
        # Misma regla que la app: deteccion de mayor confianza; sin detecciones cuenta como sana
        pred = {}
        for p in splits['test']:
            if not clases_en(p):
                continue
            im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
            b = model(im, conf=0.30, iou=0.45, imgsz=a.imgsz, agnostic_nms=True, device=dispositivo, verbose=False)[0].boxes
            top = int(b.cls[b.conf.argmax()]) if len(b) else sana
            pred[p] = top == collections.Counter(clases_en(p)).most_common(1)[0][0]
        filas = []
        for nombre, rutas in rutas_acierto.items():
            r = [pred[p] for p in rutas if p in pred]
            lo, hi = wilson(sum(r), len(r))
            filas.append({'prueba': nombre, 'imagenes': len(r), 'acierto': sum(r) / len(r) if r else float('nan'),
                          'ic95_bajo': lo, 'ic95_alto': hi})
        return filas

    general, por_clase, acierto = [], [], []
    corridas = sorted(d for d in (RES / 'corridas').iterdir() if (d / 'terminado.json').exists())
    assert corridas, 'Todavia no hay corridas terminadas'
    for d in corridas:
        log(f'[evalua] {d.name}')
        arq, sem = d.name.rsplit('_s', 1)
        model = YOLO(str(d / 'weights' / 'best.pt'))
        base = {'modelo': arq, 'semilla': int(sem), 'corrida': d.name}
        for nombre, (dat, split) in pruebas.items():
            m = model.val(data=str(dat), split=split, imgsz=a.imgsz, batch=a.batch, device=dispositivo, plots=True,
                          project=str(d / 'evaluacion'), name=nombre, exist_ok=True, verbose=False)
            general.append({**base, 'prueba': nombre, 'mAP50': m.box.map50, 'mAP50_95': m.box.map,
                            'precision': m.box.mp, 'recall': m.box.mr})
            for i, k in enumerate(m.box.ap_class_index):
                por_clase.append({**base, 'prueba': nombre, 'clase': CLASES[int(k)], 'precision': m.box.p[i],
                                  'recall': m.box.r[i], 'AP50': m.box.ap50[i], 'AP50_95': m.box.ap[i]})
        acierto += [{**base, **f} for f in acierto_principal(model)]
        del model
        liberar()

    general, por_clase, acierto = pd.DataFrame(general), pd.DataFrame(por_clase), pd.DataFrame(acierto)
    general.to_csv(RES / 'metricas_generales.csv', index=False)
    por_clase.to_csv(RES / 'metricas_por_clase.csv', index=False)
    acierto.to_csv(RES / 'acierto_clase_principal.csv', index=False)
    return pruebas, general, por_clase, acierto


# ---------------------------------------------------------------- 4. Resumen, eleccion y paquete
def resumir(pruebas, general, por_clase, acierto):
    import pandas as pd

    def resumen(df, por, cols):
        g = df.groupby(por)[cols]
        r = g.mean().round(4).astype(str) + ' ± ' + g.std().fillna(0).round(4).astype(str)
        r.insert(0, 'corridas', df.groupby(por).size())
        return r

    lineas = ['Media ± desviacion entre semillas\n']
    cols = ['mAP50', 'mAP50_95', 'precision', 'recall']
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        for prueba in pruebas:
            lineas += [f'--- {prueba} ---', resumen(general[general.prueba == prueba], 'modelo', cols).to_string(), '']
        lineas += ['--- Test por clase ---',
                   resumen(por_clase[por_clase.prueba == 'test'], ['modelo', 'clase'],
                           ['precision', 'recall', 'AP50', 'AP50_95']).to_string(), '']
        lineas += ['--- Acierto de la clase principal (reglas de la app) ---',
                   resumen(acierto, ['modelo', 'prueba'], ['acierto']).to_string(), '']

    # El modelo se elige con validacion, no con test
    val = general[general.prueba == 'valid']
    arq = val.groupby('modelo').mAP50_95.mean().idxmax()
    mejor = val[val.modelo == arq].sort_values('mAP50_95', ascending=False).iloc[0]
    lineas.append(f'Elegido por validacion: {arq} (corrida {mejor.corrida}, mAP50-95 valid = {mejor.mAP50_95:.4f})')
    if arq.startswith('yolov10'):
        lineas.append('Aviso: la salida de YOLOv10 es distinta a la de v8/v11; la app necesitara un ajuste.')
    texto = '\n'.join(lineas)
    log('\n' + texto)
    (RES / 'resumen.txt').write_text(texto + '\n', encoding='utf-8')

    destino = RES / 'mejor_modelo'
    destino.mkdir(exist_ok=True)
    shutil.copy2(RES / 'corridas' / mejor.corrida / 'weights' / 'best.pt', destino / 'best.pt')
    fila = lambda p: general[(general.corrida == mejor.corrida) & (general.prueba == p)].iloc[0]
    (destino / 'datos.json').write_text(json.dumps({
        'nombre': NOMBRE, 'modelo': arq, 'corrida': mejor.corrida, 'version_roboflow': a.version_roboflow,
        'imgsz': a.imgsz, 'clases': CLASES,
        'valid_mAP50_95': float(fila('valid').mAP50_95), 'test_mAP50': float(fila('test').mAP50),
        'test_mAP50_95': float(fila('test').mAP50_95),
    }, indent=2))

    paquete = RES / f'resultados_{NOMBRE}.zip'
    with zipfile.ZipFile(paquete, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in RES.rglob('*'):
            if not f.is_file() or f == paquete:
                continue
            # De los pesos solo el mejor modelo; de cada corrida, sus curvas y resultados
            if f.suffix == '.pt' and destino not in f.parents:
                continue
            z.write(f, f.relative_to(RES))
    log(f'Listo: {paquete} ({paquete.stat().st_size / 1e6:.1f} MB)')


def main():
    log(f'Dataset: {DATASET}\nResultados: {RES}\nModelos: {MODELOS}  semillas: {SEMILLAS}  epocas: {EPOCHS}'
        f'  imgsz: {a.imgsz}  batch: {a.batch}  device: {DEVICE}' + ('  [PRUEBA CORTA]' if a.prueba else ''))
    TRABAJO.mkdir(parents=True, exist_ok=True)
    splits = verificar()
    data = escribir_yaml('data')
    if a.etapa == 'verificar':
        return
    if a.etapa in ('entrenar', 'todo'):
        for semilla in SEMILLAS:          # primero se completa la semilla 0 de todos los modelos
            for modelo in MODELOS:
                entrenar(data, modelo, semilla)
        log('Entrenamiento completo')
    if a.etapa in ('evaluar', 'todo'):
        resumir(*evaluar(splits, data))


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        log(f'\nERROR: {e}')
        sys.exit(1)
