#!/usr/bin/env python3
"""Entrenamiento v6 - plagas y enfermedades del tomate (Majes): 4 clases y experimentos contra el brillo solar.

Pensado para la PC con RTX 5070 Ti (CachyOS, fish). No necesita API key: usa el .zip de Roboflow (formato
YOLOv8) que esta en la carpeta de trabajo.

Que hace:
  1. Descomprime el zip en ./dataset y escribe data_fixed.yaml con rutas absolutas.
  2. Si el zip trae la clase `sana`, la quita (sus cajas pasan a fondo) y deja 4 clases:
     mildiu, mosca_blanca, oidio, polilla.
  3. Verifica conteos, imagenes no estiradas y fuga (grupo de captura y video-fotos); mueve las 66
     imagenes con fuga video-fotos.
  4. Entrena por EXPERIMENTO, con los mismos hiperparametros y semillas:
       base        : aumento por defecto de Ultralytics (hsv_v = 0.4)
       fotometrico : hsv_v = 0.7 + una copia sobreexpuesta de cada imagen de entrenamiento
                     (+0.5 a +1.5 pasos de luz con recorte de pixeles quemados); solo en entrenamiento
       clahe       : todas las imagenes (entrenamiento, validacion, prueba) con ecualizacion adaptativa
                     del contraste (CLAHE) en el canal de luminosidad
  5. Evalua cada corrida en la prueba completa, la prueba fija (97, mayo), las imagenes nuevas (83, julio)
     y la PRUEBA DE ESTRES (la misma prueba sobreexpuesta de -1 a +2 pasos de luz): mAP, AP por clase y
     tasa de falso sano. Guarda comparacion_v6.csv y estres_v6.csv, imprime la tabla y copia el mejor
     modelo de cada experimento (elegido por validacion).

Uso (dentro de ~/PYTHON/entrenamientotomte con el entorno activado):
  python entrenar_yolo_v6.py --solo-verificar
  python entrenar_yolo_v6.py --prueba                         # 1 corrida corta de cada experimento
  python entrenar_yolo_v6.py                                  # base, fotometrico y clahe; 3 modelos x 3 semillas
  python entrenar_yolo_v6.py --experimentos base,fotometrico --modelos yolov8n
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
import traceback
import zipfile
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--carpeta', type=Path, default=Path.cwd())
ap.add_argument('--zip', type=Path, default=None)
ap.add_argument('--experimentos', default='base,fotometrico,clahe')
ap.add_argument('--modelos', default='yolov8n,yolo11n,yolov10n', help='yolov8n es el que usa la app')
ap.add_argument('--semillas', default='0,1,2')
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--patience', type=int, default=50)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--device', default='0')
ap.add_argument('--esperado', default='1370,404,192', help='imagenes train,valid,test como vienen de Roboflow')
ap.add_argument('--nombre', default='v6-4clases')
ap.add_argument('--solo-verificar', action='store_true')
ap.add_argument('--solo-evaluar', action='store_true')
ap.add_argument('--prueba', action='store_true', help='1 modelo, 1 semilla, 1 epoca, 10%% de train por experimento')
a = ap.parse_args()

CARPETA = a.carpeta.expanduser().resolve()
CLASES = ['mildiu', 'mosca_blanca', 'oidio', 'polilla']
ENF = set(range(4))
EXPERIMENTOS = [e.strip() for e in a.experimentos.split(',') if e.strip()]
MODELOS = [m.strip() for m in a.modelos.split(',') if m.strip()]
SEMILLAS = [int(s) for s in a.semillas.split(',')]
EPOCHS, FRACCION, NOMBRE = a.epochs, 1.0, a.nombre
if a.prueba:
    MODELOS, SEMILLAS, EPOCHS, FRACCION, NOMBRE = MODELOS[:1], SEMILLAS[:1], 1, 0.1, NOMBRE + '_prueba'
DATASET = CARPETA / 'dataset_v6'          # carpeta propia: no se mezcla con ./dataset de la v5
TRABAJO = CARPETA / 'trabajo_v6'
RUNS = CARPETA / 'runs' / NOMBRE
ESPERADO = dict(zip(('train', 'valid', 'test'), (int(x) for x in a.esperado.split(','))))
DEVICE = a.device if a.device == 'cpu' or ',' in a.device else int(a.device)
EXTS = {'.jpg', '.jpeg', '.png'}
NIVELES_EV = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]

# Imagenes con fuga video-fotos: nombre -> (subconjunto en Roboflow, subconjunto correcto)
MOVER = {
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

# Prueba fija = las 102 imagenes de prueba de la v4 (97 siguen en prueba)
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
    n = re.sub(r'\.(jpg|jpeg|png|txt)$', '', Path(p).name, flags=re.I)
    return re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', n, flags=re.I)


def grupo(p):
    return nombre_base(p).rsplit('__', 1)[0]


def imagenes(d, split):
    q = d / split / 'images'
    return sorted(p for p in q.iterdir() if p.suffix.lower() in EXTS) if q.exists() else []


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


# ------------------------------------------------------------------ transformaciones de imagen
def a_lineal(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def a_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.clip(x, 0, None), 1 / 2.4) - 0.055)


def sobreexponer(rgb, ev):
    """rgb uint8 -> uint8 con +ev pasos de luz (en luz lineal) y recorte de pixeles quemados."""
    x = rgb.astype(np.float32) / 255
    y = np.clip(a_srgb(np.clip(a_lineal(x) * 2 ** ev, 0, 1)), 0, 1)
    return (y * 255 + 0.5).astype(np.uint8)


def clahe(rgb):
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def leer(p):
    from PIL import Image, ImageOps
    return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert('RGB'))


def guardar(rgb, q):
    from PIL import Image
    Image.fromarray(rgb).save(q, quality=95)


# ------------------------------------------------------------------ 1. dataset
def preparar_dataset():
    if not (DATASET / 'data.yaml').exists():
        zips = [a.zip.expanduser().resolve()] if a.zip else sorted(CARPETA.glob('*.zip'))
        zips = [z for z in zips if not z.name.startswith('resultados_')]
        assert zips, f'No hay ningun .zip del dataset en {CARPETA}'
        assert len(zips) == 1, f'Hay varios .zip; indica cual con --zip: {[z.name for z in zips]}'
        log(f'Descomprimiendo {zips[0].name} ...')
        tmp = CARPETA / 'dataset_v6_tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(tmp)
        y = sorted(tmp.rglob('data.yaml'), key=lambda p: len(p.parts))
        assert y, 'El zip no tiene data.yaml: exportalo en formato YOLOv8'
        shutil.move(str(y[0].parent), str(DATASET))
        shutil.rmtree(tmp, ignore_errors=True)
    if not (DATASET / 'valid').exists() and (DATASET / 'val').exists():
        (DATASET / 'val').rename(DATASET / 'valid')


def quitar_sana():
    """Si el dataset trae `sana`, borra sus cajas y renumera las clases a las 4 de CLASES."""
    import yaml
    info = yaml.safe_load((DATASET / 'data.yaml').read_text())
    nombres = list(info['names'].values()) if isinstance(info['names'], dict) else list(info['names'])
    if nombres == CLASES:
        return 0
    assert set(nombres) - {'sana'} == set(CLASES), f'Clases inesperadas en el zip: {nombres}'
    nuevo = {i: CLASES.index(n) for i, n in enumerate(nombres) if n != 'sana'}
    quitadas = 0
    for lbl in DATASET.rglob('labels/*.txt'):
        out = []
        for l in lbl.read_text().splitlines():
            v = l.split()
            if not v:
                continue
            k = int(v[0])
            if k in nuevo:
                out.append(' '.join([str(nuevo[k])] + v[1:]))
            else:
                quitadas += 1
        lbl.write_text('\n'.join(out) + ('\n' if out else ''))
    info['names'] = CLASES
    info['nc'] = len(CLASES)
    (DATASET / 'data.yaml').write_text(yaml.safe_dump(info))
    log(f'Clase `sana` quitada: {quitadas} cajas pasan a fondo; clases = {CLASES}')
    return quitadas


def verificar():
    from PIL import Image
    conteo_rf = {s: len(imagenes(DATASET, s)) for s in ('train', 'valid', 'test')}
    log('Imagenes (como vienen de Roboflow):', conteo_rf)
    if conteo_rf != ESPERADO:
        original = collections.Counter()
        for s in conteo_rf:
            for p in imagenes(DATASET, s):
                de, hacia = MOVER.get(nombre_base(p), (s, s))
                original[de if s == hacia else s] += 1
        assert dict(original) == ESPERADO, f'Se esperaba {ESPERADO}. ¿Rebalance, imagenes borradas u otra version? Ajusta --esperado'
        log('  (las imagenes con fuga video-fotos ya estaban movidas)')
    movidas = 0
    for s in ('train', 'valid', 'test'):
        for p in imagenes(DATASET, s):
            destino = MOVER.get(nombre_base(p), (s, s))[1]
            if destino != s:
                for origen, sub in ((p, 'images'), (etiqueta(p), 'labels')):
                    if origen.exists():
                        (DATASET / destino / sub).mkdir(parents=True, exist_ok=True)
                        shutil.move(str(origen), str(DATASET / destino / sub / origen.name))
                movidas += 1
    log(f'Imagenes movidas por fuga video-fotos: {movidas}')
    splits = {s: imagenes(DATASET, s) for s in ('train', 'valid', 'test')}
    log('Imagenes para entrenar:', {s: len(v) for s, v in splits.items()})
    donde, por_grupo = collections.defaultdict(set), collections.defaultdict(list)
    for s, v in splits.items():
        for p in v:
            donde[grupo(p)].add(s)
            por_grupo[grupo(p)].append(p)
    fuga = {g: sorted(s) for g, s in donde.items() if len(s) > 1}
    assert not fuga, f'Fuga: {len(fuga)} grupos en mas de un subconjunto, ej. {list(fuga.items())[:3]}'
    rg = {g: (*rango_grupo(v), next(iter(donde[g]))) for g, v in por_grupo.items()}
    gap = dt.timedelta(seconds=30)
    cerca = [(g, h) for g, (vid, a0, a1, s) in rg.items() if vid
             for h, (vid2, b0, b1, t) in rg.items() if not vid2 and t != s and b0 - gap <= a1 and b1 + gap >= a0]
    assert not cerca, f'Fuga video-fotos: {len(cerca)} pares, ej. {cerca[:3]}'
    log(f'Grupos de captura: {len(donde)}, sin fuga')
    muestra = splits['train'][:: max(1, len(splits['train']) // 50)]
    assert sum(Image.open(p).width == Image.open(p).height for p in muestra) < len(muestra) / 2, 'Imagenes estiradas: usa Fit within'
    for s, v in splits.items():
        c = collections.Counter(k for p in v for k in clases_en(p))
        log(f'  cajas {s:6}', {CLASES[k]: c[k] for k in range(len(CLASES))}, '| imagenes sin cajas:', sum(not clases_en(p) for p in v))
    log(f'Prueba fija: {sum(nombre_base(p) in FIJOS for p in splits["test"])} de {len(FIJOS)} imagenes presentes')
    return splits


# ------------------------------------------------------------------ 2. versiones del dataset por experimento
def copiar_split(src, dst, split, transformar=None):
    (dst / split / 'images').mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / split / 'labels', dst / split / 'labels', dirs_exist_ok=True)
    for p in imagenes(src, split):
        q = dst / split / 'images' / p.name
        if q.exists():
            continue
        if transformar is None:
            shutil.copy2(p, q)
        else:
            guardar(transformar(leer(p)), q)


def dataset_de(exp):
    """Carpeta del dataset para el experimento (se crea una vez)."""
    if exp == 'base':
        return DATASET
    d = TRABAJO / f'dataset_{exp}'
    marca = d / '.listo'
    if marca.exists():
        return d
    log(f'Preparando dataset del experimento {exp} ...')
    if exp == 'fotometrico':
        for s in ('valid', 'test'):
            copiar_split(DATASET, d, s)
        copiar_split(DATASET, d, 'train')
        rng = np.random.default_rng(0)
        for p in imagenes(DATASET, 'train'):
            ev = float(rng.uniform(0.5, 1.5))
            q = d / 'train' / 'images' / f'{p.stem}__sobreexp{p.suffix}'
            if not q.exists():
                guardar(sobreexponer(leer(p), ev), q)
            lbl = etiqueta(p)
            if lbl.exists():
                shutil.copy2(lbl, d / 'train' / 'labels' / f'{p.stem}__sobreexp.txt')
    elif exp == 'clahe':
        for s in ('train', 'valid', 'test'):
            copiar_split(DATASET, d, s, clahe)
    else:
        raise ValueError(f'Experimento desconocido: {exp}')
    marca.write_text('ok')
    return d


def escribir_yaml(ruta, d, test='test/images'):
    import yaml
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(yaml.safe_dump({'path': str(d), 'train': 'train/images', 'val': 'valid/images',
                                    'test': test, 'names': dict(enumerate(CLASES))}))
    return ruta


def prueba_estres(exp):
    """Prueba sobreexpuesta de -1 a +2 pasos; para clahe, CLAHE despues de la sobreexposicion."""
    salida = {}
    for ev in NIVELES_EV:
        d = TRABAJO / f'estres_{exp}' / f'ev{ev:+.1f}'
        if not (d / '.listo').exists():
            (d / 'images').mkdir(parents=True, exist_ok=True)
            shutil.copytree(DATASET / 'test' / 'labels', d / 'labels', dirs_exist_ok=True)
            for p in imagenes(DATASET, 'test'):
                x = sobreexponer(leer(p), ev) if ev != 0 else leer(p)
                guardar(clahe(x) if exp == 'clahe' else x, d / 'images' / p.name)
            (d / '.listo').write_text('ok')
        import yaml
        y = d / 'data.yaml'
        y.write_text(yaml.safe_dump({'path': str(d), 'train': 'images', 'val': 'images', 'test': 'images',
                                     'names': dict(enumerate(CLASES))}))
        salida[ev] = y
    return salida


# ------------------------------------------------------------------ 3. entrenamiento
def configurar_gpu():
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}')
    else:
        log('AVISO: PyTorch no ve la GPU.')


def liberar():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def entrenar(exp, data, modelo, semilla, fallos):
    from ultralytics import YOLO
    d = RUNS / exp / f'{modelo}_s{semilla}'
    if (d / 'terminado.json').exists():
        log(f'[listo] {exp}/{d.name}')
        return
    batch, t0 = a.batch, time.time()
    extra = {'hsv_v': 0.7} if exp == 'fotometrico' else {}
    while True:
        try:
            ultimo = d / 'weights' / 'last.pt'
            if ultimo.exists():
                log(f'[continua] {exp}/{d.name}')
                YOLO(str(ultimo)).train(resume=True)
            else:
                log(f'[entrena] {exp}/{d.name} batch={batch} {extra}')
                YOLO(f'{modelo}.pt').train(
                    data=str(data), epochs=EPOCHS, patience=a.patience, imgsz=a.imgsz, batch=batch, device=DEVICE,
                    workers=a.workers, cache='ram', amp=True, seed=semilla, deterministic=False, cos_lr=True,
                    close_mosaic=15, fraction=FRACCION, plots=True, project=str(RUNS / exp), name=d.name,
                    exist_ok=True, **extra)
            break
        except AssertionError as e:
            if 'finished' in str(e) or 'nothing to resume' in str(e):
                break
            raise
        except Exception as e:
            liberar()
            if ('out of memory' in str(e).lower() or type(e).__name__ == 'OutOfMemoryError') and batch > 2:
                batch //= 2
                log(f'  Falta memoria: reintento con batch={batch}')
                shutil.rmtree(d, ignore_errors=True)
                continue
            log(f'  FALLO {exp}/{d.name}: {type(e).__name__}: {e}')
            fallos.append({'corrida': f'{exp}/{d.name}', 'error': str(e), 'detalle': traceback.format_exc()})
            return
    (d / 'terminado.json').write_text(json.dumps({'experimento': exp, 'modelo': modelo, 'semilla': semilla,
                                                  'batch': batch, 'minutos': (time.time() - t0) / 60}))
    liberar()


# ------------------------------------------------------------------ 4. evaluacion
def subconjunto(nombre, d, rutas):
    lista = TRABAJO / f'{nombre}.txt'
    lista.write_text('\n'.join(str(p) for p in rutas) + '\n')
    return escribir_yaml(TRABAJO / f'{nombre}.yaml', d, test=str(lista))


def falso_sano(model, carpeta_img, disp):
    fs = n = 0
    for p in sorted(Path(carpeta_img).iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        lbl = Path(carpeta_img).parent / 'labels' / (p.stem + '.txt')
        cl = [int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        if not set(cl) & ENF:
            continue
        b = model(str(p), conf=0.30, iou=0.45, imgsz=a.imgsz, agnostic_nms=True, verbose=False, device=disp)[0].boxes
        n += 1
        fs += int(not (set(b.cls.int().tolist()) & ENF))
    return fs / n if n else math.nan


def evaluar():
    import pandas as pd
    import torch
    from ultralytics import YOLO
    disp = DEVICE if DEVICE == 'cpu' or torch.cuda.is_available() else 'cpu'
    filas, estres = [], []
    for exp in EXPERIMENTOS:
        d = dataset_de(exp)
        test = imagenes(d, 'test')
        pruebas = {'valid': (escribir_yaml(TRABAJO / f'data_{exp}.yaml', d), 'val'),
                   'test': (escribir_yaml(TRABAJO / f'data_{exp}.yaml', d), 'test'),
                   'prueba_fija': (subconjunto(f'fija_{exp}', d, [p for p in test if nombre_base(p) in FIJOS]), 'test'),
                   'nuevas': (subconjunto(f'nuevas_{exp}', d, [p for p in test if nombre_base(p) not in FIJOS]), 'test')}
        niveles = prueba_estres(exp)
        corridas = sorted(c for c in (RUNS / exp).iterdir() if (c / 'terminado.json').exists()) if (RUNS / exp).exists() else []
        for c in corridas:
            log(f'[evalua] {exp}/{c.name}')
            info = json.loads((c / 'terminado.json').read_text())
            model = YOLO(str(c / 'weights' / 'best.pt'))
            fila = {'experimento': exp, 'modelo': info['modelo'], 'semilla': info['semilla'], 'corrida': c.name,
                    'minutos': info.get('minutos'), 'pesos': str(c / 'weights' / 'best.pt')}
            for nombre, (y, split) in pruebas.items():
                r = model.val(data=str(y), split=split, imgsz=a.imgsz, batch=16, device=disp, plots=False, verbose=False,
                              project=str(c / 'evaluacion'), name=nombre, exist_ok=True)
                fila[f'{nombre}_mAP50'] = r.box.map50
                fila[f'{nombre}_mAP50_95'] = r.box.map
                if nombre == 'test':
                    for i, k in enumerate(r.box.ap_class_index):
                        fila[f'test_AP50_{CLASES[int(k)]}'] = r.box.ap50[i]
            fila['test_falso_sano'] = falso_sano(model, d / 'test' / 'images', disp)
            filas.append(fila)
            for ev, y in niveles.items():
                r = model.val(data=str(y), split='test', imgsz=a.imgsz, batch=16, device=disp, plots=False, verbose=False,
                              project=str(c / 'evaluacion'), name=f'estres_ev{ev:+.1f}', exist_ok=True)
                e = {'experimento': exp, 'modelo': info['modelo'], 'semilla': info['semilla'], 'ev': ev,
                     'mAP50': r.box.map50, 'falso_sano': falso_sano(model, Path(y).parent / 'images', disp)}
                for i, k in enumerate(r.box.ap_class_index):
                    e[f'AP50_{CLASES[int(k)]}'] = r.box.ap50[i]
                estres.append(e)
            del model
            liberar()
    F, E = pd.DataFrame(filas), pd.DataFrame(estres)
    F.to_csv(CARPETA / 'comparacion_v6.csv', index=False)
    E.to_csv(CARPETA / 'estres_v6.csv', index=False)
    return F, E


def resumir(F, E):
    import pandas as pd
    if F.empty:
        log('No hay corridas terminadas para resumir')
        return

    def ms(df, por, cols):
        g = df.groupby(por)[cols]
        r = (g.mean() * 100).round(1).astype(str) + ' ± ' + (g.std().fillna(0) * 100).round(1).astype(str)
        r.insert(0, 'corridas', df.groupby(por).size())
        return r

    lin = ['=== mAP@0.5 (4 clases) por experimento y modelo: media ± DE entre semillas ===']
    with pd.option_context('display.width', 240, 'display.max_columns', 40):
        lin.append(ms(F, ['experimento', 'modelo'], ['valid_mAP50', 'test_mAP50', 'prueba_fija_mAP50', 'nuevas_mAP50', 'test_falso_sano']).to_string())
        cls = [c for c in F.columns if c.startswith('test_AP50_')]
        lin += ['', '--- AP@0.5 por clase en prueba ---', ms(F, ['experimento'], cls).to_string()]
        ap_mosca = [c for c in E.columns if c == 'AP50_mosca_blanca']
        lin += ['', '--- Prueba de estres (media de modelos y semillas): mAP@0.5, falso sano y AP de mosca blanca ---',
                ms(E, ['experimento', 'ev'], ['mAP50', 'falso_sano'] + ap_mosca).to_string()]
    for exp, g in F.groupby('experimento'):
        mejor = g.sort_values('valid_mAP50_95', ascending=False).iloc[0]
        dst = CARPETA / f'mejor_{exp}_{mejor.modelo}_s{mejor.semilla}.pt'
        shutil.copy2(mejor.pesos, dst)
        lin.append(f'Mejor por validacion en {exp}: {mejor.modelo} s{mejor.semilla} '
                   f'(valid mAP50-95 {mejor.valid_mAP50_95:.4f}, prueba mAP50 {mejor.test_mAP50:.4f}) -> {dst.name}')
    txt = '\n'.join(lin)
    log('\n' + txt)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / 'resumen_v6.txt').write_text(txt + '\n', encoding='utf-8')
    paquete = CARPETA / f'resultados_{NOMBRE}.zip'
    with zipfile.ZipFile(paquete, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in (CARPETA / 'comparacion_v6.csv', CARPETA / 'estres_v6.csv', RUNS / 'resumen_v6.txt', *CARPETA.glob('mejor_*.pt')):
            if f.exists():
                z.write(f, f.name)
        for f in RUNS.rglob('*'):
            if f.is_file() and f.suffix != '.pt' and 'evaluacion' not in f.parts:
                z.write(f, Path('runs') / f.relative_to(RUNS))
    log(f'Paquete para llevar a la PC: {paquete} ({paquete.stat().st_size / 1e6:.1f} MB)')


def main():
    log(f'Experimentos: {EXPERIMENTOS} | modelos: {MODELOS} | semillas: {SEMILLAS} | epocas: {EPOCHS}'
        + ('  [PRUEBA CORTA]' if a.prueba else ''))
    TRABAJO.mkdir(parents=True, exist_ok=True)
    preparar_dataset()
    quitar_sana()
    verificar()
    escribir_yaml(CARPETA / 'data_fixed.yaml', DATASET)
    if a.solo_verificar:
        return
    if not a.solo_evaluar:
        configurar_gpu()
        fallos = []
        for exp in EXPERIMENTOS:
            data = escribir_yaml(TRABAJO / f'data_{exp}.yaml', dataset_de(exp))
            for semilla in SEMILLAS:
                for modelo in MODELOS:
                    entrenar(exp, data, modelo, semilla, fallos)
        if fallos:
            RUNS.mkdir(parents=True, exist_ok=True)
            (RUNS / 'fallos.json').write_text(json.dumps(fallos, indent=2, ensure_ascii=False))
            log(f'{len(fallos)} corridas fallaron; ver {RUNS / "fallos.json"}')
    resumir(*evaluar())


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        log(f'\nERROR: {e}')
        sys.exit(1)
