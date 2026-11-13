#!/usr/bin/env python3
"""Entrenamiento y comparacion YOLOv8 / YOLO11 / YOLOv10 - plagas y enfermedades del tomate (Majes).

Pensado para la PC con RTX 5070 Ti (16 GB), CachyOS y fish. No necesita API key ni internet para el
dataset: usa el .zip de Roboflow (version 2, formato YOLOv8) que esta en la carpeta de trabajo.

Que hace:
  1. Busca el .zip en la carpeta de trabajo, lo descomprime en ./dataset, ubica data.yaml y escribe
     data_fixed.yaml con rutas absolutas.
  2. Verifica el dataset: orden de clases (el mismo de la app), conteos, que no este estirado y que
     ninguna planta quede en dos subconjuntos. Mueve las 66 imagenes con fuga video-fotos al
     subconjunto correcto (tareas/06_bitacora_dataset.md, seccion 6b).
  3. Entrena YOLOv8, YOLO11 y YOLOv10 uno tras otro con los MISMOS hiperparametros y las mismas
     semillas. Si uno falla, sigue con los demas. Si falta memoria de GPU, reintenta con la mitad
     del batch. Si se corta, al volver a lanzarlo retoma donde quedo.
  4. Evalua el best.pt de cada corrida en test (y valid, cada franja horaria y el test fijo de la
     v4), mide el acierto de la clase principal con las reglas de la app, guarda
     comparacion_modelos.csv, imprime la tabla y copia el mejor como mejor_modelo_<nombre>.pt.

Uso (dentro de ~/PYTHON/entrenamientotomte con el entorno activado):
  python entrenar_yolo.py --solo-verificar      # descomprime y verifica (segundos)
  python entrenar_yolo.py --prueba              # 1 modelo, 1 epoca, 10% de train (minutos)
  python entrenar_yolo.py                       # todo: tamanos n y s, 3 modelos, 3 semillas
  python entrenar_yolo.py --tamanos s --semillas 42
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

# ------------------------------------------------------------------ argumentos
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--carpeta', type=Path, default=Path.cwd(), help='carpeta de trabajo (donde esta el .zip)')
ap.add_argument('--zip', type=Path, default=None, help='zip del dataset (por defecto, el unico .zip de la carpeta)')
ap.add_argument('--tamanos', default='n,s', help="tamanos a entrenar: n, s, m (separados por coma). La app usa n")
ap.add_argument('--semillas', default='0,1,2', help='las mismas semillas para todos los modelos')
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--patience', type=int, default=50)
ap.add_argument('--imgsz', type=int, default=640, help='la app usa 640: no cambiar sin ajustar la app')
ap.add_argument('--batch', type=int, default=32, help='batch fijo, igual para todos (se reduce solo si falta memoria)')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--device', default='0')
ap.add_argument('--esperado', default='1370,404,192', help='imagenes train,valid,test tal como vienen de Roboflow')
ap.add_argument('--nombre', default='v5-1966-completas', help='nombre de esta tanda')
ap.add_argument('--solo-verificar', action='store_true', help='descomprime y verifica, sin entrenar')
ap.add_argument('--solo-evaluar', action='store_true', help='evalua las corridas terminadas, sin entrenar')
ap.add_argument('--prueba', action='store_true', help='prueba corta: 1 modelo, 1 epoca, 10%% de train')
a = ap.parse_args()

CARPETA = a.carpeta.expanduser().resolve()
CLASES = ['mildiu', 'mosca_blanca', 'oidio', 'polilla', 'sana']   # mismo orden que la app y el backend
ARQUITECTURAS = ['yolov8', 'yolo11', 'yolov10']
TAMANOS = [t.strip() for t in a.tamanos.split(',') if t.strip()]
SEMILLAS = [int(s) for s in a.semillas.split(',')]
EPOCHS, FRACCION = a.epochs, 1.0
NOMBRE = a.nombre
if a.prueba:
    ARQUITECTURAS, TAMANOS, SEMILLAS, EPOCHS, FRACCION = ['yolo11'], TAMANOS[:1], SEMILLAS[:1], 1, 0.1
    NOMBRE += '_prueba'
DATASET = CARPETA / 'dataset'
RUNS = CARPETA / 'runs' / NOMBRE                # cada corrida en ./runs/<tanda>/<modelo>_s<semilla>
ESPERADO = dict(zip(('train', 'valid', 'test'), (int(x) for x in a.esperado.split(','))))
DEVICE = a.device if a.device == 'cpu' or ',' in a.device else int(a.device)
EXTS = {'.jpg', '.jpeg', '.png'}

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

# Test fijo = las 102 imagenes de test de la v4: todas las versiones se comparan con estas mismas.
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


# ------------------------------------------------------------------ utilidades de nombres
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
    d = DATASET / split / 'images'
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d.exists() else []


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


# ------------------------------------------------------------------ 1. dataset
def preparar_dataset():
    """Descomprime el zip (si hace falta) y deja el dataset en ./dataset con data.yaml en la raiz."""
    if not (DATASET / 'data.yaml').exists():
        zips = [a.zip.expanduser().resolve()] if a.zip else sorted(CARPETA.glob('*.zip'))
        assert zips, f'No hay ningun .zip en {CARPETA}'
        assert len(zips) == 1, f'Hay varios .zip en {CARPETA}; indica cual con --zip: {[z.name for z in zips]}'
        log(f'Descomprimiendo {zips[0].name} en {DATASET} ...')
        tmp = CARPETA / 'dataset_tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(tmp)
        yamls = sorted(tmp.rglob('data.yaml'), key=lambda p: len(p.parts))
        assert yamls, 'El zip no tiene data.yaml: exportalo de Roboflow en formato YOLOv8'
        shutil.move(str(yamls[0].parent), str(DATASET))
        shutil.rmtree(tmp, ignore_errors=True)
    # Roboflow a veces llama "val" a la carpeta de validacion
    if not (DATASET / 'valid').exists() and (DATASET / 'val').exists():
        (DATASET / 'val').rename(DATASET / 'valid')


def escribir_yaml(nombre, test='test/images'):
    import yaml
    # Rutas absolutas: las relativas de Roboflow fallan segun la carpeta desde donde se ejecuta
    f = CARPETA / f'{nombre}.yaml'
    f.write_text(yaml.safe_dump({
        'path': str(DATASET), 'train': 'train/images', 'val': 'valid/images', 'test': test,
        'names': dict(enumerate(CLASES)),
    }, allow_unicode=True))
    return f


def verificar():
    import yaml
    from PIL import Image

    info = yaml.safe_load((DATASET / 'data.yaml').read_text())
    nombres = list(info['names'].values()) if isinstance(info['names'], dict) else list(info['names'])
    assert nombres == CLASES, f'Orden de clases distinto al de la app: {nombres}'

    conteo_rf = {s: len(imagenes(s)) for s in ('train', 'valid', 'test')}
    log('Imagenes (como vienen de Roboflow):', conteo_rf)
    if conteo_rf != ESPERADO:
        # Si ya se movieron en una ejecucion anterior, se cuentan en su subconjunto de Roboflow
        original = collections.Counter()
        for s in conteo_rf:
            for p in imagenes(s):
                de, hacia = MOVER.get(nombre_base(p), (s, s))
                original[de if s == hacia else s] += 1
        assert dict(original) == ESPERADO, f'Se esperaba {ESPERADO}. ¿Se uso Rebalance o es otra version?'
        log('  (las imagenes con fuga video-fotos ya estaban movidas)')

    movidas = 0
    for s in ('train', 'valid', 'test'):
        for p in imagenes(s):
            destino = MOVER.get(nombre_base(p), (s, s))[1]
            if destino != s:
                for origen, sub in ((p, 'images'), (etiqueta(p), 'labels')):
                    if origen.exists():
                        (DATASET / destino / sub).mkdir(parents=True, exist_ok=True)
                        shutil.move(str(origen), str(DATASET / destino / sub / origen.name))
                movidas += 1
    log(f'Imagenes movidas de subconjunto por fuga video-fotos: {movidas}')

    splits = {s: imagenes(s) for s in ('train', 'valid', 'test')}
    log('Imagenes para entrenar:', {s: len(v) for s, v in splits.items()})

    donde, por_grupo = collections.defaultdict(set), collections.defaultdict(list)
    for s, v in splits.items():
        for p in v:
            donde[grupo(p)].add(s)
            por_grupo[grupo(p)].append(p)
    fuga = {g: sorted(s) for g, s in donde.items() if len(s) > 1}
    assert not fuga, f'Fuga de datos: {len(fuga)} plantas en mas de un subconjunto, ej. {list(fuga.items())[:3]}'
    log(f'Grupos (plantas/videos): {len(donde)}, ninguno repartido entre subconjuntos')

    # Un video grabado a menos de 30 s de una rafaga de fotos es casi seguro la misma planta
    rg = {g: (*rango_grupo(v), next(iter(donde[g]))) for g, v in por_grupo.items()}
    gap = dt.timedelta(seconds=30)
    cerca = [(g, h) for g, (vid, a0, a1, s) in rg.items() if vid
             for h, (vid2, b0, b1, t) in rg.items() if not vid2 and t != s and b0 - gap <= a1 and b1 + gap >= a0]
    assert not cerca, f'Fuga video-fotos: {len(cerca)} pares en distinto subconjunto, ej. {cerca[:3]}'
    log('Videos y rafagas cercanas: todos en el mismo subconjunto')

    muestra = splits['train'][:: max(1, len(splits['train']) // 50)]
    cuadradas = sum(Image.open(p).width == Image.open(p).height for p in muestra)
    assert cuadradas < len(muestra) / 2, 'Imagenes cuadradas: la version parece hecha con Stretch. Usa Fit within.'

    log('Cajas por clase:')
    for s, v in splits.items():
        c = collections.Counter(k for p in v for k in clases_en(p))
        log(f'  {s:6}', {CLASES[k]: c[k] for k in range(len(CLASES))})
    log('Imagenes de test por franja:', dict(collections.Counter(periodo(p) for p in splits['test'])))
    log(f'Test fijo: {len(FIJOS)} imagenes, {sum(nombre_base(p) in FIJOS for p in splits["test"])} presentes en test')
    return splits


def elegir_cache():
    # cache en RAM si el dataset entra holgado (32 GB de RAM); si no, en disco
    gb = sum(f.stat().st_size for f in DATASET.rglob('*') if f.is_file()) / 1e9
    modo = 'ram' if gb < 12 else 'disk'
    log(f'Dataset en disco: {gb:.1f} GB -> cache={modo}')
    return modo


# ------------------------------------------------------------------ 2. entrenamiento
def configurar_gpu():
    import torch
    if torch.cuda.is_available():
        # TF32 en las matrices y convoluciones: mas rapido en RTX 50xx, sin perdida apreciable
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}')
    else:
        log('AVISO: PyTorch no ve la GPU; se entrenaria en CPU (muy lento).')


def liberar():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def es_falta_de_memoria(e):
    return 'out of memory' in str(e).lower() or type(e).__name__ == 'OutOfMemoryError'


def entrenar(data, cache, modelo, semilla, fallos):
    from ultralytics import YOLO

    d = RUNS / f'{modelo}_s{semilla}'
    if (d / 'terminado.json').exists():
        log(f'[listo] {d.name}')
        return
    batch, t0 = a.batch, time.time()
    while True:
        ultimo = d / 'weights' / 'last.pt'
        try:
            if ultimo.exists():
                log(f'[continua] {d.name}')
                YOLO(str(ultimo)).train(resume=True)
            else:
                log(f'[entrena] {d.name}  batch={batch}')
                YOLO(f'{modelo}.pt').train(
                    data=str(data), epochs=EPOCHS, patience=a.patience, imgsz=a.imgsz, batch=batch,
                    device=DEVICE, workers=a.workers, cache=cache, amp=True, seed=semilla,
                    deterministic=False, cos_lr=True, close_mosaic=15, fraction=FRACCION, plots=True,
                    project=str(RUNS), name=d.name, exist_ok=True,
                )
            break
        except AssertionError as e:
            # Ultralytics no reanuda una corrida que ya llego al final: se da por terminada
            if 'finished' in str(e) or 'nothing to resume' in str(e):
                break
            raise
        except Exception as e:
            liberar()
            if es_falta_de_memoria(e) and batch > 2:
                batch //= 2
                log(f'  Falta memoria de GPU: se reintenta {d.name} desde cero con batch={batch}')
                shutil.rmtree(d, ignore_errors=True)
                continue
            log(f'  FALLO {d.name}: {type(e).__name__}: {e}  (se sigue con los demas)')
            fallos.append({'corrida': d.name, 'error': f'{type(e).__name__}: {e}', 'detalle': traceback.format_exc()})
            return
    (d / 'terminado.json').write_text(json.dumps({
        'modelo': modelo, 'semilla': semilla, 'batch': batch, 'minutos': (time.time() - t0) / 60,
    }))
    liberar()


# ------------------------------------------------------------------ 3. evaluacion
def wilson(ok, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = ok / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - m, c + m)


def subconjunto(nombre, rutas):
    # test = lista de imagenes: asi se evalua cada franja por separado
    lista = CARPETA / f'{nombre}.txt'
    lista.write_text('\n'.join(str(p) for p in rutas) + '\n')
    return escribir_yaml(nombre, test=str(lista))


def evaluar(splits, data):
    import pandas as pd
    import torch
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    sana = CLASES.index('sana')
    disp = DEVICE if DEVICE == 'cpu' or torch.cuda.is_available() else 'cpu'
    split_final = 'test' if splits['test'] else 'val'
    pruebas = {'valid': (data, 'val'), 'test': (data, split_final)}
    for f in ('manana', 'mediodia', 'tarde'):
        rutas = [p for p in splits['test'] if periodo(p) == f]
        if rutas:
            pruebas[f'test_{f}'] = (subconjunto(f'test_{f}', rutas), 'test')
    fijos = [p for p in splits['test'] if nombre_base(p) in FIJOS]
    if fijos:
        pruebas['test_fijo'] = (subconjunto('test_fijo', fijos), 'test')
    rutas_acierto = {'test': splits['test'], 'test_fijo': fijos,
                     **{f'test_{f}': [p for p in splits['test'] if periodo(p) == f] for f in ('manana', 'mediodia', 'tarde')}}

    def acierto_principal(model):
        # Misma regla que la app: deteccion de mayor confianza; sin detecciones cuenta como sana
        pred = {}
        for p in splits['test']:
            if not clases_en(p):
                continue
            im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
            b = model(im, conf=0.30, iou=0.45, imgsz=a.imgsz, agnostic_nms=True, device=disp, verbose=False)[0].boxes
            top = int(b.cls[b.conf.argmax()]) if len(b) else sana
            pred[p] = top == collections.Counter(clases_en(p)).most_common(1)[0][0]
        filas = []
        for nombre, rutas in rutas_acierto.items():
            r = [pred[p] for p in rutas if p in pred]
            lo, hi = wilson(sum(r), len(r))
            filas.append({'prueba': nombre, 'imagenes': len(r), 'acierto': sum(r) / len(r) if r else float('nan'),
                          'ic95_bajo': lo, 'ic95_alto': hi})
        return filas

    general, por_clase, acierto, comparacion = [], [], [], []
    corridas = sorted(d for d in RUNS.iterdir() if (d / 'terminado.json').exists()) if RUNS.exists() else []
    assert corridas, 'Todavia no hay corridas terminadas'
    for d in corridas:
        log(f'[evalua] {d.name}')
        info = json.loads((d / 'terminado.json').read_text())
        modelo, sem = d.name.rsplit('_s', 1)
        pesos = d / 'weights' / 'best.pt'
        model = YOLO(str(pesos))
        base = {'modelo': modelo, 'semilla': int(sem), 'corrida': d.name}
        res = {}
        for nombre, (dat, split) in pruebas.items():
            m = model.val(data=str(dat), split=split, imgsz=a.imgsz, batch=16, device=disp, plots=True,
                          project=str(d / 'evaluacion'), name=nombre, exist_ok=True, verbose=False)
            res[nombre] = m
            general.append({**base, 'prueba': nombre, 'mAP50': m.box.map50, 'mAP50_95': m.box.map,
                            'precision': m.box.mp, 'recall': m.box.mr})
            for i, k in enumerate(m.box.ap_class_index):
                por_clase.append({**base, 'prueba': nombre, 'clase': CLASES[int(k)], 'precision': m.box.p[i],
                                  'recall': m.box.r[i], 'AP50': m.box.ap50[i], 'AP50_95': m.box.ap[i]})
        ac = acierto_principal(model)
        acierto += [{**base, **f} for f in ac]
        t, v = res['test'], res['valid']
        comparacion.append({
            **base, 'evaluado_en': split_final, 'precision': t.box.mp, 'recall': t.box.mr, 'mAP50': t.box.map50,
            'mAP50_95': t.box.map, 'valid_mAP50_95': v.box.map,
            'acierto_clase_principal': next(f['acierto'] for f in ac if f['prueba'] == 'test'),
            'ms_por_imagen': t.speed['inference'], 'minutos_entrenamiento': info.get('minutos', float('nan')),
            'batch': info.get('batch', a.batch), 'pesos': str(pesos),
        })
        del model
        liberar()

    tablas = {n: pd.DataFrame(x) for n, x in
              (('generales', general), ('por_clase', por_clase), ('acierto', acierto), ('comparacion', comparacion))}
    tablas['comparacion'].to_csv(CARPETA / 'comparacion_modelos.csv', index=False)
    tablas['generales'].to_csv(RUNS / 'metricas_generales.csv', index=False)
    tablas['por_clase'].to_csv(RUNS / 'metricas_por_clase.csv', index=False)
    tablas['acierto'].to_csv(RUNS / 'acierto_clase_principal.csv', index=False)
    return pruebas, tablas


# ------------------------------------------------------------------ 4. resumen y mejor modelo
def resumir(pruebas, tablas):
    import pandas as pd
    comp = tablas['comparacion']

    def resumen(df, por, cols):
        g = df.groupby(por)[cols]
        r = g.mean().round(4).astype(str) + ' ± ' + g.std().fillna(0).round(4).astype(str)
        r.insert(0, 'corridas', df.groupby(por).size())
        return r

    lineas = ['=== Comparacion (media ± desviacion entre semillas) ===']
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        lineas.append(resumen(comp, 'modelo', ['precision', 'recall', 'mAP50', 'mAP50_95', 'acierto_clase_principal',
                                               'ms_por_imagen', 'minutos_entrenamiento']).to_string())
        g = tablas['generales']
        for prueba in pruebas:
            lineas += ['', f'--- mAP por prueba: {prueba} ---',
                       resumen(g[g.prueba == prueba], 'modelo', ['mAP50', 'mAP50_95', 'precision', 'recall']).to_string()]
        pc = tablas['por_clase']
        lineas += ['', '--- Test por clase ---',
                   resumen(pc[pc.prueba == 'test'], ['modelo', 'clase'], ['precision', 'recall', 'AP50']).to_string()]
        lineas += ['', '--- Acierto de la clase principal (reglas de la app) ---',
                   resumen(tablas['acierto'], ['modelo', 'prueba'], ['acierto']).to_string()]

    # El mejor se elige con VALIDACION (no con test, para que el test siga siendo una medida honesta)
    arq = comp.groupby('modelo').valid_mAP50_95.mean().idxmax()
    mejor = comp[comp.modelo == arq].sort_values('valid_mAP50_95', ascending=False).iloc[0]
    destino = CARPETA / f'mejor_modelo_{mejor.corrida}.pt'
    shutil.copy2(mejor.pesos, destino)
    lineas += ['', f'Mejor por validacion: {arq} (corrida {mejor.corrida}, mAP50-95 valid = {mejor.valid_mAP50_95:.4f},'
               f' test mAP50 = {mejor.mAP50:.4f})', f'Copiado en {destino}']
    if arq.startswith('yolov10'):
        lineas.append('Aviso: la salida de YOLOv10 es distinta a la de v8/v11; la app necesitara un ajuste.')
    texto = '\n'.join(lineas)
    log('\n' + texto)
    (RUNS / 'resumen.txt').write_text(texto + '\n', encoding='utf-8')

    paquete = CARPETA / f'resultados_{NOMBRE}.zip'
    with zipfile.ZipFile(paquete, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(CARPETA / 'comparacion_modelos.csv', 'comparacion_modelos.csv')
        z.write(destino, destino.name)
        for f in RUNS.rglob('*'):
            # de cada corrida, curvas y resultados; de los pesos solo el mejor (ya agregado)
            if f.is_file() and f.suffix != '.pt':
                z.write(f, Path('runs') / f.relative_to(RUNS))
    log(f'Paquete para llevar a la PC: {paquete} ({paquete.stat().st_size / 1e6:.1f} MB)')


def main():
    log(f'Carpeta: {CARPETA}\nModelos: {[f"{x}{t}" for t in TAMANOS for x in ARQUITECTURAS]}  semillas: {SEMILLAS}'
        f'  epocas: {EPOCHS}  patience: {a.patience}  imgsz: {a.imgsz}  batch: {a.batch}'
        + ('  [PRUEBA CORTA]' if a.prueba else ''))
    preparar_dataset()
    splits = verificar()
    data = escribir_yaml('data_fixed')
    log(f'data_fixed.yaml: {data}')
    if a.solo_verificar:
        return
    if not a.solo_evaluar:
        configurar_gpu()
        cache = elegir_cache()
        fallos = []
        # Primero se completa la semilla 0 de todos los modelos; asi hay comparacion aunque se corte
        for semilla in SEMILLAS:
            for t in TAMANOS:
                for arq in ARQUITECTURAS:
                    entrenar(data, cache, f'{arq}{t}', semilla, fallos)
        if fallos:
            RUNS.mkdir(parents=True, exist_ok=True)
            (RUNS / 'fallos.json').write_text(json.dumps(fallos, indent=2, ensure_ascii=False))
            log(f'{len(fallos)} corridas fallaron; detalle en {RUNS / "fallos.json"}')
        log('Entrenamiento terminado')
    resumir(*evaluar(splits, data))


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        log(f'\nERROR: {e}')
        sys.exit(1)
