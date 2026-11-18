#!/usr/bin/env python3
"""Experimentos de control para responder al revisor (obs. 2 y 3). Tomate Majes, 4 clases.

Cada experimento cambia UNA sola cosa respecto de la referencia; mismas etiquetas (version 3 de Roboflow),
mismo modelo, mismos hiperparametros y semillas:

  grupo           : division por grupo de captura (referencia, igual a la v6 base)
  aleatorio_pool  : entrenamiento y validacion repartidos AL AZAR POR IMAGEN (con fuga), misma prueba por
                    grupo (sin fuga)  -> cuanto se infla la VALIDACION y si la prueba honesta cambia (obs. 2)
  aleatorio_total : todo repartido al azar 70/20/10 por imagen, como se hace habitualmente; se mide en su
                    propia prueba aleatoria -> cuanto se infla la cifra que se REPORTARIA (obs. 2)
  solo_mayo       : entrena solo con las fotos del 3 de mayo (sin las sesiones de julio), misma prueba
                    -> efecto de agregar sesiones nuevas, con todo lo demas igual (obs. 3)
  mixto_reducido  : mismo numero de imagenes que solo_mayo pero con grupos de todas las sesiones
                    -> separa "mas sesiones" de "mas imagenes" (obs. 3)

Para cada experimento informa cuanta fuga tiene (fraccion de imagenes de prueba con fotos del mismo grupo
de captura en entrenamiento) y evalua: validacion propia, prueba por grupo (180), prueba fija (97, mayo),
imagenes nuevas (83, julio) y, en aleatorio_total, su prueba aleatoria. Guarda control.csv, resumen y los
mejores pesos de cada corrida (para calcular IC por bootstrap despues).

Reutiliza ./dataset_v6 si ya lo preparo entrenar_yolo_v6.py; si no, descomprime el zip.

Uso:
  python entrenar_control.py --solo-verificar --zip plagas-tomate-majes-v2.v3-v6-2026-11-17.yolov8.zip
  python entrenar_control.py --prueba --zip ...
  python entrenar_control.py --zip ...                      # 5 experimentos x yolov8n x 3 semillas
"""
import argparse
import collections
import datetime as dt
import gc
import json
import os
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
ap.add_argument('--experimentos', default='grupo,aleatorio_pool,aleatorio_total,solo_mayo,mixto_reducido')
ap.add_argument('--modelos', default='yolov8n')
ap.add_argument('--semillas', default='0,1,2')
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--patience', type=int, default=50)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--device', default='0')
ap.add_argument('--esperado', default='1370,404,192')
ap.add_argument('--nombre', default='control')
ap.add_argument('--solo-verificar', action='store_true')
ap.add_argument('--solo-evaluar', action='store_true')
ap.add_argument('--prueba', action='store_true')
a = ap.parse_args()

CARPETA = a.carpeta.expanduser().resolve()
CLASES = ['mildiu', 'mosca_blanca', 'oidio', 'polilla']
EXPERIMENTOS = [e.strip() for e in a.experimentos.split(',') if e.strip()]
MODELOS = [m.strip() for m in a.modelos.split(',') if m.strip()]
SEMILLAS = [int(s) for s in a.semillas.split(',')]
EPOCHS, FRACCION, NOMBRE = a.epochs, 1.0, a.nombre
if a.prueba:
    MODELOS, SEMILLAS, EPOCHS, FRACCION, NOMBRE = MODELOS[:1], SEMILLAS[:1], 1, 0.1, NOMBRE + '_prueba'
DATASET = CARPETA / 'dataset_v6'
TRABAJO = CARPETA / 'trabajo_control'
RUNS = CARPETA / 'runs' / NOMBRE
ESPERADO = dict(zip(('train', 'valid', 'test'), (int(x) for x in a.esperado.split(','))))
DEVICE = a.device if a.device == 'cpu' or ',' in a.device else int(a.device)
EXTS = {'.jpg', '.jpeg', '.png'}
SEMILLA_DIVISION = 2026

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


def sesion(p):
    return re.findall(r'(\d{8})_\d{6}', nombre_base(p))[-1]


def imagenes(d, split):
    q = d / split / 'images'
    return sorted(p for p in q.iterdir() if p.suffix.lower() in EXTS) if q.exists() else []


def etiqueta(p):
    return p.parent.parent / 'labels' / (p.stem + '.txt')


# ------------------------------------------------------------------ dataset (igual que entrenar_yolo_v6.py)
def preparar_dataset():
    if not (DATASET / 'data.yaml').exists():
        zips = [a.zip.expanduser().resolve()] if a.zip else sorted(CARPETA.glob('*.zip'))
        zips = [z for z in zips if not z.name.startswith('resultados_')]
        assert len(zips) == 1, f'Indica el zip del dataset con --zip: {[z.name for z in zips]}'
        log(f'Descomprimiendo {zips[0].name} ...')
        tmp = CARPETA / 'dataset_v6_tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(tmp)
        y = sorted(tmp.rglob('data.yaml'), key=lambda p: len(p.parts))
        shutil.move(str(y[0].parent), str(DATASET))
        shutil.rmtree(tmp, ignore_errors=True)
    import yaml
    info = yaml.safe_load((DATASET / 'data.yaml').read_text())
    nombres = list(info['names'].values()) if isinstance(info['names'], dict) else list(info['names'])
    if nombres != CLASES:
        assert set(nombres) - {'sana'} == set(CLASES), f'Clases inesperadas: {nombres}'
        nuevo = {i: CLASES.index(n) for i, n in enumerate(nombres) if n != 'sana'}
        for lbl in DATASET.rglob('labels/*.txt'):
            out = [' '.join([str(nuevo[int(v[0])])] + v[1:]) for v in (l.split() for l in lbl.read_text().splitlines()) if v and int(v[0]) in nuevo]
            lbl.write_text('\n'.join(out) + ('\n' if out else ''))
        info['names'], info['nc'] = CLASES, 4
        (DATASET / 'data.yaml').write_text(yaml.safe_dump(info))
        log('Clase `sana` quitada')
    conteo = {s: len(imagenes(DATASET, s)) for s in ('train', 'valid', 'test')}
    if conteo != ESPERADO:
        original = collections.Counter()
        for s in conteo:
            for p in imagenes(DATASET, s):
                de, hacia = MOVER.get(nombre_base(p), (s, s))
                original[de if s == hacia else s] += 1
        assert dict(original) == ESPERADO, f'Se esperaba {ESPERADO}; ajusta --esperado'
    for s in ('train', 'valid', 'test'):
        for p in imagenes(DATASET, s):
            destino = MOVER.get(nombre_base(p), (s, s))[1]
            if destino != s:
                for origen, sub in ((p, 'images'), (etiqueta(p), 'labels')):
                    if origen.exists():
                        (DATASET / destino / sub).mkdir(parents=True, exist_ok=True)
                        shutil.move(str(origen), str(DATASET / destino / sub / origen.name))
    splits = {s: imagenes(DATASET, s) for s in ('train', 'valid', 'test')}
    donde = collections.defaultdict(set)
    for s, v in splits.items():
        for p in v:
            donde[grupo(p)].add(s)
    assert not [g for g, s in donde.items() if len(s) > 1], 'La division por grupo tiene fuga: revisa el dataset'
    log('Division por grupo:', {s: len(v) for s, v in splits.items()}, '- sin fuga')
    return splits


# ------------------------------------------------------------------ armado de cada experimento
def enlazar(p, dst_split_dir):
    for src, sub in ((p, 'images'), (etiqueta(p), 'labels')):
        if not src.exists():
            continue
        q = dst_split_dir / sub / src.name
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            try:
                os.link(src, q)
            except OSError:
                shutil.copy2(src, q)


def armar(exp, base):
    """Devuelve {'train': [...], 'valid': [...], 'test': [...]} de rutas del dataset_v6 para el experimento."""
    rng = np.random.default_rng(SEMILLA_DIVISION)
    tr, va, te = base['train'], base['valid'], base['test']
    if exp == 'grupo':
        return {'train': tr, 'valid': va, 'test': te}
    if exp == 'aleatorio_pool':
        pool = list(tr) + list(va)
        idx = rng.permutation(len(pool))
        return {'train': [pool[i] for i in idx[len(va):]], 'valid': [pool[i] for i in idx[:len(va)]], 'test': te}
    if exp == 'aleatorio_total':
        todo = list(tr) + list(va) + list(te)
        idx = rng.permutation(len(todo))
        nva, nte = len(va), len(te)
        return {'train': [todo[i] for i in idx[nva + nte:]], 'valid': [todo[i] for i in idx[:nva]],
                'test': [todo[i] for i in idx[nva:nva + nte]]}
    if exp == 'solo_mayo':
        return {'train': [p for p in tr if sesion(p) == '20260503'], 'valid': [p for p in va if sesion(p) == '20260503'], 'test': te}
    if exp == 'mixto_reducido':
        # Mismo numero de imagenes que solo_mayo, pero con grupos de TODAS las sesiones (se eligen grupos
        # completos al azar): separa el efecto de "mas sesiones" del de "mas imagenes".
        def reducir(lista, objetivo):
            grupos = collections.defaultdict(list)
            for p in lista:
                grupos[grupo(p)].append(p)
            claves = sorted(grupos)
            out = []
            for i in rng.permutation(len(claves)):
                if len(out) >= objetivo:
                    break
                out += grupos[claves[i]]
            return out
        return {'train': reducir(tr, sum(sesion(p) == '20260503' for p in tr)),
                'valid': reducir(va, sum(sesion(p) == '20260503' for p in va)), 'test': te}
    raise ValueError(f'Experimento desconocido: {exp}')


def fuga(train, test):
    g = {grupo(p) for p in train}
    return sum(grupo(p) in g for p in test) / max(len(test), 1)


def construir(exp, base):
    import yaml
    sp = armar(exp, base)
    d = TRABAJO / exp
    if not (d / '.listo').exists():
        shutil.rmtree(d, ignore_errors=True)
        for s, rutas in sp.items():
            for p in rutas:
                enlazar(p, d / s)
        (d / '.listo').write_text('ok')
    y = d / 'data.yaml'
    y.write_text(yaml.safe_dump({'path': str(d), 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
                                 'names': dict(enumerate(CLASES))}))
    info = {'experimento': exp, **{f'n_{s}': len(v) for s, v in sp.items()},
            'fuga_valid': fuga(sp['train'], sp['valid']), 'fuga_test': fuga(sp['train'], sp['test'])}
    log(f"  {exp:16} train {info['n_train']:5} valid {info['n_valid']:4} test {info['n_test']:4} | "
        f"imagenes de valid con su grupo en train: {info['fuga_valid']:.0%} | de test: {info['fuga_test']:.0%}")
    return d, y, info


# ------------------------------------------------------------------ entrenamiento y evaluacion
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
    while True:
        try:
            if (d / 'weights' / 'last.pt').exists():
                log(f'[continua] {exp}/{d.name}')
                YOLO(str(d / 'weights' / 'last.pt')).train(resume=True)
            else:
                log(f'[entrena] {exp}/{d.name} batch={batch}')
                YOLO(f'{modelo}.pt').train(
                    data=str(data), epochs=EPOCHS, patience=a.patience, imgsz=a.imgsz, batch=batch, device=DEVICE,
                    workers=a.workers, cache='ram', amp=True, seed=semilla, deterministic=False, cos_lr=True,
                    close_mosaic=15, fraction=FRACCION, plots=True, project=str(RUNS / exp), name=d.name, exist_ok=True)
            break
        except AssertionError as e:
            if 'finished' in str(e) or 'nothing to resume' in str(e):
                break
            raise
        except Exception as e:
            liberar()
            if ('out of memory' in str(e).lower() or type(e).__name__ == 'OutOfMemoryError') and batch > 2:
                batch //= 2
                shutil.rmtree(d, ignore_errors=True)
                log(f'  Falta memoria: reintento con batch={batch}')
                continue
            log(f'  FALLO {exp}/{d.name}: {e}')
            fallos.append({'corrida': f'{exp}/{d.name}', 'error': str(e), 'detalle': traceback.format_exc()})
            return
    (d / 'terminado.json').write_text(json.dumps({'experimento': exp, 'modelo': modelo, 'semilla': semilla,
                                                  'batch': batch, 'minutos': (time.time() - t0) / 60}))
    liberar()


def yaml_lista(nombre, d, rutas):
    import yaml
    lista = TRABAJO / f'{nombre}.txt'
    lista.write_text('\n'.join(str(p) for p in rutas) + '\n')
    y = TRABAJO / f'{nombre}.yaml'
    y.write_text(yaml.safe_dump({'path': str(d), 'train': 'train/images', 'val': 'valid/images', 'test': str(lista),
                                 'names': dict(enumerate(CLASES))}))
    return y


def evaluar(construidos, base):
    import pandas as pd
    import torch
    from ultralytics import YOLO
    disp = DEVICE if DEVICE == 'cpu' or torch.cuda.is_available() else 'cpu'
    te = base['test']
    honestas = {'prueba_grupo': yaml_lista('prueba_grupo', DATASET, te),
                'prueba_fija': yaml_lista('prueba_fija', DATASET, [p for p in te if nombre_base(p) in FIJOS]),
                'nuevas': yaml_lista('nuevas', DATASET, [p for p in te if nombre_base(p) not in FIJOS])}
    filas = []
    for exp, (d, y, info) in construidos.items():
        corridas = sorted(c for c in (RUNS / exp).iterdir() if (c / 'terminado.json').exists()) if (RUNS / exp).exists() else []
        for c in corridas:
            log(f'[evalua] {exp}/{c.name}')
            t = json.loads((c / 'terminado.json').read_text())
            model = YOLO(str(c / 'weights' / 'best.pt'))
            fila = {**info, 'modelo': t['modelo'], 'semilla': t['semilla'], 'minutos': t.get('minutos')}
            pruebas = {'valid_propia': (y, 'val')}
            if exp == 'aleatorio_total':
                pruebas['prueba_aleatoria_propia'] = (y, 'test')   # las de grupo se filtraron a su train: no se usan
            else:
                pruebas.update({k: (v, 'test') for k, v in honestas.items()})
            for nombre, (yy, split) in pruebas.items():
                r = model.val(data=str(yy), split=split, imgsz=a.imgsz, batch=16, device=disp, plots=False, verbose=False,
                              project=str(c / 'evaluacion'), name=nombre, exist_ok=True)
                fila[f'{nombre}_mAP50'] = r.box.map50
                fila[f'{nombre}_mAP50_95'] = r.box.map
            filas.append(fila)
            del model
            liberar()
    F = pd.DataFrame(filas)
    F.to_csv(CARPETA / 'control.csv', index=False)
    return F


def resumir(F):
    import pandas as pd
    if F.empty:
        log('No hay corridas terminadas')
        return
    cols = [c for c in F.columns if c.endswith('_mAP50')]
    g = F.groupby('experimento')
    r = (g[cols].mean() * 100).round(1).astype(str) + ' ± ' + (g[cols].std().fillna(0) * 100).round(1).astype(str)
    r.insert(0, 'fuga_test', (g.fuga_test.first() * 100).round(0).astype(str) + ' %')
    r.insert(0, 'fuga_valid', (g.fuga_valid.first() * 100).round(0).astype(str) + ' %')
    r.insert(0, 'n_train', g.n_train.first())
    r.insert(0, 'corridas', g.size())
    with pd.option_context('display.width', 240, 'display.max_columns', 30):
        txt = '=== Experimentos de control: mAP@0.5 (%) media ± DE entre semillas ===\n' + r.to_string()
    txt += ('\n\nLectura: aleatorio_pool frente a grupo -> inflacion de la validacion por fuga (misma prueba honesta);'
            '\naleatorio_total frente a grupo -> inflacion de la cifra reportada en una prueba con fuga;'
            '\nsolo_mayo frente a grupo -> efecto de agregar las sesiones de julio con todo lo demas igual;'
            '\nsolo_mayo frente a mixto_reducido -> efecto de la diversidad de sesiones con casi el mismo numero de imagenes.')
    log('\n' + txt)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / 'resumen_control.txt').write_text(txt + '\n', encoding='utf-8')
    paquete = CARPETA / f'resultados_{NOMBRE}.zip'
    with zipfile.ZipFile(paquete, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in (CARPETA / 'control.csv', RUNS / 'resumen_control.txt'):
            z.write(f, f.name)
        for exp in EXPERIMENTOS:
            sp = TRABAJO / exp
            for s in ('train', 'valid', 'test'):
                q = sp / s / 'images'
                if q.exists():
                    z.writestr(f'particiones/{exp}_{s}.txt', '\n'.join(nombre_base(p) for p in sorted(q.iterdir())) + '\n')
        for f in RUNS.rglob('*'):
            if f.is_file() and 'evaluacion' not in f.parts and (f.suffix != '.pt' or f.name == 'best.pt'):
                z.write(f, Path('runs') / f.relative_to(RUNS))
    log(f'Paquete para llevar a la PC: {paquete} ({paquete.stat().st_size / 1e6:.1f} MB)')


def main():
    log(f'Experimentos: {EXPERIMENTOS} | modelos: {MODELOS} | semillas: {SEMILLAS} | epocas: {EPOCHS}'
        + ('  [PRUEBA CORTA]' if a.prueba else ''))
    TRABAJO.mkdir(parents=True, exist_ok=True)
    base = preparar_dataset()
    log('Particiones de cada experimento:')
    construidos = {exp: construir(exp, base) for exp in EXPERIMENTOS}
    if a.solo_verificar:
        return
    if not a.solo_evaluar:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        fallos = []
        for semilla in SEMILLAS:
            for exp, (d, y, info) in construidos.items():
                for modelo in MODELOS:
                    entrenar(exp, y, modelo, semilla, fallos)
        if fallos:
            RUNS.mkdir(parents=True, exist_ok=True)
            (RUNS / 'fallos.json').write_text(json.dumps(fallos, indent=2, ensure_ascii=False))
    resumir(evaluar(construidos, base))


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        log(f'\nERROR: {e}')
        sys.exit(1)
