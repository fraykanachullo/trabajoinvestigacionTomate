"""A8. Prueba de estres: sobreexposicion sintetica de las 180 imagenes de prueba.

Para cada nivel de exposicion (pasos EV), la imagen se lleva a luz lineal (sRGB -> lineal), se multiplica por
2^EV, se recorta a 1 (pixeles quemados, como en una foto sobreexpuesta real) y se vuelve a sRGB. EV < 0 simula
subexposicion. Las etiquetas no cambian. Se evaluan los 9 modelos v5 y el v4 con Ultralytics val
(mAP con 5 clases y sin hoja sana) y con las reglas de la app (falso sano).
Salida: estres/<nivel>/images|labels, metricas/estres_exposicion.csv
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps
from ultralytics import YOLO

W = Path(__file__).parent
SRC = W / 'dataset' / 'test'
OUT = W / 'estres'
CL = ['mildiu', 'mosca_blanca', 'oidio', 'polilla', 'sana']
NIVELES = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]


def a_lineal(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def a_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


imgs = sorted((SRC / 'images').iterdir())
satur = {}
for ev in NIVELES:
    d = OUT / f'ev{ev:+.1f}'
    (d / 'images').mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC / 'labels', d / 'labels', dirs_exist_ok=True)
    s = []
    for p in imgs:
        q = d / 'images' / p.name
        a = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert('RGB'), np.float32) / 255
        b = np.clip(a_srgb(np.clip(a_lineal(a) * 2 ** ev, 0, 1)), 0, 1)
        s.append(100 * (b >= 250 / 255).any(-1).mean())
        if not q.exists():
            Image.fromarray((b * 255 + 0.5).astype(np.uint8)).save(q, quality=95)
    satur[ev] = float(np.median(s))
    (d / 'data.yaml').write_text(yaml.safe_dump({'path': str(d.resolve()), 'train': 'images', 'val': 'images',
                                                 'test': 'images', 'names': dict(enumerate(CL))}))
    print('nivel', ev, 'saturados mediana', round(satur[ev], 2), flush=True)

filas = []
for w in sorted((W / 'pesos').glob('*.pt')):
    m = YOLO(str(w))
    for ev in NIVELES:
        d = OUT / f'ev{ev:+.1f}'
        r = m.val(data=str(d / 'data.yaml'), split='test', imgsz=640, batch=16, device='cpu', plots=False,
                  verbose=False, project=str((W / 'val_runs').resolve()), name=f'{w.stem}_ev{ev:+.1f}', exist_ok=True)
        ap = {CL[int(c)]: r.box.ap50[i] for i, c in enumerate(r.box.ap_class_index)}
        # falso sano con las reglas de la app
        fs = n = 0
        for p in sorted((d / 'images').iterdir()):
            lbl = d / 'labels' / (p.stem + '.txt')
            cl = [int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            if not set(cl) & {0, 1, 2, 3}:
                continue
            b = m(str(p), conf=0.30, iou=0.45, imgsz=640, agnostic_nms=True, verbose=False, device='cpu')[0].boxes
            n += 1
            fs += int(not (set(b.cls.int().tolist()) & {0, 1, 2, 3}))
        filas.append({'modelo': w.stem, 'ev': ev, 'saturados_mediana': satur[ev], 'mAP50_5': r.box.map50,
                      'mAP50_4': np.mean([v for k, v in ap.items() if k != 'sana']), 'falso_sano': fs / n,
                      **{f'AP50_{k}': v for k, v in ap.items()}})
    print('[hecho]', w.stem, flush=True)
pd.DataFrame(filas).to_csv(W / 'metricas' / 'estres_exposicion.csv', index=False)
print('listo')
