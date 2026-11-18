"""A2b. mAP oficial de Ultralytics (model.val) para los 10 modelos en fijo97 y test180, con etiquetas v5."""
import re
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

W = Path(__file__).parent
D = W / 'dataset'
CL = ['mildiu', 'mosca_blanca', 'oidio', 'polilla', 'sana']
fijos = set((W / 'test_fijo_v4.txt').read_text().split())


def nombre_base(n):
    n = re.sub(r'\.(jpg|jpeg|png|txt)$', '', Path(n).name, flags=re.I)
    return re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', n, flags=re.I)


imgs = sorted((D / 'test' / 'images').iterdir())
listas = {'fijo97': [p for p in imgs if nombre_base(p) in fijos], 'test180': imgs}
cfg = {}
for k, v in listas.items():
    (W / f'{k}.txt').write_text('\n'.join(str(p) for p in v) + '\n')
    y = W / f'{k}.yaml'
    y.write_text(yaml.safe_dump({'path': str(D), 'train': 'train/images', 'val': 'valid/images',
                                 'test': str(W / f'{k}.txt'), 'names': dict(enumerate(CL))}))
    cfg[k] = y
filas, clases = [], []
for w in sorted((W / 'pesos').glob('*.pt')):
    m = YOLO(str(w))
    for k, y in cfg.items():
        r = m.val(data=str(y), split='test', imgsz=640, batch=16, device='cpu', plots=False, verbose=False,
                  project=str(W / 'val_runs'), name=f'{w.stem}_{k}', exist_ok=True)
        filas.append({'modelo': w.stem, 'subconjunto': k, 'imagenes': len(listas[k]), 'mAP50': r.box.map50,
                      'mAP50_95': r.box.map, 'precision': r.box.mp, 'recall': r.box.mr})
        for i, c in enumerate(r.box.ap_class_index):
            clases.append({'modelo': w.stem, 'subconjunto': k, 'clase': CL[int(c)], 'AP50': r.box.ap50[i],
                           'precision': r.box.p[i], 'recall': r.box.r[i]})
    print('[hecho]', w.stem, flush=True)
pd.DataFrame(filas).to_csv(W / 'metricas' / 'map_ultralytics.csv', index=False)
pd.DataFrame(clases).to_csv(W / 'metricas' / 'ap_clase_ultralytics.csv', index=False)
