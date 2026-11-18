"""A6. mAP de las 4 plagas/enfermedades sin `sana` (apreciacion del revisor, seccion 8B).

Corre model.val de Ultralytics en fijo97, nuevas83 y test180 (etiquetas v5) y guarda AP50 y AP50-95 por
clase. mAP4 = promedio de mildiu, mosca_blanca, oidio y polilla; mAP5 = las 5 clases (como hasta ahora).
Salida: metricas/ap_clase_completo.csv y metricas/map_4_vs_5.csv
"""
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

W = Path(__file__).parent
CL = ['mildiu', 'mosca_blanca', 'oidio', 'polilla', 'sana']
SUB = {'fijo97': W / 'fijo97.yaml', 'nuevas83': W / 'nuevas83.yaml', 'test180': W / 'test180.yaml'}
filas = []
for w in sorted((W / 'pesos').glob('*.pt')):
    m = YOLO(str(w))
    for k, y in SUB.items():
        r = m.val(data=str(y), split='test', imgsz=640, batch=16, device='cpu', plots=False, verbose=False,
                  project=str((W / 'val_runs').resolve()), name=f'{w.stem}_{k}_c', exist_ok=True)
        for i, c in enumerate(r.box.ap_class_index):
            filas.append({'modelo': w.stem, 'subconjunto': k, 'clase': CL[int(c)],
                          'AP50': r.box.ap50[i], 'AP50_95': r.box.ap[i]})
    print('[hecho]', w.stem, flush=True)
A = pd.DataFrame(filas)
A.to_csv(W / 'metricas' / 'ap_clase_completo.csv', index=False)
res = []
for (mo, su), d in A.groupby(['modelo', 'subconjunto']):
    e = d[d.clase != 'sana']
    res.append({'modelo': mo, 'subconjunto': su, 'mAP50_5clases': d.AP50.mean(), 'mAP50_4plagas': e.AP50.mean(),
                'mAP50_95_5clases': d.AP50_95.mean(), 'mAP50_95_4plagas': e.AP50_95.mean(),
                'AP50_sana': d.loc[d.clase == 'sana', 'AP50'].mean()})
pd.DataFrame(res).to_csv(W / 'metricas' / 'map_4_vs_5.csv', index=False)
print('listo')
