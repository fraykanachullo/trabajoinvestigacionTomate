"""A2. Metricas con incertidumbre a partir de las predicciones por imagen (a1).

1. mAP@0.5 y mAP@0.5:0.95 (interpolacion de 101 puntos, emparejamiento por imagen como COCO) en:
     fijo97  = las 97 imagenes del test fijo de la v4 que siguen en test, con etiquetas v5 (las mismas para todos)
     test180 = la prueba completa de la v5
   con IC 95 % por bootstrap agrupado por grupo de captura (1000 replicas).
2. Comparacion pareada v4 (YOLO11n s1) frente a v5 en fijo97: diferencia de mAP con IC bootstrap pareado.
3. Metricas por imagen con las reglas de la app (conf 0.30, IoU 0.45, NMS agnostica):
     acierto de la clase principal (regla actual), deteccion de "hay plaga/enfermedad",
     sensibilidad y especificidad por clase, tasa de falso sano, lineas base, McNemar v4 vs v5.
Salida: metricas/*.csv y metricas/resumen.json
"""
import collections
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

W = Path(__file__).parent
CL = ['mildiu', 'mosca_blanca', 'oidio', 'polilla', 'sana']
ENF = [0, 1, 2, 3]
SANA = 4
B = 1000
rng = np.random.default_rng(0)
OUT = W / 'metricas'
OUT.mkdir(exist_ok=True)
IOUS = np.linspace(0.5, 0.95, 10)


def nombre_base(n):
    n = Path(n).name
    n = re.sub(r'\.(jpg|jpeg|png|txt)$', '', n, flags=re.I)
    return re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', n, flags=re.I)


def grupo(n):
    return nombre_base(n).rsplit('__', 1)[0]


# ---------------- verdad de referencia (etiquetas v5)
GT = {}
for p in sorted((W / 'dataset/test/labels').glob('*.txt')):
    cajas = []
    for l in p.read_text().splitlines():
        v = l.split()
        if len(v) >= 5:
            k, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
            cajas.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, k])
    GT[p.stem] = np.array(cajas).reshape(-1, 5)
imgs = sorted(GT)
fijos = set((W / 'test_fijo_v4.txt').read_text().split()) if (W / 'test_fijo_v4.txt').exists() else set()
SUB = {'fijo97': [i for i in imgs if nombre_base(i) in fijos], 'test180': imgs}
print({k: len(v) for k, v in SUB.items()}, flush=True)

PRED = {}
for f in sorted((W / 'predicciones').glob('*.json')):
    d = json.loads(f.read_text())
    PRED[f.stem] = {Path(k).stem: v for k, v in d.items()}
MODELOS = sorted(PRED)


def iou(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def por_imagen(modelo):
    """Para cada imagen y clase: (confianzas, tp[n,10]) y n de cajas reales; emparejamiento voraz por imagen."""
    res = {}
    for im in imgs:
        g = GT[im]
        p = np.array(PRED[modelo][im]['map']).reshape(-1, 6)
        d = {}
        for c in range(5):
            gc = g[g[:, 4] == c][:, :4]
            pc = p[p[:, 5] == c]
            pc = pc[np.argsort(-pc[:, 4])]
            tp = np.zeros((len(pc), 10), bool)
            if len(pc) and len(gc):
                M = iou(pc[:, :4], gc)
                for t, thr in enumerate(IOUS):
                    usado = np.zeros(len(gc), bool)
                    for i in range(len(pc)):
                        cand = np.where((M[i] >= thr) & ~usado)[0]
                        if len(cand):
                            j = cand[np.argmax(M[i, cand])]
                            usado[j] = True
                            tp[i, t] = True
            d[c] = (pc[:, 4], tp, len(gc))
        res[im] = d
    return res


def ap101(conf, tp, ngt):
    if ngt == 0:
        return None
    if len(conf) == 0:
        return np.zeros(tp.shape[1] if tp.ndim == 2 else 10)
    o = np.argsort(-conf, kind='stable')
    tpc = np.cumsum(tp[o], 0)
    fpc = np.cumsum(~tp[o], 0)
    rec = tpc / ngt
    prec = tpc / np.maximum(tpc + fpc, 1e-12)
    x = np.linspace(0, 1, 101)
    aps = []
    for t in range(tp.shape[1]):
        r = np.concatenate([[0.0], rec[:, t], [1.0]])
        pr = np.concatenate([[1.0], prec[:, t], [0.0]])
        pr = np.flip(np.maximum.accumulate(np.flip(pr)))
        aps.append(np.trapezoid(np.interp(x, r, pr), x))
    return np.array(aps)


def map_de(pi, lista):
    """mAP50, mAP50-95 y AP50 por clase sobre una lista de imagenes (puede tener repetidas)."""
    apc = {}
    for c in range(5):
        conf = np.concatenate([pi[i][c][0] for i in lista]) if lista else np.zeros(0)
        tp = np.concatenate([pi[i][c][1] for i in lista]) if lista else np.zeros((0, 10), bool)
        ngt = sum(pi[i][c][2] for i in lista)
        a = ap101(conf, tp.reshape(-1, 10), ngt)
        if a is not None:
            apc[CL[c]] = a
    if not apc:
        return np.nan, np.nan, {}
    m = np.mean([a for a in apc.values()], 0)
    return m[0], m.mean(), {k: v[0] for k, v in apc.items()}


def remuestreo(lista):
    g = collections.defaultdict(list)
    for i in lista:
        g[grupo(i)].append(i)
    claves = list(g)
    for _ in range(B):
        sel = rng.choice(len(claves), len(claves), replace=True)
        yield [i for s in sel for i in g[claves[s]]]


PI = {m: por_imagen(m) for m in MODELOS}
print('emparejamiento listo', flush=True)

# ---------------- 1 y 2: mAP con IC, y comparacion pareada
filas, clases, pares = [], [], []
for sub, lista in SUB.items():
    reps = list(remuestreo(lista))
    boot = {m: np.array([map_de(PI[m], r)[:2] for r in reps]) for m in MODELOS}
    for m in MODELOS:
        m50, m5095, apc = map_de(PI[m], lista)
        lo, hi = np.nanpercentile(boot[m], [2.5, 97.5], axis=0)
        filas.append({'subconjunto': sub, 'modelo': m, 'mAP50': m50, 'mAP50_ic_bajo': lo[0], 'mAP50_ic_alto': hi[0],
                      'mAP50_95': m5095, 'mAP50_95_ic_bajo': lo[1], 'mAP50_95_ic_alto': hi[1]})
        for k, v in apc.items():
            clases.append({'subconjunto': sub, 'modelo': m, 'clase': k, 'AP50': v})
    if sub == 'fijo97' and 'v4_yolo11n_s1' in boot:
        base = boot['v4_yolo11n_s1']
        b0 = map_de(PI['v4_yolo11n_s1'], lista)
        for m in MODELOS:
            if m == 'v4_yolo11n_s1':
                continue
            dif = boot[m] - base
            pt = np.array(map_de(PI[m], lista)[:2]) - np.array(b0[:2])
            lo, hi = np.nanpercentile(dif, [2.5, 97.5], axis=0)
            pares.append({'modelo_v5': m, 'dif_mAP50': pt[0], 'ic_bajo': lo[0], 'ic_alto': hi[0],
                          'dif_mAP50_95': pt[1], 'ic95_bajo': lo[1], 'ic95_alto': hi[1]})
    print('bootstrap listo', sub, flush=True)

cajas = [{'subconjunto': s, 'clase': CL[c], 'cajas': int(sum((GT[i][:, 4] == c).sum() for i in l)),
          'imagenes_con_clase': int(sum((GT[i][:, 4] == c).any() for i in l))} for s, l in SUB.items() for c in range(5)]

# ---------------- 3: metricas por imagen (reglas de la app)


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def principal_real(g):
    return collections.Counter(g[:, 4].astype(int).tolist()).most_common(1)[0][0] if len(g) else None


img_rows, img_res = [], []
for m in MODELOS:
    for im in imgs:
        g = GT[im]
        p = np.array(PRED[m][im]['app']).reshape(-1, 6)
        top = int(p[np.argmax(p[:, 4]), 5]) if len(p) else SANA
        real = set(g[:, 4].astype(int).tolist())
        pred = set(p[:, 5].astype(int).tolist())
        img_rows.append({'modelo': m, 'imagen': im, 'grupo': grupo(im), 'fijo97': nombre_base(im) in fijos,
                         'principal_real': principal_real(g), 'principal_pred': top,
                         'acierto': (principal_real(g) == top) if len(g) else None,
                         'real_enf': bool(real & set(ENF)), 'pred_enf': bool(pred & set(ENF)),
                         **{f'real_{CL[c]}': c in real for c in ENF}, **{f'pred_{CL[c]}': c in pred for c in ENF}})
df = pd.DataFrame(img_rows)
for (m, sub), d in [((m, s), df[(df.modelo == m) & (df.fijo97 if s == 'fijo97' else True)]) for m in MODELOS for s in SUB]:
    con = d[d.acierto.notna()]
    k, n = int(con.acierto.sum()), len(con)
    enf = d[d.real_enf]
    sanas = d[~d.real_enf]
    fs = int((~enf.pred_enf).sum())
    r = {'modelo': m, 'subconjunto': sub, 'imagenes_etiquetadas': n, 'acierto_principal': k / n,
         'acierto_ic': wilson(k, n), 'imgs_con_plaga': len(enf), 'falso_sano': fs / max(len(enf), 1),
         'falso_sano_ic': wilson(fs, len(enf)), 'imgs_solo_sanas': len(sanas),
         'especificidad_hay_plaga': float((~sanas.pred_enf).mean()) if len(sanas) else math.nan}
    for c in ENF:
        pos, neg = d[d[f'real_{CL[c]}']], d[~d[f'real_{CL[c]}']]
        r[f'sens_{CL[c]}'] = float(pos[f'pred_{CL[c]}'].mean()) if len(pos) else math.nan
        r[f'esp_{CL[c]}'] = float((~neg[f'pred_{CL[c]}']).mean()) if len(neg) else math.nan
        r[f'n_{CL[c]}'] = len(pos)
    img_res.append(r)

# lineas base del acierto de la clase principal
base = []
for sub, lista in SUB.items():
    reales = [principal_real(GT[i]) for i in lista if len(GT[i])]
    may = collections.Counter(reales).most_common(1)[0]
    base.append({'subconjunto': sub, 'siempre_sana': reales.count(SANA) / len(reales),
                 'siempre_mayoritaria': may[1] / len(reales), 'clase_mayoritaria': CL[may[0]], 'n': len(reales)})

# McNemar v4 vs v5 (acierto, mismas imagenes del test fijo)
mcn = []
a4 = df[(df.modelo == 'v4_yolo11n_s1') & df.fijo97 & df.acierto.notna()].set_index('imagen').acierto
for m in MODELOS:
    if m == 'v4_yolo11n_s1':
        continue
    a5 = df[(df.modelo == m) & df.fijo97 & df.acierto.notna()].set_index('imagen').acierto
    b_ = int((a4 & ~a5).sum())
    c_ = int((~a4 & a5).sum())
    p = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
    mcn.append({'modelo_v5': m, 'v4_bien_v5_mal': b_, 'v4_mal_v5_bien': c_, 'p_mcnemar_exacta': p})

pd.DataFrame(filas).to_csv(OUT / 'map_con_ic.csv', index=False)
pd.DataFrame(clases).to_csv(OUT / 'ap_por_clase.csv', index=False)
pd.DataFrame(pares).to_csv(OUT / 'v4_vs_v5_pareado.csv', index=False)
pd.DataFrame(cajas).to_csv(OUT / 'cajas_por_clase_prueba.csv', index=False)
pd.DataFrame(img_res).to_csv(OUT / 'metricas_por_imagen.csv', index=False)
pd.DataFrame(base).to_csv(OUT / 'lineas_base.csv', index=False)
pd.DataFrame(mcn).to_csv(OUT / 'mcnemar_v4_v5.csv', index=False)
df.to_csv(OUT / 'predicciones_por_imagen_app.csv', index=False)
print('listo', flush=True)
