"""A5. Busca errores probables de etiquetado usando los 9 modelos v5 como "segundos revisores".

En validacion + prueba (561 imagenes; en entrenamiento los modelos memorizan y no sirven de juez):
  FALTANTE      : >= 6 de 9 modelos marcan un objeto (conf >= 0.35) en el mismo lugar y no hay ninguna
                  caja real (de ninguna clase) que lo cubra (IoU < 0.3)  -> probable lesion sin etiquetar
  CLASE         : caja real donde >= 6 de 9 modelos predicen OTRA clase (IoU >= 0.5) y casi ninguno la
                  clase etiquetada -> probable clase equivocada (o clase dudosa)
  NO_VISTA      : caja real de plaga/enfermedad que ningun modelo toca (ni con conf >= 0.10, IoU >= 0.3)
                  -> etiqueta dudosa o muy dificil
En las 1966 imagenes (revision estructural, sin modelos):
  DUPLICADA     : dos cajas de la misma clase casi iguales (IoU >= 0.7)
  DIMINUTA      : caja menor a 16 px de lado a 640 px
  ENORME        : caja de plaga/enfermedad que ocupa > 25 % de la imagen (posible hoja entera)
Salida: errores/candidatos.csv, errores/resumen.txt y hojas errores/hoja_<tipo>.jpg con las cajas dibujadas
(verde = etiqueta actual, rojo = lo que proponen los modelos).
"""
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

W = Path(__file__).parent
D = W / 'dataset'
OUT = W / 'errores'
OUT.mkdir(exist_ok=True)
CL = ['mildiu', 'mosca_blanca', 'oidio', 'polilla', 'sana']
ENF = {0, 1, 2, 3}
VOTOS = 6


def nombre_base(n):
    n = re.sub(r'\.(jpg|jpeg|png|txt)$', '', Path(n).name, flags=re.I)
    return re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', n, flags=re.I)


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def gt_de(lbl):
    c = []
    if lbl.exists():
        for l in lbl.read_text().splitlines():
            v = l.split()
            if len(v) >= 5:
                k, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
                c.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, k])
    return np.array(c).reshape(-1, 5)


cand = []
# ---------- revision estructural en las 1966
for s in ('train', 'valid', 'test'):
    for img in sorted((D / s / 'images').iterdir()):
        g = gt_de(D / s / 'labels' / (img.stem + '.txt'))
        if not len(g):
            continue
        with Image.open(img) as im:
            w, h = ImageOps.exif_transpose(im).size
        esc = 640 / max(w, h)
        for i, b in enumerate(g):
            lado = min((b[2] - b[0]) * w, (b[3] - b[1]) * h) * esc
            area = (b[2] - b[0]) * (b[3] - b[1])
            if lado < 16:
                cand.append({'tipo': 'DIMINUTA', 'subconjunto': s, 'imagen': nombre_base(img), 'clase_etiqueta': CL[int(b[4])],
                             'clase_modelos': '', 'detalle': f'lado {lado:.0f} px a 640', 'caja': b[:4].round(4).tolist(), 'ruta': str(img)})
            if int(b[4]) in ENF and area > 0.25:
                cand.append({'tipo': 'ENORME', 'subconjunto': s, 'imagen': nombre_base(img), 'clase_etiqueta': CL[int(b[4])],
                             'clase_modelos': '', 'detalle': f'{area * 100:.0f} % de la imagen', 'caja': b[:4].round(4).tolist(), 'ruta': str(img)})
        for c in set(g[:, 4].astype(int)):
            gc = g[g[:, 4] == c][:, :4]
            M = iou(gc, gc)
            np.fill_diagonal(M, 0)
            for i, j in zip(*np.where(np.triu(M) >= 0.7)):
                cand.append({'tipo': 'DUPLICADA', 'subconjunto': s, 'imagen': nombre_base(img), 'clase_etiqueta': CL[c],
                             'clase_modelos': '', 'detalle': f'IoU {M[i, j]:.2f} entre dos cajas', 'caja': gc[i].round(4).tolist(), 'ruta': str(img)})

# ---------- con los modelos, en validacion y prueba
for s, carpeta in (('valid', 'predicciones_valid'), ('test', 'predicciones')):
    preds = {}
    for f in sorted((W / carpeta).glob('v5_*.json')):
        preds[f.stem] = {Path(k).stem: v for k, v in json.loads(f.read_text()).items()}
    if len(preds) < 9:
        print('faltan predicciones de', s, len(preds))
        continue
    for img in sorted((D / s / 'images').iterdir()):
        g = gt_de(D / s / 'labels' / (img.stem + '.txt'))
        P = [np.array(preds[m][img.stem]['app']).reshape(-1, 6) for m in preds]
        L = [np.array(preds[m][img.stem]['map']).reshape(-1, 6) for m in preds]
        # FALTANTE: agrupar predicciones de alta confianza de los 9 modelos
        altas = [(mi, p) for mi, pm in enumerate(P) for p in pm if p[4] >= 0.35]
        usados = [False] * len(altas)
        for i, (mi, p) in enumerate(altas):
            if usados[i]:
                continue
            grupo = [i]
            for j in range(i + 1, len(altas)):
                if not usados[j] and altas[j][0] not in {altas[k][0] for k in grupo} and iou(p[:4], altas[j][1][:4])[0, 0] >= 0.5:
                    grupo.append(j)
            modelos = {altas[k][0] for k in grupo}
            if len(modelos) >= VOTOS:
                for k in grupo:
                    usados[k] = True
                caja = np.mean([altas[k][1][:4] for k in grupo], 0)
                cubre = iou(caja, g[:, :4]).max() if len(g) else 0
                if cubre < 0.3:
                    clases = collections.Counter(int(altas[k][1][5]) for k in grupo).most_common(1)[0][0]
                    conf = np.mean([altas[k][1][4] for k in grupo])
                    cand.append({'tipo': 'FALTANTE', 'subconjunto': s, 'imagen': nombre_base(img), 'clase_etiqueta': '(ninguna)',
                                 'clase_modelos': CL[clases], 'detalle': f'{len(modelos)}/9 modelos, conf media {conf:.2f}',
                                 'caja': caja.round(4).tolist(), 'ruta': str(img), 'puntaje': len(modelos) * conf})
        # CLASE y NO_VISTA, por cada caja real
        for b in g:
            c = int(b[4])
            otra, misma, toca = collections.Counter(), 0, 0
            for mi in range(len(P)):
                if len(P[mi]):
                    M = iou(b[:4], P[mi][:, :4])[0]
                    ok = M >= 0.5
                    if (P[mi][ok, 5] == c).any():
                        misma += 1
                    elif ok.any():
                        otra[int(P[mi][ok][np.argmax(P[mi][ok, 4]), 5])] += 1
                if len(L[mi]):
                    Ml = iou(b[:4], L[mi][:, :4])[0]
                    if ((Ml >= 0.3) & (L[mi][:, 4] >= 0.10)).any():
                        toca += 1
            if otra and sum(otra.values()) >= VOTOS and misma <= 1:
                k, n = otra.most_common(1)[0]
                cand.append({'tipo': 'CLASE', 'subconjunto': s, 'imagen': nombre_base(img), 'clase_etiqueta': CL[c],
                             'clase_modelos': CL[k], 'detalle': f'{n}/9 modelos dicen {CL[k]}, {misma}/9 dicen {CL[c]}',
                             'caja': b[:4].round(4).tolist(), 'ruta': str(img), 'puntaje': n})
            if c in ENF and toca == 0:
                cand.append({'tipo': 'NO_VISTA', 'subconjunto': s, 'imagen': nombre_base(img), 'clase_etiqueta': CL[c],
                             'clase_modelos': '', 'detalle': 'ningun modelo la toca', 'caja': b[:4].round(4).tolist(), 'ruta': str(img)})

C = pd.DataFrame(cand)
C.drop(columns=['ruta']).to_csv(OUT / 'candidatos.csv', index=False)

lin = ['=== Candidatos a error de etiquetado ===']
lin.append(C.groupby(['tipo', 'subconjunto']).size().unstack(fill_value=0).to_string())
lin.append('\nPor clase (etiqueta actual o propuesta por los modelos):')
C['clase'] = np.where(C.clase_etiqueta.isin(['(ninguna)']), C.clase_modelos, C.clase_etiqueta)
lin.append(C.groupby(['tipo', 'clase']).size().unstack(fill_value=0).to_string())
lin.append('\nCLASE: etiqueta -> lo que dicen los modelos')
if (C.tipo == 'CLASE').any():
    lin.append(C[C.tipo == 'CLASE'].groupby(['clase_etiqueta', 'clase_modelos']).size().to_string())
lin.append(f"\nImagenes distintas con algun candidato: {C.imagen.nunique()} de 1966 "
           f"(FALTANTE/CLASE/NO_VISTA en {C[C.tipo.isin(['FALTANTE','CLASE','NO_VISTA'])].imagen.nunique()} de 561 de valid+test)")
txt = '\n'.join(lin)
(OUT / 'resumen.txt').write_text(txt, encoding='utf-8')
print(txt)

# ---------- hojas visuales (las mas claras de cada tipo)
COLOR = {'mildiu': 'orange', 'mosca_blanca': 'white', 'oidio': 'cyan', 'polilla': 'magenta', 'sana': 'lime'}
for tipo in ('FALTANTE', 'CLASE', 'NO_VISTA', 'ENORME', 'DUPLICADA'):
    d = C[C.tipo == tipo]
    if not len(d):
        continue
    if 'puntaje' in d and d.puntaje.notna().any():
        d = d.sort_values('puntaje', ascending=False)
    d = d.drop_duplicates('imagen').head(12)
    T = 360
    hoja = Image.new('RGB', (T * 4, (T + 34) * ((len(d) + 3) // 4)), 'white')
    dr = ImageDraw.Draw(hoja)
    for n, (_, r) in enumerate(d.iterrows()):
        im = ImageOps.exif_transpose(Image.open(r.ruta)).convert('RGB')
        w, h = im.size
        s = T / max(w, h)
        im = im.resize((int(w * s), int(h * s)))
        di = ImageDraw.Draw(im)
        lbl = D / r.subconjunto / 'labels' / (Path(r.ruta).stem + '.txt')
        for b in gt_de(lbl):
            di.rectangle([b[0] * im.width, b[1] * im.height, b[2] * im.width, b[3] * im.height], outline='lime', width=2)
        x1, y1, x2, y2 = r.caja
        di.rectangle([x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height], outline='red', width=4)
        x0, y0 = (n % 4) * T, (n // 4) * (T + 34)
        hoja.paste(im, (x0, y0 + 32))
        dr.text((x0 + 2, y0 + 2), f'{n + 1}. {r.imagen[:44]}', fill='black')
        dr.text((x0 + 2, y0 + 16), f'{r.clase_etiqueta} -> {r.clase_modelos} | {r.detalle}'[:60], fill='darkred')
    hoja.save(OUT / f'hoja_{tipo}.jpg', quality=85)
print('hojas listas')
