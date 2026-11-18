"""A3. Exposicion de cada imagen y rendimiento por nivel de exposicion (obs. 1, ruta a).

Por imagen (las 1966; version 1024 px de Roboflow, sin modificar):
  brillo      = luminancia media (0-255)
  saturados   = % de pixeles con algun canal >= 250 (sobreexposicion)
  oscuros     = % de pixeles con luminancia <= 20
  contraste   = desviacion estandar de la luminancia
  especular   = % de pixeles muy brillantes y poco saturados (V >= 230 y S <= 40): proxy de reflejos
En prueba (180): terciles de % saturados y de brillo; por tercil, acierto de la clase principal, deteccion
de plaga y tasa de falso sano (media de los 9 modelos v5 y modelo elegido), con IC; regresion logistica
del acierto ~ exposicion (z) + sesion, con errores agrupados por grupo de captura.
Salida: exposicion/*.csv, exposicion/resumen.txt
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from PIL import Image, ImageOps

W = Path(__file__).parent
OUT = W / 'exposicion'
OUT.mkdir(exist_ok=True)


def nombre_base(n):
    n = Path(n).name
    n = re.sub(r'\.(jpg|jpeg|png|txt)$', '', n, flags=re.I)
    return re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', n, flags=re.I)


def info(n):
    b = nombre_base(n)
    f = re.findall(r'(\d{8})_(\d{2})(\d{2})\d{2}', b)[-1]
    h = int(f[1])
    return b.rsplit('__', 1)[0], f[0], 'manana' if h < 11 else 'mediodia' if h < 14 else 'tarde', h + int(f[2]) / 60


filas = []
for s in ('train', 'valid', 'test'):
    for p in sorted((W / 'dataset' / s / 'images').iterdir()):
        im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        im.thumbnail((640, 640))
        a = np.asarray(im, np.float32)
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        hsv = np.asarray(im.convert('HSV'), np.float32)
        g, ses, franja, hora = info(p.name)
        filas.append({'imagen': p.stem, 'subconjunto': s, 'grupo': g, 'sesion': ses, 'franja': franja, 'hora': hora,
                      'brillo': lum.mean(), 'saturados': 100 * (a >= 250).any(-1).mean(),
                      'oscuros': 100 * (lum <= 20).mean(), 'contraste': lum.std(),
                      'especular': 100 * ((hsv[..., 2] >= 230) & (hsv[..., 1] <= 40)).mean()})
E = pd.DataFrame(filas)
E.to_csv(OUT / 'exposicion_por_imagen.csv', index=False)

lineas = ['=== Exposicion por franja y sesion (todas las imagenes; mediana) ===']
lineas.append(E.groupby('franja')[['brillo', 'saturados', 'especular', 'contraste']].median().round(2).to_string())
lineas.append(E.groupby('sesion')[['brillo', 'saturados', 'especular', 'contraste']].median().round(2).to_string())
lineas.append('Imagenes por sesion y franja:\n' + pd.crosstab(E.sesion, E.franja).to_string())

# ---- rendimiento por tercil en prueba
P = pd.read_csv(W / 'metricas' / 'predicciones_por_imagen_app.csv')
P = P[P.modelo.str.startswith('v5_')]
T = E[E.subconjunto == 'test'].copy()
for var in ('saturados', 'brillo', 'especular'):
    T[f'tercil_{var}'] = pd.qcut(T[var].rank(method='first'), 3, labels=['bajo', 'medio', 'alto'])
D = P.merge(T, on='imagen', suffixes=('', '_e'))


def ic_boot(d, col, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    gr = d.groupby('grupo')[col].agg(['sum', 'count'])
    v = gr.values
    est = v[:, 0].sum() / v[:, 1].sum()
    bs = []
    for _ in range(B):
        s = v[rng.integers(0, len(v), len(v))]
        bs.append(s[:, 0].sum() / s[:, 1].sum())
    return est, *np.percentile(bs, [2.5, 97.5])


res = []
for var in ('saturados', 'brillo', 'especular'):
    for t in ('bajo', 'medio', 'alto'):
        d = D[D[f'tercil_{var}'] == t]
        acc = d[d.acierto.notna()].assign(acierto=lambda x: x.acierto.astype(float))
        fs = d[d.real_enf].assign(fs=lambda x: (~x.pred_enf).astype(float))
        a = ic_boot(acc, 'acierto')
        f = ic_boot(fs, 'fs') if len(fs) else (np.nan, np.nan, np.nan)
        rango = T.loc[T[f'tercil_{var}'] == t, var]
        res.append({'variable': var, 'tercil': t, 'rango': f'{rango.min():.1f}-{rango.max():.1f}',
                    'imagenes': int(len(rango)), 'acierto': a[0], 'acierto_ic': f'{a[1]:.3f}-{a[2]:.3f}',
                    'falso_sano': f[0], 'falso_sano_ic': f'{f[1]:.3f}-{f[2]:.3f}',
                    'sesiones': dict(T.loc[T[f'tercil_{var}'] == t, 'sesion'].value_counts())})
R = pd.DataFrame(res)
R.to_csv(OUT / 'rendimiento_por_tercil.csv', index=False)
lineas += ['', '=== Prueba: rendimiento por tercil (9 modelos v5, IC bootstrap por grupo) ===', R.to_string(index=False)]

# ---- regresion logistica: acierto ~ exposicion + sesion (errores agrupados por grupo)
acc = D[D.acierto.notna()].copy()
acc['acierto'] = acc.acierto.astype(int)
for var in ('saturados', 'brillo', 'especular'):
    acc[f'z_{var}'] = (acc[var] - acc[var].mean()) / acc[var].std()
    try:
        m = smf.logit(f'acierto ~ z_{var} + C(sesion) + C(modelo)', data=acc).fit(
            disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(acc.grupo)[0]})
        b, lo, hi = m.params[f'z_{var}'], *m.conf_int().loc[f'z_{var}']
        lineas.append(f'Logit acierto ~ z_{var} + sesion + modelo: OR por 1 DE = {np.exp(b):.2f} '
                      f'(IC95 {np.exp(lo):.2f}-{np.exp(hi):.2f}), p = {m.pvalues[f"z_{var}"]:.3f}, n = {len(acc)}')
    except Exception as e:
        lineas.append(f'Logit {var}: no convergio ({e})')

texto = '\n'.join(lineas)
(OUT / 'resumen.txt').write_text(texto, encoding='utf-8')
print(texto)
