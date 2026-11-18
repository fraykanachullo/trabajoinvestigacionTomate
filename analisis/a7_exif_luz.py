"""A7. Luz de la escena por foto a partir del EXIF de la camara (Samsung A56), y su relacion con el detector.

BrightnessValue (Bv, APEX) es la estimacion de la camara de la luminancia de la escena: cada +1 duplica la luz.
Iluminancia aproximada de la escena: E (lux) ~ 2.5 * 2^Bv (aproximacion APEX con constante de
calibracion ~ 2.5 lux; se usa para orden de magnitud, la comparacion se hace en Bv).
Tambien ExposureTime e ISO. Las fotos se emparejan con el dataset por la hora en el nombre (AAAAMMDD_HHMMSS).
Los cuadros de video no tienen EXIF y quedan fuera.
Salida: exposicion/exif_por_imagen.csv y exposicion/exif_resumen.txt
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from PIL import ExifTags, Image
from pillow_heif import register_heif_opener

register_heif_opener()
W = Path(__file__).parent
CRUDO = Path('D:/ProyectoInvestigacion/datasetcrudo')
OUT = W / 'exposicion'
TAG = {v: k for k, v in ExifTags.TAGS.items()}

filas = []
for p in CRUDO.rglob('*'):
    if p.suffix.lower() not in ('.jpg', '.jpeg', '.heic'):
        continue
    m = re.search(r'(\d{8}_\d{6})', p.name)
    if not m:
        continue
    try:
        ex = Image.open(p).getexif()
        ifd = ex.get_ifd(0x8769)
        g = lambda k: ifd.get(TAG[k], ex.get(TAG[k]))
        bv, et, iso = g('BrightnessValue'), g('ExposureTime'), g('ISOSpeedRatings')
        filas.append({'clave': m.group(1) + ('_001' if '_001' in p.stem else ''), 'carpeta': str(p.parent.relative_to(CRUDO)),
                      'Bv': float(bv) if bv is not None else np.nan,
                      'exposicion_s': float(et) if et is not None else np.nan,
                      'iso': float(iso) if iso is not None else np.nan})
    except Exception as e:
        print('error', p.name, e)
X = pd.DataFrame(filas).drop_duplicates('clave')
X['lux_aprox'] = 2.5 * 2 ** X.Bv
print('fotos con EXIF', len(X), 'con Bv', X.Bv.notna().sum(), flush=True)

E = pd.read_csv(OUT / 'exposicion_por_imagen.csv')
E['clave'] = E.imagen.str.extract(r'__(\d{8}_\d{6}(?:_001)?)(?:_jpg|$)', expand=False)
D = E.merge(X, on='clave', how='left')
D.to_csv(OUT / 'exif_por_imagen.csv', index=False)

lin = [f'Imagenes del dataset con EXIF de luz: {D.Bv.notna().sum()} de {len(D)} (sin EXIF: cuadros de video)']
lin.append('\nBv (mediana) y lux aproximados por franja y sesion:')
lin.append(D.groupby(['sesion', 'franja'])[['Bv', 'lux_aprox', 'saturados']].median().round(2).to_string())
lin.append('\nCorrelacion de Spearman entre Bv y las medidas de la imagen:')
lin.append(D[['Bv', 'saturados', 'brillo', 'especular', 'contraste']].corr(method='spearman').round(2).loc['Bv'].to_string())

P = pd.read_csv(W / 'metricas' / 'predicciones_por_imagen_app.csv')
P = P[P.modelo.str.startswith('v5_')]
A = P.merge(D[['imagen', 'Bv', 'saturados', 'sesion', 'grupo']].rename(columns={'grupo': 'grupo_e'}), on='imagen')
A = A[A.acierto.notna() & A.Bv.notna()].copy()
A['acierto'] = A.acierto.astype(int)
A['z_Bv'] = (A.Bv - A.Bv.mean()) / A.Bv.std()
A['tercil_Bv'] = pd.qcut(A.Bv.rank(method='first'), 3, labels=['baja', 'media', 'alta'])
lin.append(f'\nPrueba (9 modelos v5), imagenes con EXIF: {A.imagen.nunique()}')
lin.append(A.groupby('tercil_Bv', observed=True).agg(n=('imagen', 'nunique'), Bv_min=('Bv', 'min'), Bv_max=('Bv', 'max'),
                                                     acierto=('acierto', 'mean')).round(3).to_string())
for f in ('acierto ~ z_Bv + C(modelo)', 'acierto ~ z_Bv + C(sesion) + C(modelo)'):
    try:
        m = smf.logit(f, data=A).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(A.grupo)[0]})
        b, (lo, hi) = m.params['z_Bv'], m.conf_int().loc['z_Bv']
        lin.append(f'{f}: OR por 1 DE de Bv = {np.exp(b):.2f} (IC95 {np.exp(lo):.2f}-{np.exp(hi):.2f}), p = {m.pvalues["z_Bv"]:.3f}')
    except Exception as e:
        lin.append(f'{f}: no convergio ({e})')
txt = '\n'.join(lin)
(OUT / 'exif_resumen.txt').write_text(txt, encoding='utf-8')
print(txt)
