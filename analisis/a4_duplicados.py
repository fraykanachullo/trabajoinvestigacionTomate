"""A4. Auditoria de casi-duplicados entre subconjuntos con huella perceptual (pHash, independiente del tiempo).

Para cada imagen: pHash de 64 bits (DCT de la imagen en gris a 32x32, 8x8 frecuencias bajas).
Se comparan todos los pares de subconjuntos distintos (train-valid, train-test, valid-test) por distancia de
Hamming. Como referencia se usa la distancia entre fotos del MISMO grupo de captura (fotos de la misma
rafaga, que si son casi duplicados). Guarda:
  duplicados/pares_cercanos.csv  pares entre subconjuntos con distancia <= 12
  duplicados/resumen.json         distribuciones y conteos por umbral
  duplicados/pares_mas_cercanos.jpg  hoja con los 16 pares mas parecidos, para revisarlos a ojo
"""
import collections
import itertools
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

W = Path(__file__).parent
D = W / 'dataset'
OUT = W / 'duplicados'
OUT.mkdir(exist_ok=True)


def nombre_base(n):
    n = Path(n).name
    n = re.sub(r'\.(jpg|jpeg|png|txt)$', '', n, flags=re.I)
    return re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', n, flags=re.I)


def grupo(p):
    return nombre_base(p).rsplit('__', 1)[0]


def phash(p):
    im = ImageOps.exif_transpose(Image.open(p)).convert('L').resize((32, 32), Image.LANCZOS)
    c = dctn(np.asarray(im, np.float64), norm='ortho')[:8, :8].flatten()
    return (c[1:] > np.median(c[1:])).astype(np.uint8)  # 63 bits (sin el componente continuo)


filas = []
for s in ('train', 'valid', 'test'):
    for p in sorted((D / s / 'images').iterdir()):
        filas.append((s, p, grupo(p), phash(p)))
print('imagenes', len(filas), flush=True)

H = np.stack([f[3] for f in filas])
split = np.array([f[0] for f in filas])
grp = np.array([f[2] for f in filas])
# distancia de Hamming entre todos los pares (63 bits)
dist = (H[:, None, :] != H[None, :, :]).sum(-1)
iu = np.triu_indices(len(filas), 1)
d = dist[iu]
mismo_grupo = grp[iu[0]] == grp[iu[1]]
distinto_split = split[iu[0]] != split[iu[1]]

res = {
    'imagenes': len(filas),
    'pares_mismo_grupo': int(mismo_grupo.sum()),
    'mismo_grupo_percentiles_p10_p25_p50': [float(np.percentile(d[mismo_grupo], q)) for q in (10, 25, 50)],
    'pares_entre_subconjuntos': int(distinto_split.sum()),
    'entre_subconjuntos_minimo': int(d[distinto_split].min()),
    'entre_subconjuntos_por_umbral': {str(t): int(((d <= t) & distinto_split).sum()) for t in (0, 2, 4, 6, 8, 10, 12)},
    'mismo_grupo_por_umbral_fraccion': {str(t): round(float((d[mismo_grupo] <= t).mean()), 3) for t in (4, 6, 8, 10, 12)},
}
cerca = np.where(distinto_split & (d <= 12))[0]
orden = cerca[np.argsort(d[cerca])]
with open(OUT / 'pares_cercanos.csv', 'w', encoding='utf-8') as f:
    f.write('distancia,subconjunto_a,imagen_a,subconjunto_b,imagen_b\n')
    for k in orden:
        a, b = iu[0][k], iu[1][k]
        f.write(f'{d[k]},{filas[a][0]},{nombre_base(filas[a][1])},{filas[b][0]},{nombre_base(filas[b][1])}\n')
(OUT / 'resumen.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

# Hoja de contacto con los 16 pares mas parecidos
top = orden[:16]
if len(top):
    T = 220
    hoja = Image.new('RGB', (T * 4 + 30, (T + 18) * ((len(top) + 1) // 2)), 'white')
    dr = ImageDraw.Draw(hoja)
    for i, k in enumerate(top):
        a, b = iu[0][k], iu[1][k]
        x0 = (i % 2) * (2 * T + 30)
        y0 = (i // 2) * (T + 18)
        for j, idx in enumerate((a, b)):
            im = ImageOps.exif_transpose(Image.open(filas[idx][1])).convert('RGB')
            im.thumbnail((T, T))
            hoja.paste(im, (x0 + j * T, y0 + 16))
        dr.text((x0 + 2, y0 + 2), f'd={d[k]}  {filas[a][0]}-{filas[b][0]}', fill='black')
    hoja.save(OUT / 'pares_mas_cercanos.jpg', quality=85)
