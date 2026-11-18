"""A11. Validacion del aviso "Mucha luz" de la app con la prueba de estres y el modelo v6 instalado.

Para las 180 imagenes de prueba en cada nivel de exposicion (estres/ev*/images, creado por a8):
- aviso de la app: luminancia 0.299R+0.587G+0.114B muestreada cada 6 px; "Mucha luz" si media > 200 o
  > 15 % de pixeles con luminancia > 240 (scanner_page.dart, _checkLux).
- deteccion con el modelo v6 (conf 0.30, iou 0.45, NMS agnostica, como la app): acierto por imagen =
  alguna clase detectada esta entre las clases etiquetadas (etiquetas v6, Roboflow version 3).
- falla inducida por la luz: la imagen acierta con su exposicion original y falla con la sobreexposicion.
Salida: v6/aviso_luz.csv (por imagen y nivel) y v6/aviso_luz_resumen.txt
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from ultralytics import YOLO

W = Path(__file__).parent
MODELO = Path(r"D:/ProyectoInvestigacion/app/backend/app/models/best.pt")
V3 = Path(sys.argv[1])                       # carpeta del zip de Roboflow version 3 (solo labels)
NIV = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
base = lambda n: re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", n, flags=re.I)

gt = {}
for s in ["train", "valid", "test"]:
    for f in (V3 / s / "labels").glob("*.txt"):
        gt[base(f.stem)] = {int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()}

def aviso(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)[::6, ::6]
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return float(lum.mean()), float((lum > 240).mean() * 100)

m = YOLO(str(MODELO))
filas = []
for ev in NIV:
    for p in sorted((W / "estres" / f"ev{ev:+.1f}" / "images").iterdir()):
        g = gt[base(p.stem)]
        media, sobre = aviso(p)
        b = m(str(p), conf=0.30, iou=0.45, imgsz=640, agnostic_nms=True, verbose=False, device="cpu")[0].boxes
        det = set(b.cls.int().tolist())
        filas.append({"imagen": base(p.stem), "ev": ev, "con_plaga": bool(g), "media_lum": round(media, 1),
                      "pct_sobre240": round(sobre, 2), "aviso": media > 200 or sobre > 15,
                      "acierto": bool(det & g) if g else (len(det) == 0), "falso_sano": bool(g) and not det})
    print("nivel", ev, flush=True)
d = pd.DataFrame(filas)
d.to_csv(W / "v6" / "aviso_luz.csv", index=False)
