"""A1. Predicciones por imagen de los 10 modelos sobre las 180 imagenes de prueba de la v5.

Guarda predicciones/<modelo>.json con, por imagen:
  map: cajas con conf >= 0.001 y NMS por clase (IoU 0.7), como la validacion de Ultralytics -> para mAP
  app: cajas con conf >= 0.30, IoU 0.45 y NMS agnostica, como la app -> para metricas por imagen
Coordenadas normalizadas x1, y1, x2, y2, confianza, clase.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

W = Path(__file__).parent
TEST = W / 'dataset' / 'test' / 'images'
OUT = W / 'predicciones'
OUT.mkdir(exist_ok=True)
imgs = sorted(p for p in TEST.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))


def cajas(r):
    b = r.boxes
    if not len(b):
        return []
    xyxyn = b.xyxyn.tolist()
    return [[round(v, 5) for v in c] + [round(float(s), 5), int(k)] for c, s, k in zip(xyxyn, b.conf.tolist(), b.cls.tolist())]


for w in sorted((W / 'pesos').glob('*.pt')):
    destino = OUT / f'{w.stem}.json'
    if destino.exists():
        print('[listo]', w.stem, flush=True)
        continue
    model = YOLO(str(w))
    res = {}
    for p in imgs:
        im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        a = model(im, conf=0.001, iou=0.7, imgsz=640, max_det=300, verbose=False, device='cpu')[0]
        b = model(im, conf=0.30, iou=0.45, imgsz=640, agnostic_nms=True, verbose=False, device='cpu')[0]
        res[p.name] = {'map': cajas(a), 'app': cajas(b)}
    destino.write_text(json.dumps(res))
    print('[hecho]', w.stem, len(res), flush=True)
