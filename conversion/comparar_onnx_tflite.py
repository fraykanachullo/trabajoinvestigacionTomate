"""Compara best.onnx y best_float16.tflite con la misma entrada (letterbox 640, gris 114, como la app)."""
import sys, glob
import numpy as np, onnxruntime as ort, tensorflow as tf
from PIL import Image, ImageOps
d = sys.argv[1]
fs = sorted(glob.glob('D:/ProyectoInvestigacion/para_roboflow/test/images/*.jpg'))[::2][:96]
so = ort.InferenceSession(f'{d}/best.onnx'); it = tf.lite.Interpreter(f'{d}/tflite_out/best_float16.tflite'); it.allocate_tensors()
ei, eo = it.get_input_details()[0], it.get_output_details()[0]
print('tflite entrada', ei['shape'], 'salida', eo['shape'])
dif, igual = [], 0
for f in fs:
    im = ImageOps.exif_transpose(Image.open(f)).convert('RGB'); w, h = im.size; s = min(640 / w, 640 / h)
    nw, nh = round(w * s), round(h * s); x = np.full((640, 640, 3), 114 / 255, np.float32)
    x[(640 - nh) // 2:(640 - nh) // 2 + nh, (640 - nw) // 2:(640 - nw) // 2 + nw] = np.asarray(im.resize((nw, nh), Image.BILINEAR), np.float32) / 255
    a = so.run(None, {so.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
    it.set_tensor(ei['index'], x[None]); it.invoke(); b = it.get_tensor(eo['index'])[0]
    sa, sb = a[4:], b[4:]; dif.append(np.abs(sa - sb).max())
    ia, ib = sa.max(0).argmax(), sb.max(0).argmax()
    igual += int(sa[:, ia].argmax() == sb[:, ib].argmax() and abs(sa.max() - sb.max()) < 0.02)
print(f'imagenes {len(fs)}  diferencia maxima de score {max(dif):.4f}  deteccion principal igual {igual}/{len(fs)}')
