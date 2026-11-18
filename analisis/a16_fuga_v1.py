"""A16. Obs. 2 (minimo aceptable del revisor): modelo v1 evaluado en su propio test con fuga (103 imagenes).

Mismo modelo y mismas etiquetas: se separa la prueba v1 en imagenes con 'foto hermana' (mismo grupo de captura,
< 30 s, en entrenamiento o validacion) y sin ella, y se compara el mAP. Tambien se evalua en validacion.
Salida: v1_fuga/resultados_fuga_v1.csv y resumen_fuga_v1.txt
"""
import re
from pathlib import Path
import pandas as pd, yaml
from ultralytics import YOLO
W = Path(__file__).parent / "v1_fuga"; DS = W / "ds"
MOD = Path(r"D:/ProyectoInvestigacion/app/modelos_anteriores/v1_yolov8n_exp3/best.pt")
filas = []
for s in ["train", "valid", "test"]:
    for p in (DS / s / "images").iterdir():
        m = re.search(r"(\d{8})_(\d{2})(\d{2})(\d{2})", p.name)
        pref = p.name.split(m.group(0))[0] if m else ""
        filas.append(dict(split=s, img=p, pref=pref, dia=m.group(1) if m else "", t=int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) if m else -1))
d = pd.DataFrame(filas).sort_values(["pref", "dia", "t"]).reset_index(drop=True)
g, prev = [], None
for _, r in d.iterrows():
    nuevo = prev is None or r.pref != prev.pref or r.dia != prev.dia or r.t < 0 or r.t - prev.t >= 30
    g.append((g[-1] + 1) if (g and nuevo) else (g[-1] if g else 0)); prev = r
d["grupo"] = g
sp = d.groupby("grupo").split.agg(set)
d["con_hermana"] = d.apply(lambda r: r.split == "test" and bool(sp[r.grupo] & {"train", "valid"}), axis=1)
t = d[d.split == "test"]
nombres = yaml.safe_load(open(DS / "data.yaml"))["names"]
def lista(nombre, imgs):
    f = W / f"{nombre}.txt"; f.write_text("\n".join(str(p) for p in imgs))
    y = W / f"{nombre}.yaml"; y.write_text(yaml.safe_dump({"path": str(W), "train": str(f), "val": str(f), "names": dict(enumerate(nombres))}))
    return y
conj = {"test_103": t.img, "test_con_hermana": t[t.con_hermana].img, "test_sin_hermana": t[~t.con_hermana].img,
        "validacion": d[d.split == "valid"].img}
m = YOLO(str(MOD)); res = []
for k, imgs in conj.items():
    r = m.val(data=str(lista(k, imgs)), split="val", imgsz=640, batch=16, device="cpu", plots=False, verbose=False,
              project=str(W / "val"), name=k, exist_ok=True)
    ap = {nombres[int(c)]: r.box.ap50[i] for i, c in enumerate(r.box.ap_class_index)}
    sin_sana = [v for c, v in ap.items() if c != "sana"]
    res.append(dict(conjunto=k, imagenes=len(imgs), mAP50=r.box.map50, mAP50_95=r.box.map, mAP50_sin_sana=sum(sin_sana) / len(sin_sana)))
R = pd.DataFrame(res); R.to_csv(W / "resultados_fuga_v1.csv", index=False)
txt = (f"Grupos de captura: {d.grupo.nunique()}; grupos repartidos entre subconjuntos: {(sp.apply(len) > 1).sum()}\n"
       f"Prueba v1: {len(t)} imagenes; con foto hermana en entrenamiento/validacion: {t.con_hermana.sum()}\n\n"
       + (R.assign(**{c: (R[c] * 100).round(1) for c in ["mAP50", "mAP50_95", "mAP50_sin_sana"]})).to_string(index=False))
(W / "resumen_fuga_v1.txt").write_text(txt, encoding="utf-8"); print(txt)
