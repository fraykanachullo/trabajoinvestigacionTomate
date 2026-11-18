"""A12. Muestra estratificada para medir el acuerdo entre anotadores (obs. 10).

100 imagenes del dataset v6 (Roboflow version 3): 20 con polilla, 20 con mosca blanca, 20 con oidio,
25 con mildiu y 15 sin plaga, repartidas entre las tres sesiones y con una sola imagen por grupo de captura
(misma carpeta y menos de 30 s), para no repetir la misma planta. Semilla fija.
Salida: acuerdo_anotadores/imagenes/ (sin etiquetas, renombradas A001..A100), referencia/labels (etiquetas
del dataset, solo para calificar), muestra.csv (correspondencia) y clases.txt.
"""
import csv, random, re, sys, zipfile, collections
from pathlib import Path
ZIP = Path(sys.argv[1]); OUT = Path(r"D:/ProyectoInvestigacion/acuerdo_anotadores")
CL = ["mildiu", "mosca_blanca", "oidio", "polilla"]
random.seed(0)
z = zipfile.ZipFile(ZIP)
lbl = {n: z.read(n).decode() for n in z.namelist() if n.endswith(".txt") and "/labels/" in n}
info = []
for n, t in lbl.items():
    stem = Path(n).stem
    base = re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", stem, flags=re.I)
    fecha = (re.search(r"2026(\d{4})_(\d{6})", base) or [None])
    if not fecha: continue
    m = re.search(r"2026(\d{4})_(\d{2})(\d{2})(\d{2})", base)
    ses = m.group(1); seg = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    carpeta = base.split("__")[:-1]
    cls = {int(l.split()[0]) for l in t.splitlines() if l.strip()}
    img = [x for x in z.namelist() if x.startswith(n.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".")]
    info.append(dict(lbl=n, img=img[0], base=base, ses=ses, seg=seg, carpeta="__".join(carpeta), cls=cls))
# grupos de captura: misma carpeta y < 30 s
info.sort(key=lambda d: (d["carpeta"], d["ses"], d["seg"]))
g = -1; prev = None
for d in info:
    if not prev or d["carpeta"] != prev["carpeta"] or d["ses"] != prev["ses"] or d["seg"] - prev["seg"] >= 30: g += 1
    d["grupo"] = g; prev = d
cuotas = [(3, 20), (1, 20), (2, 20), (0, 25), (None, 15)]
usados_g, elegidas = set(), []
for c, k in cuotas:
    cand = [d for d in info if (c in d["cls"] if c is not None else not d["cls"]) and d["grupo"] not in usados_g]
    random.shuffle(cand)
    por_ses = collections.defaultdict(list)
    for d in cand: por_ses[d["ses"]].append(d)
    tomadas = []
    while len(tomadas) < k and any(por_ses.values()):
        for s in sorted(por_ses):
            while por_ses[s] and por_ses[s][0]["grupo"] in usados_g: por_ses[s].pop(0)
            if por_ses[s] and len(tomadas) < k:
                d = por_ses[s].pop(0); tomadas.append(d); usados_g.add(d["grupo"])
    for d in tomadas: d["estrato"] = CL[c] if c is not None else "sin_plaga"
    elegidas += tomadas
random.shuffle(elegidas)
(OUT / "imagenes").mkdir(parents=True, exist_ok=True); (OUT / "referencia" / "labels").mkdir(parents=True, exist_ok=True)
with open(OUT / "muestra.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "imagen_original", "sesion", "estrato", "clases_referencia"])
    for i, d in enumerate(elegidas, 1):
        nid = f"A{i:03d}"; ext = Path(d["img"]).suffix
        (OUT / "imagenes" / (nid + ext)).write_bytes(z.read(d["img"]))
        (OUT / "referencia" / "labels" / (nid + ".txt")).write_text(lbl[d["lbl"]])
        w.writerow([nid, d["base"], d["ses"], d["estrato"], " ".join(CL[c] for c in sorted(d["cls"]))])
(OUT / "clases.txt").write_text("\n".join(CL) + "\n")
print("elegidas", len(elegidas), collections.Counter(d["estrato"] for d in elegidas), collections.Counter(d["ses"] for d in elegidas))
cajas = collections.Counter(); [cajas.update(int(l.split()[0]) for l in lbl[d["lbl"]].splitlines() if l.strip()) for d in elegidas]
print("cajas de referencia", {CL[k]: v for k, v in sorted(cajas.items())})
