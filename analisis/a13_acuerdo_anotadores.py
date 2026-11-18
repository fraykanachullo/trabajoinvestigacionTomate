"""A13. Acuerdo entre anotadores (obs. 10): kappa por imagen y F1 por caja.

Uso: python a13_acuerdo_anotadores.py anotador_A_fray.zip anotador_B_hiram.zip
     (zip exportado de makesense.ai en formato YOLO, o carpeta con A001.txt ... A100.txt)
Compara A-B (acuerdo entre anotadores) y cada uno con las etiquetas del dataset v6 (referencia/labels).
Salida: acuerdo_anotadores/resultados_acuerdo.csv y resultados_acuerdo.txt
"""
import sys, zipfile, csv
from pathlib import Path
import numpy as np
D = Path(r"D:/ProyectoInvestigacion/acuerdo_anotadores")
CL = ["mildiu", "mosca_blanca", "oidio", "polilla"]
IDS = [r["id"] for r in csv.DictReader(open(D / "muestra.csv", encoding="utf-8"))]

def cargar(fuente):
    p = Path(fuente) if Path(fuente).is_absolute() else D / fuente
    txt = {}
    if p.suffix == ".zip":
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            if n.endswith(".txt") and Path(n).stem in IDS: txt[Path(n).stem] = z.read(n).decode()
    else:
        for f in p.glob("*.txt"):
            if f.stem in IDS: txt[f.stem] = f.read_text()
    cajas = {}
    for i in IDS:
        b = []
        for l in txt.get(i, "").splitlines():
            v = l.split()
            if len(v) >= 5:
                c, x, y, w, h = int(v[0]), *map(float, v[1:5])
                b.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        cajas[i] = b
    return cajas, len(txt)

def iou(a, b):
    iw = max(0, min(a[3], b[3]) - max(a[1], b[1])); ih = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    i = iw * ih; u = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - i
    return i / u if u > 0 else 0

def kappa(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool); po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po

def comparar(A, B, nA, nB):
    fil = []
    pres = lambda C, c: [any(b[0] == c for b in C[i]) for i in IDS]
    todos_a, todos_b = [], []
    for c in range(4):
        a, b = pres(A, c), pres(B, c); k, po = kappa(a, b); todos_a += a; todos_b += b
        fil.append(dict(nivel="imagen", clase=CL[c], kappa=k, acuerdo=po, n_A=sum(a), n_B=sum(b)))
    k, po = kappa(todos_a, todos_b)
    fil.append(dict(nivel="imagen", clase="todas (imagen x clase)", kappa=k, acuerdo=po, n_A=sum(todos_a), n_B=sum(todos_b)))
    a = [bool(A[i]) for i in IDS]; b = [bool(B[i]) for i in IDS]; k, po = kappa(a, b)
    fil.append(dict(nivel="imagen", clase="con plaga / sin plaga", kappa=k, acuerdo=po, n_A=sum(a), n_B=sum(b)))
    for t in (0.5, 0.3):
        tot = [0, 0, 0]
        for c in range(4):
            m = na = nb = 0
            for i in IDS:
                ca = [x for x in A[i] if x[0] == c]; cb = [x for x in B[i] if x[0] == c]; na += len(ca); nb += len(cb)
                pares = sorted(((iou(x, y), j, k2) for j, x in enumerate(ca) for k2, y in enumerate(cb)), reverse=True)
                ua, ub = set(), set()
                for v, j, k2 in pares:
                    if v < t: break
                    if j not in ua and k2 not in ub: ua.add(j); ub.add(k2); m += 1
            f1 = 2 * m / (na + nb) if na + nb else np.nan
            fil.append(dict(nivel=f"caja IoU>={t}", clase=CL[c], f1=f1, emparejadas=m, n_A=na, n_B=nb))
            tot = [tot[0] + m, tot[1] + na, tot[2] + nb]
        fil.append(dict(nivel=f"caja IoU>={t}", clase="todas", f1=2 * tot[0] / (tot[1] + tot[2]), emparejadas=tot[0], n_A=tot[1], n_B=tot[2]))
    return fil

R, nR = cargar(D / "referencia" / "labels")
fuentes = {"A": sys.argv[1], "B": sys.argv[2] if len(sys.argv) > 2 else None}
an = {k: cargar(v) for k, v in fuentes.items() if v}
pares = []
if "B" in an: pares.append(("A vs B (entre anotadores)", an["A"], an["B"]))
pares += [(f"{k} vs dataset v6", an[k], (R, nR)) for k in an]
filas, out = [], []
for nom, (A, nA), (B, nB) in pares:
    out.append(f"\n== {nom}  (archivos: {nA} / {nB} de {len(IDS)}) ==")
    for f in comparar(A, B, nA, nB):
        f["comparacion"] = nom; filas.append(f)
        if f["nivel"] == "imagen":
            out.append(f"{f['nivel']:12} {f['clase']:26} kappa {f['kappa']:.2f}  acuerdo {f['acuerdo']*100:5.1f}%  (n {f['n_A']}/{f['n_B']})")
        else:
            out.append(f"{f['nivel']:12} {f['clase']:26} F1 {f['f1']:.2f}  emparejadas {f['emparejadas']} de {f['n_A']}/{f['n_B']}")
txt = "\n".join(out); print(txt)
(D / "resultados_acuerdo.txt").write_text(txt, encoding="utf-8")
with open(D / "resultados_acuerdo.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["comparacion", "nivel", "clase", "kappa", "acuerdo", "f1", "emparejadas", "n_A", "n_B"]); w.writeheader(); w.writerows(filas)
