"""A14b. Acuerdo entre el ingeniero agronomo y las etiquetas del dataset v6 (obs. 10).

Uso: python a14b_calificar_ingeniero.py [respuestas_ingeniero.csv]
En el CSV, marcar con "x" (o 1) cada clase que el ingeniero marco en la hoja. Las fotos con
"no_se_puede" marcado se excluyen y se reportan aparte.
Salida: validacion_ingeniero/resultados_ingeniero.txt
"""
import csv
import sys
from pathlib import Path

import numpy as np

D = Path(r"D:/ProyectoInvestigacion/acuerdo_anotadores")
V = D / "validacion_ingeniero"
CL = ["mildiu", "mosca_blanca", "oidio", "polilla"]
NOMBRE = {"mildiu": "Mildiu", "mosca_blanca": "Mosca blanca", "oidio": "Oidio", "polilla": "Polilla"}

archivo = Path(sys.argv[1]) if len(sys.argv) > 1 else V / "respuestas_ingeniero.csv"
marca = lambda v: str(v).strip().lower() in {"x", "1", "si", "sí", "true"}


def referencia(id_muestra):
    f = D / "referencia" / "labels" / f"{id_muestra}.txt"
    return {CL[int(l.split()[0])] for l in f.read_text().splitlines() if l.strip()}


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), po


filas = list(csv.DictReader(open(archivo, encoding="utf-8")))
llenas = [f for f in filas if any(marca(f[c]) for c in CL + ["ninguna", "no_se_puede"])]
dudosas = [f for f in llenas if marca(f["no_se_puede"])]
validas = [f for f in llenas if not marca(f["no_se_puede"])]
out = [f"Fotos respondidas: {len(llenas)} de {len(filas)}; 'no se puede saber': {len(dudosas)}; analizadas: {len(validas)}"]
ing_todo, ref_todo = [], []
for c in CL:
    ing = [marca(f[c]) for f in validas]
    ref = [c in referencia(f["id_muestra"]) for f in validas]
    ing_todo += ing
    ref_todo += ref
    k, po = kappa(ing, ref)
    out.append(f"{NOMBRE[c]:13} kappa {k:5.2f}  acuerdo {po * 100:5.1f}%  (ingeniero {sum(ing)}, dataset {sum(ref)})")
k, po = kappa(ing_todo, ref_todo)
out.append(f"{'Todas':13} kappa {k:5.2f}  acuerdo {po * 100:5.1f}%  (imagen x clase)")
ing = [any(marca(f[c]) for c in CL) for f in validas]
ref = [bool(referencia(f["id_muestra"])) for f in validas]
k, po = kappa(ing, ref)
out.append(f"{'Con plaga':13} kappa {k:5.2f}  acuerdo {po * 100:5.1f}%  (con plaga / sin plaga)")
exacta = sum({c for c in CL if marca(f[c])} == referencia(f["id_muestra"]) for f in validas)
out.append(f"Coincidencia exacta del conjunto de clases: {exacta} de {len(validas)} fotos")
txt = "\n".join(out)
print(txt)
(V / "resultados_ingeniero.txt").write_text(txt, encoding="utf-8")
