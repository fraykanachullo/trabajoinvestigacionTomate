"""A14. Hoja de validacion para un ingeniero agronomo (obs. 10): 50 fotos de la muestra (10 por estrato),
con casillas por clase. Genera validacion_ingeniero/hoja_validacion.tex (se compila a PDF) y la plantilla
respuestas_ingeniero.csv para pasar sus respuestas."""
import csv, random
from pathlib import Path
from PIL import Image, ImageOps
D = Path(r"D:/ProyectoInvestigacion/acuerdo_anotadores"); V = D / "validacion_ingeniero"
filas = list(csv.DictReader(open(D / "muestra.csv", encoding="utf-8")))
random.seed(1); sel = []
for e in ["mildiu", "mosca_blanca", "oidio", "polilla", "sin_plaga"]:
    c = [f for f in filas if f["estrato"] == e]; random.shuffle(c); sel += c[:10]
sel.sort(key=lambda f: f["id"])
for i, f in enumerate(sel, 1):
    src = next((D / "imagenes").glob(f["id"] + ".*"))
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB"); im.thumbnail((1400, 1400)); im.save(V / "img" / f"V{i:02d}.jpg", quality=88)
    f["vid"] = f"V{i:02d}"
caja = r"$\square$~"
op = (caja + "Mildiu \quad " + caja + "Oídio \quad " + caja + "Mosca blanca \quad " + caja + "Polilla del tomate \quad "
      + caja + "Ninguna (sana) \quad " + caja + "No se puede saber")
cuerpo = []
for k in range(0, len(sel), 2):
    for f in sel[k:k + 2]:
        BR = chr(92) * 2  # salto de linea de LaTeX
        cuerpo.append(r"\noindent\textbf{Foto %s}" % f["vid"] + BR + "[2pt]"
                      + r"\centerline{\includegraphics[height=0.36\textheight,width=\linewidth,keepaspectratio]{img/%s.jpg}}" % f["vid"]
                      + BR + "[2pt]\n" + r"{\small " + op + "}" + BR + "[3pt]" + r"{\small Comentario: \hrulefill}" + "\n" + r"\vspace{6pt}" + "\n")
    cuerpo.append(r"\newpage")
tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[spanish]{babel}\usepackage{amssymb}
\usepackage[margin=1.6cm]{geometry}\usepackage{graphicx}\usepackage{lmodern}\pagestyle{plain}
\begin{document}
\section*{Validación de clases por especialista}
\textbf{Proyecto:} detección de plagas y enfermedades del tomate en el Valle de Majes (Universidad Peruana Unión).\par\smallskip
\textbf{Qué se pide:} para cada una de las 50 fotos, marque \textbf{todas} las plagas o enfermedades que vea con claridad (puede marcar más de una). Si la hoja no tiene ninguna, marque ``Ninguna''. Si la foto no permite decidir, marque ``No se puede saber''. No hace falta señalar dónde están. Tiempo estimado: 40--60 minutos.\par\smallskip
\textbf{Clases:} \emph{mildiu} (\emph{Phytophthora infestans}); \emph{oídio} (polvo blanco); \emph{mosca blanca} (\emph{Bemisia tabaci}); \emph{polilla del tomate} (\emph{Tuta absoluta}: insecto o minas en la hoja).\par\bigskip
Nombre: \hrulefill\quad Profesión / CIP: \hrulefill\par\medskip
Años de experiencia en tomate: \hrulefill\quad Fecha: \hrulefill\quad Firma: \hrulefill
\newpage
""" + "\n".join(cuerpo) + r"\end{document}"
(V / "hoja_validacion.tex").write_text(tex, encoding="utf-8")
with open(V / "respuestas_ingeniero.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["foto", "id_muestra", "mildiu", "oidio", "mosca_blanca", "polilla", "ninguna", "no_se_puede", "comentario"])
    for f in sel: w.writerow([f["vid"], f["id"], "", "", "", "", "", "", ""])
print(len(sel), "fotos")
