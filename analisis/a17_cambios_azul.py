"""A17. Articulo con los cambios en azul respecto a la version entregada antes de la revision.

Compara main_antes_revision.tex (version anterior) con main.tex (version corregida):
- parrafos de texto: cada oracion que no existia antes se pinta de azul;
- tablas, figuras, listas, resumen, titulo y referencias nuevas o modificadas: bloque completo en azul;
- lo eliminado no se muestra (solo se marca lo nuevo o cambiado).
Salida: main_cambios_azul.tex
"""
import re
from pathlib import Path

D = Path(r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final")
viejo = (D / "main_revisado_ing.tex").read_text(encoding="utf-8")
nuevo = (D / "main.tex").read_text(encoding="utf-8")
AZ = "\\color{blue}"
norm = lambda s: re.sub(r"\s+", " ", s).strip()


def oraciones(texto):
    partes = re.split(r"(?<=[.;:?!])\s+(?=[A-ZÁÉÍÓÚÑ¿(\\$0-9])", texto.strip())
    out, buf = [], ""
    for p in partes:
        buf = (buf + " " + p).strip() if buf else p
        if buf.count("{") == buf.count("}") and buf.count("$") % 2 == 0:
            out.append(buf); buf = ""
    if buf:
        out.append(buf)
    return out


cuerpo_viejo = norm(viejo)
oraciones_viejas = set()
for bloque in re.split(r"\n\s*\n", viejo):
    for o in oraciones(bloque):
        oraciones_viejas.add(norm(o))
bloques_viejos = {norm(b) for b in re.split(r"\n\s*\n", viejo)}


def es_nueva(o):
    n = norm(o)
    return n not in oraciones_viejas and n not in cuerpo_viejo


def pintar_parrafo(b):
    os_ = oraciones(b)
    return " ".join(("{" + AZ + " " + o + "}") if es_nueva(o) else o for o in os_)


def pintar_argumento(b, comando):
    # \comando{...}: pinta todo el argumento
    i = b.find("\\" + comando + "{")
    j = i + len(comando) + 2
    return b[:j] + AZ + " " + b[j:]


ini = nuevo.index("\\begin{document}")
pre, cuerpo = nuevo[:ini], nuevo[ini:]
# Preambulo: titulo, resumen y palabras clave
for cmd in ["Title", "abstract", "keyword"]:
    m = re.search(r"\\" + cmd + r"\{.*?\}\n", pre, re.S)
    if m and norm(m.group(0)) not in cuerpo_viejo:
        pre = pre[:m.start()] + pintar_argumento(m.group(0), cmd) + pre[m.end():]
pre = pre.replace("\\begin{document}", "")

ENTORNOS = ("\\begin{table}", "\\begin{figure}")
salida = []
for b in re.split(r"(\n\s*\n)", cuerpo):
    s = b.strip()
    if not s or re.fullmatch(r"\n\s*\n", b) or s.startswith("%"):
        salida.append(b); continue
    if norm(b) in bloques_viejos:
        salida.append(b); continue
    if s.startswith(ENTORNOS):
        b = re.sub(r"(\\begin\{(table|figure)\}(\[[^\]]*\])?)", lambda m: m.group(1) + AZ, b, count=1)
    elif s.startswith(("\\section", "\\subsection")):
        b = re.sub(r"(\\(sub)?section\{)", lambda m: m.group(1) + AZ + " ", b, count=1) if norm(s) not in cuerpo_viejo else b
    elif s.startswith("\\begin{itemize}"):
        b = "{" + AZ + "\n" + b + "\n}"
    elif s.startswith("\\bibitem"):
        m = re.match(r"(\\bibitem(\[[^\]]*\])?\{[^}]*\}\s*)(.*)", b, re.S)
        b = m.group(1) + "{" + AZ + " " + m.group(3).strip() + "}" if m else b
    elif s.startswith(("\\begin{adjustwidth}", "\\end{", "\\begin{thebibliography}", "\\vspace", "\\begin{document}")):
        pass
    elif re.match(r"\\(authorcontributions|funding|institutionalreview|informedconsent|dataavailability|acknowledgments|conflictsofinterest)\{", s):
        cmd = re.match(r"\\(\w+)\{", s).group(1)
        b = pintar_argumento(b, cmd)
    elif s.startswith("\\textbf{") or not s.startswith("\\") or s.startswith(("\\citet", "\\noindent")):
        # parrafo de texto (puede contener un itemize al final)
        if "\\begin{itemize}" in b:
            k = b.index("\\begin{itemize}")
            b = pintar_parrafo(b[:k]) + "\n{" + AZ + "\n" + b[k:] + "\n}"
        else:
            b = pintar_parrafo(b)
    salida.append(b)

doc = pre + "\\begin{document}" + "".join(salida)[len("\\begin{document}"):] if "".join(salida).lstrip().startswith("\\begin{document}") else pre + "".join(salida)
# xcolor para \color
doc = doc.replace("\\usetikzlibrary{positioning,arrows.meta}", "\\usetikzlibrary{positioning,arrows.meta}\n\\usepackage{xcolor}", 1)
(D / "main_cambios_azul.tex").write_text(doc, encoding="utf-8")
print("ok")
