"""A17c. Version en ingles con los cambios en azul.

La version revisada estaba en espanol (main_revisado_ing.tex). Se marcan en azul, en main_en.tex, las mismas
partes que cambiaron en la version en espanol (main.tex): ambos manuscritos tienen los mismos bloques en el mismo
orden; en cada parrafo, si el numero de oraciones coincide se marca oracion por oracion, y si no, el parrafo completo
cuando tiene alguna oracion nueva. Salida: main_en_cambios_azul.tex
"""
import re
from pathlib import Path

D = Path(r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final")
viejo = (D / "main_revisado_ing.tex").read_text(encoding="utf-8")
es = (D / "main.tex").read_text(encoding="utf-8")
en = (D / "main_en.tex").read_text(encoding="utf-8")
AZ = "\\color{blue}"
norm = lambda s: re.sub(r"\s+", " ", s).strip()


def oraciones(texto, ingles=False):
    may = r"[A-Z(\\$0-9]" if ingles else r"[A-ZÁÉÍÓÚÑ¿(\\$0-9]"
    partes = re.split(r"(?<=[.;:?!])\s+(?=" + may + ")", texto.strip())
    out, buf = [], ""
    for p in partes:
        buf = (buf + " " + p).strip() if buf else p
        if buf.count("{") == buf.count("}") and buf.count("$") % 2 == 0:
            out.append(buf); buf = ""
    if buf:
        out.append(buf)
    return out


cuerpo_viejo = norm(viejo)
oraciones_viejas = {norm(o) for b in re.split(r"\n\s*\n", viejo) for o in oraciones(b)}
bloques_viejos = {norm(b) for b in re.split(r"\n\s*\n", viejo)}
es_nueva = lambda o: norm(o) not in oraciones_viejas and norm(o) not in cuerpo_viejo


def arg(b, cmd):
    i = b.find("\\" + cmd + "{"); j = i + len(cmd) + 2
    return b[:j] + AZ + " " + b[j:]


def separar(t):
    i = t.index("\\begin{document}")
    return t[:i], [x for x in re.split(r"\n\s*\n", t[i:])]


pre_es, bl_es = separar(es)
pre_en, bl_en = separar(en)
print("bloques es/en:", len(bl_es), len(bl_en))
# Preambulo: titulo, resumen y palabras clave cambiaron todos
for cmd in ["Title", "abstract", "keyword"]:
    m = re.search(r"\\" + cmd + r"\{.*?\}\n", pre_en, re.S)
    pre_en = pre_en[:m.start()] + arg(m.group(0), cmd) + pre_en[m.end():]
pre_en = pre_en.replace("\\usetikzlibrary{positioning,arrows.meta}", "\\usetikzlibrary{positioning,arrows.meta}\n\\usepackage{xcolor}", 1)

ENT = ("\\begin{table}", "\\begin{figure}")
out = []
for k, b_en in enumerate(bl_en):
    b_es = bl_es[k] if k < len(bl_es) and len(bl_es) == len(bl_en) else None
    s_en = b_en.strip()
    if b_es is None:
        out.append(b_en); continue
    s_es = b_es.strip()
    if not s_en or s_en.startswith("%") or norm(b_es) in bloques_viejos:
        out.append(b_en); continue
    if s_en.startswith(ENT):
        b_en = re.sub(r"(\\begin\{(table|figure)\}(\[[^\]]*\])?)", lambda m: m.group(1) + AZ, b_en, count=1)
    elif s_en.startswith(("\\section", "\\subsection")):
        if norm(s_es) not in cuerpo_viejo:
            b_en = re.sub(r"(\\(sub)?section\{)", lambda m: m.group(1) + AZ + " ", b_en, count=1)
    elif s_en.startswith("\\begin{itemize}"):
        b_en = "{" + AZ + "\n" + b_en + "\n}"
    elif s_en.startswith("\\bibitem"):
        m = re.match(r"(\\bibitem(\[[^\]]*\])?\{[^}]*\}\s*)(.*)", b_en, re.S)
        if m:
            b_en = m.group(1) + "{" + AZ + " " + m.group(3).strip() + "}"
    elif s_en.startswith(("\\begin{adjustwidth}", "\\end{", "\\begin{thebibliography}", "\\vspace", "\\begin{document}")):
        pass
    elif re.match(r"\\(authorcontributions|funding|institutionalreview|informedconsent|dataavailability|acknowledgments|conflictsofinterest)\{", s_en):
        b_en = arg(b_en, re.match(r"\\(\w+)\{", s_en).group(1))
    else:
        cola = ""
        if "\\begin{itemize}" in b_en:
            i = b_en.index("\\begin{itemize}"); cola = "\n{" + AZ + "\n" + b_en[i:] + "\n}"; b_en = b_en[:i]
            if "\\begin{itemize}" in b_es:
                b_es = b_es[:b_es.index("\\begin{itemize}")]
        o_es, o_en = oraciones(b_es), oraciones(b_en, True)
        flags = [es_nueva(o) for o in o_es]
        if len(o_es) == len(o_en):
            b_en = " ".join(("{" + AZ + " " + o + "}") if f else o for o, f in zip(o_en, flags))
        elif any(flags):
            b_en = " ".join("{" + AZ + " " + o + "}" for o in o_en)
        b_en += cola
    out.append(b_en)

(D / "main_en_cambios_azul.tex").write_text(pre_en + "\n\n".join(out), encoding="utf-8")
print("ok")
