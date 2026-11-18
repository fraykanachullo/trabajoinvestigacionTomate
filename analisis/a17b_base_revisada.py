"""A17b. Reconstruye el fuente de la version que reviso el ingeniero (PDF del 13/11, 13:15).

main_antes_revision.tex (13/11, 15:23) agrego despues los links de reproducibilidad. Aqui se quitan esas
adiciones para que la base del 'cambios en azul' sea exactamente la version revisada.
Salida: main_revisado_ing.tex
"""
import re
from pathlib import Path

D = Path(r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final")
s = (D / "main_antes_revision.tex").read_text(encoding="utf-8")
frase = " El \\emph{dataset}, los \\emph{scripts} y los modelos son públicos para que el trabajo pueda reproducirse."
assert frase in s
s = s.replace(frase, "")
assert "corregida, publicado con licencia abierta." in s
s = s.replace("corregida, publicado con licencia abierta.", "corregida.")
i = s.index("\\subsection{Disponibilidad de los Datos y Reproducibilidad}")
j = s.index("\\subsection{Etapa v1: Resultados Preliminares}")
k = s.rfind("%====", i, j)
s = s[:i] + (s[k:j] if k > 0 else "") + s[j:]
m = re.search(r"\\dataavailability\{.*?\}\n", s, re.S)
s = s[:m.start()] + "\\dataavailability{Los datos están disponibles a solicitud del autor de correspondencia.}\n" + s[m.end():]
m = re.search(r"\\bibitem\[Kana Chullo\(2026\)\]\{Dataset2026\}.*?(?=\n\s*\n|\\end\{thebibliography\})", s, re.S)
s = s[:m.start()] + s[m.end():]
(D / "main_revisado_ing.tex").write_text(s, encoding="utf-8")
print("ok")
