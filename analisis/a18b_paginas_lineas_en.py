"""A18. Ubica pagina y linea de cada cambio en el PDF con cambios en azul (main_en_cambios_azul.pdf)."""
import json
import re
import subprocess

PDF = r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final/main_en_cambios_azul.pdf"
ANCLAS = {
    "1": [("Título", "under High Solar Radiation"), ("objetivo y P3–P4", "Five research questions were posed"),
          ("métodos de estrés y mitigación", "Overexposure stress test"), ("resultado P3", "The stress test does separate"),
          ("resultado P4", "Mitigation of Overexposure"), ("aviso de luz", "did not appear in any of the 180"),
          ("discusión", "Overexposure degrades the detector")],
    "2": [("medición con el modelo v1", "To measure it with the same model"), ("limitaciones (v)", "controlled leakage experiment")],
    "3": [("Tabla 5 y texto", "compare the v4 model with the v5 models"), ("v6", "With the corrected labels and four classes"),
          ("discusión", "Enlarging the dataset improved generalization")],
    "5": [("definición de grupo", "was defined as the set"), ("auditoría pHash", "856")],
    "6": [("métodos", "clustered by capture group"), ("Wilcoxon", "Wilcoxon")],
    "7": [("métodos", "false-healthy rate"), ("Tabla 7", "Image-level metrics on the full test")],
    "8": [("despliegue v6", "replaced v5"), ("latencia v6", "574")],
    "9": [("sección y Tabla 1", "Related Work")],
    "10": [("experiencia del anotador", "has seven years of experience"), ("regla de 16 px y polilla", "This size rule was applied"),
           ("corrección v6", "the 319 flagged images were reviewed"), ("protocolo de acuerdo", "Inter-annotator agreement and external"),
           ("auditoría", "To find labeling errors")],
    "11": [("disponibilidad", "Availability and Reproducibility"), ("Tabla 4", "Public resources")],
    "12": [("preguntas", "Five research questions were posed"), ("limitaciones", "Limitations")],
    "13": [("Fig. 2", "v5 matches v4"), ("Fig. 4", "saturated pixels by time band"), ("Fig. 5", "Stress test: the 180"),
           ("Fig. 7", "Audit examples")],
    "14": [("96 of 96", "96 of 96")],
    "15": [("referencias", "ACCESS.2024.3403491")],
    "17": [("título", "under High Solar Radiation"), ("resumen", "High solar radiation in arid regions"), ("palabras clave", "solar radiation; overexposure")],
    "Ap. 1": [("hoja sana eliminada", "healthy-leaf class was removed")],
    "Ap. 2": [("etapa v1", "In v1, split by image")],
    "Ap. 3": [("elección de YOLOv8n", "YOLOv8n is kept because it wins")],
    "Ap. 4": [("solo nano", "nano models")],
}
paginas = []
n = 1
while True:
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", "-f", str(n), "-l", str(n), PDF, "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout.strip() and n > 20:
        break
    t = r.stdout.decode("utf-8", "replace")
    if not t.strip():
        break
    paginas.append(t.splitlines()); n += 1


def ubicar(frase):
    clave = re.sub(r"\s+", " ", frase).lower()
    for p, lineas in enumerate(paginas, 1):
        for k, l in enumerate(lineas):
            if clave in re.sub(r"\s+", " ", l).lower():
                for j in range(k, min(k + 4, len(lineas))):
                    m = re.search(r"\s(\d{1,3})\s*$", lineas[j])
                    if m:
                        return f"p. {p}, l. {m.group(1)}"
                return f"p. {p}"
    return None


res = {}
for obs, lst in ANCLAS.items():
    partes = []
    for nombre, frase in lst:
        u = ubicar(frase)
        partes.append(f"{nombre}: {u}" if u else f"{nombre}: (no ubicado)")
    res[obs] = "; ".join(partes)
    print(obs, "->", res[obs])
json.dump(res, open(r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final/paginas_lineas_en.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
