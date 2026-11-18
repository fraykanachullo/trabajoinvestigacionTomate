"""A18. Ubica pagina y linea de cada cambio en el PDF con cambios en azul (main_cambios_azul.pdf)."""
import json
import re
import subprocess

PDF = r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final/main_cambios_azul.pdf"
ANCLAS = {
    "1": [("Título", "Bajo Alta Radiaci"), ("objetivo y P3–P4", "Se plantearon cinco preguntas"),
          ("métodos de estrés y mitigación", "Prueba de estrés por sobreexposición"), ("resultado P3", "La prueba de estrés sí separa"),
          ("resultado P4", "Mitigación de la Sobreexposición"), ("aviso de luz", "no apareció en ninguna de las 180"),
          ("discusión", "La sobreexposición degrada al detector")],
    "2": [("medición con el modelo v1", "Para medirlo con el mismo modelo"), ("limitaciones (v)", "experimento controlado de fuga")],
    "3": [("Tabla 5 y texto", "comparan el modelo de la v4 con los de la v5"), ("v6", "Con las etiquetas corregidas y cuatro clases"),
          ("discusión", "Ampliar el conjunto mejoró la generalización")],
    "5": [("definición de grupo", "Se definió como"), ("auditoría pHash", "856")],
    "6": [("métodos", "bootstrap agrupado por grupo de captura"), ("Wilcoxon", "Wilcoxon")],
    "7": [("métodos", "tasa de falso sano"), ("Tabla 7", "Métricas por imagen en la prueba completa")],
    "8": [("despliegue v6", "reemplazó a la v5 en el servidor"), ("latencia v6", "574")],
    "9": [("sección y Tabla 1", "Trabajos Relacionados")],
    "10": [("experiencia del anotador", "tiene siete años de experiencia"), ("regla de 16 px y polilla", "Esta regla de tamaño se aplicó"),
           ("corrección v6", "se revisaron a mano las 319"), ("protocolo de acuerdo", "Acuerdo entre anotadores y validación"),
           ("auditoría", "Para buscar errores de etiquetado")],
    "11": [("disponibilidad", "Disponibilidad y Reproducibilidad"), ("Tabla 4", "Recursos públicos")],
    "12": [("preguntas", "Se plantearon cinco preguntas"), ("limitaciones", "Limitaciones")],
    "13": [("Fig. 2", "la v5 iguala a la v4"), ("Fig. 4", "píxeles saturados por franja"), ("Fig. 5", "Prueba de estrés: las 180"),
           ("Fig. 7", "Ejemplos de la auditoría")],
    "14": [("96 de 96", "96 de 96")],
    "15": [("referencias", "ACCESS.2024.3403491")],
    "17": [("título", "Bajo Alta Radiaci"), ("resumen", "La alta radiación solar de las zonas"), ("palabras clave", "radiación solar; sobreexposición")],
    "Ap. 1": [("hoja sana eliminada", "se eliminó la clase hoja sana")],
    "Ap. 2": [("etapa v1", "En la v1, dividida por imagen")],
    "Ap. 3": [("elección de YOLOv8n", "Se mantiene YOLOv8n porque gana")],
    "Ap. 4": [("solo nano", "modelos nano")],
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
                        return f"pág. {p}, l. {m.group(1)}"
                return f"pág. {p}"
    return None


res = {}
for obs, lst in ANCLAS.items():
    partes = []
    for nombre, frase in lst:
        u = ubicar(frase)
        partes.append(f"{nombre}: {u}" if u else f"{nombre}: (no ubicado)")
    res[obs] = "; ".join(partes)
    print(obs, "->", res[obs])
json.dump(res, open(r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final/paginas_lineas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
