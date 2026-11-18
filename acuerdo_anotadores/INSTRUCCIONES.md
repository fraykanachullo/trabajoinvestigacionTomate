# Prueba de acuerdo entre anotadores (observación 10 del revisor)

**Objetivo:** medir si dos personas, trabajando por separado, etiquetan igual las mismas fotos.
- Si coinciden, las etiquetas son confiables. Si no, se sabe en qué clase fallan.
- Es lo que pide el revisor ("kappa entre anotadores").

**Quiénes:** Fray (anotador A) y Hiram (anotador B), **cada uno por su cuenta**.
- No mirar las etiquetas de Roboflow ni las del compañero.
- No comentar las fotos entre ustedes hasta terminar los dos.

**Qué:** las 100 fotos de la carpeta `imagenes/` (A001 … A100), con las 4 clases de `clases.txt`. Toma unas 2–3 horas.

## Herramienta: makesense.ai
Gratis, sin cuenta. Las fotos no salen de tu computadora.

1. Abrir **https://www.makesense.ai** → *Get Started*.
2. Arrastrar las 100 fotos de `imagenes/` → *Object Detection*.
3. En *Create labels* → *Load labels from file* → elegir `clases.txt`.
   - Deben quedar 4: `mildiu`, `mosca_blanca`, `oidio`, `polilla`, **en ese orden**.
4. Para cada foto, elegir la herramienta **Rect**, dibujar cada caja y asignarle su clase.
5. Al terminar: *Actions* → *Export Annotations* → **"A .zip package containing files in YOLO format"**.
6. Renombrar el zip:
   - Fray: `anotador_A_fray.zip`
   - Hiram: `anotador_B_hiram.zip`
7. Guardarlo en esta carpeta (`acuerdo_anotadores/`).

Consejo: makesense no guarda solo. Si vas a parar a mitad, exporta el zip y luego vuelve a cargarlo con
*Actions → Import Annotations*.

## Reglas (las mismas del artículo; aplicarlas igual en todas las fotos)
1. **Una caja por lesión, insecto o mina.**
   - **Mildiu:** cada mancha o lesión.
   - **Oídio:** cada zona de polvo blanco.
   - **Mosca blanca:** cada insecto. Si están amontonados y no se separan, una caja para el grupo.
   - **Polilla del tomate** (*Tuta absoluta*): el insecto o la mina/galería en la hoja.
2. **Solo lo reconocible del primer plano.** No etiquetar plantas lejanas ni zonas borrosas.
3. **Hojas superpuestas:** una caja por cada lesión, aunque las hojas se crucen.
4. **No etiquetar objetos muy pequeños**, menores a 16 px cuando la foto se ve a 640 px. Como guía, algo
   que en pantalla completa mide menos de un grano de arroz.
5. **Hojas sanas: no se etiquetan.** Una foto sin plagas queda **sin cajas**, y eso también es una respuesta.
6. Ajustar la caja al borde de la lesión, sin dejar mucho fondo.
7. Si dudas de la clase, pon la que creas más probable. No dejes la caja sin clase.

## Qué se calcula después (lo hace el script `a13_acuerdo_anotadores.py`)
- **Por foto:** para cada clase, si los dos dicen "hay" o "no hay". Se resume con el **kappa de Cohen**.
  - Mayor a 0.6 = acuerdo sustancial; mayor a 0.8 = casi perfecto.
- **Por caja:** cuántas cajas coinciden en clase y posición (IoU ≥ 0.5 y, más flexible, IoU ≥ 0.3).
  Se resume con F1 por clase.
- **Cada anotador frente a las etiquetas del dataset (v6),** para ver qué tan reproducibles son.
