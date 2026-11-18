# Resultados v6: 4 clases, etiquetas corregidas y experimentos contra el brillo (17/11/2026)

- Origen: `Resultados de los entrenamientos/v6-4clases/resultados_v6-4clases.zip`, entrenado en la RTX 5070 Ti.
- Análisis: `analisis_revision/a9_analisis_v6.py`. Figura: `a10_fig_v6.py`.
- Salidas: `analisis_revision/v6/` (`robustez_v6.csv`, `estres_v6_resumen.csv`, `fig_estres_v6.png`).
- Diseño: 3 experimentos × 3 arquitecturas (YOLOv8n, YOLO11n, YOLOv10n) × 3 semillas = 27 corridas.
  - Misma partición por grupos: 1405 / 381 / 180, sin fuga.
  - Dataset: Roboflow versión 3, clase `sana` quitada.
- Los experimentos de control (`entrenar_control.py`, obs. 2 y 3) **no se corrieron** por falta de tiempo en la RTX.

## 1. Rendimiento normal (media ± DE de las 9 corridas, %)

| Experimento | Validación | Prueba (180) | Prueba fija (97, mayo) | Nuevas (83, julio) | Falso sano (prueba) |
|---|---|---|---|---|---|
| Base | 32.9 ± 2.4 | **26.5 ± 3.8** | **20.2 ± 2.8** | 37.6 ± 7.0 | **44.4 ± 8.6** |
| Aumento fotométrico | 29.5 ± 3.9 | 25.1 ± 4.1 | 17.8 ± 2.4 | 37.9 ± 8.7 | 52.9 ± 16.8 |
| CLAHE | 31.2 ± 2.8 | 24.6 ± 3.7 | 18.0 ± 3.5 | 35.7 ± 7.0 | 45.4 ± 12.5 |

### Por arquitectura, base (mAP@0.5 en prueba)

| Modelo | v6 (base) | v5 sin hoja sana |
|---|---|---|
| YOLOv8n | 29.8 ± 3.2 | 25.8 ± 3.4 |
| YOLO11n | 26.4 ± 1.0 | 24.0 ± 4.9 |
| YOLOv10n | 23.4 ± 3.9 | 29.8 ± 1.3 |

- La v5 se calculó quitando `sana` de un modelo de 5 clases, con las etiquetas sin corregir.
- **YOLOv8n vuelve a ser el mejor** en v6, y es el modelo de la app.

### AP por clase en prueba (media de 9 corridas)

| Experimento | Mildiu | Mosca blanca | Oídio | Polilla |
|---|---|---|---|---|
| Base | 20.6 | 31.5 | 6.4 | 47.7 |
| Fotométrico | 20.1 | 33.5 | 5.0 | 42.0 |
| CLAHE | 19.4 | 26.4 | 6.7 | 46.0 |

### Diferencias pareadas frente a base

Pares por modelo y semilla; prueba de Wilcoxon con 9 pares.
- **Fotométrico:**
  - Baja en validación (−3.4, 1/9 pares mejores, p = 0.008) y en la prueba fija (−2.4, p = 0.039).
  - Prueba completa −1.4 (p = 0.65); falso sano +8.5 (p = 0.16, no significativo).
- **CLAHE:** ninguna diferencia significativa; todas las medias son algo peores.

**Lectura:** corregir las etiquetas y quitar `sana` no cambió el nivel global.
- Media v6 base 26.5, frente a 26.5 de media de las tres arquitecturas v5 sin `sana`.
- El techo actual lo ponen los datos: oídio con AP 6, pocas cajas y criterio variable. El protocolo de entrenamiento ya no es el límite.

## 2. Prueba de estrés: la mitigación contra el brillo

Las mismas 180 fotos de prueba se sobreexpusieron artificialmente, de −1 a +2 pasos de luz.
- En CLAHE, el preprocesado también se aplica a las fotos sobreexpuestas.
- Figura: `fig_estres_v6.png`.

| EV | Base mAP | Fotom. mAP | CLAHE mAP | Base mosca bl. | Fotom. mosca bl. | Base falso sano | Fotom. falso sano |
|---|---|---|---|---|---|---|---|
| −1 | 25.6 | 24.8 | 25.2 | 27.4 | 32.3 | 47 % | 54 % |
| 0 | 26.3 | 24.8 | 24.6 | 30.6 | 33.1 | 44 % | 53 % |
| +0.5 | 23.0 | 22.8 | 22.2 | 22.2 | 26.9 | 49 % | 56 % |
| +1 | 17.9 | **19.1** | 17.2 | 9.8 | **14.5** | 51 % | 59 % |
| +1.5 | 11.4 | **13.4** | 12.6 | 2.9 | 3.0 | 59 % | 61 % |
| +2 | 7.5 | **10.3** | 7.8 | 0.5 | 0.6 | 65 % | 65 % |

**Retención** = mAP con sobreexposición / mAP original, por corrida:

| | Base | Fotométrico | CLAHE | Fotom. − base (pares mejores, p) |
|---|---|---|---|---|
| +1 EV | 68 % | 76 % | 69 % | +8.0 (6/9, p = 0.098) |
| +1.5 EV | 44 % | 54 % | 51 % | +10.5 (7/9, p = 0.074) |
| +2 EV | 29 % | **42 %** | 31 % | **+13.3 (9/9, p = 0.004)** |

**Qué muestra (núcleo del aporte sobre la radiación):**
1. **La caída por brillo se reproduce con el modelo nuevo y con etiquetas corregidas.**
   - Con +1 paso, el mAP base cae 32 % y la mosca blanca pierde dos tercios de su AP.
   - Con +2 pasos queda el 29 %. La subexposición (−1) no afecta.
2. **El aumento fotométrico sí da robustez, pero tiene un costo.**
   - La pérdida se reduce de forma consistente: a +2 EV, las 9 corridas retienen más que su par base.
   - Con fotos muy quemadas rinde más en términos absolutos: 10.3 frente a 7.5 a +2 EV.
   - Con fotos normales rinde un poco menos (prueba fija −2.4) y declara más fotos "sanas".
   - Es un **compromiso**, no una mejora gratis.
3. **CLAHE no ayuda:** redistribuir el contraste no recupera píxeles quemados.
4. **Ninguna estrategia de software rescata la mosca blanca por encima de +1.5 EV** (AP ≈ 3 en los tres casos).
   - Cuando el insecto blanco se satura, la información desaparece de la foto.
   - La mitigación eficaz tiene que ser **en la captura**: dar sombra a la hoja, corregir la exposición o avisar con "Mucha luz".
   - Esto justifica el protocolo de campo pareado sol/sombra (`13_plan_radiacion.md`) y el aviso validado en la app.
5. **Polilla** (lesiones oscuras): resiste hasta +1 en los tres casos. El fotométrico la baja con fotos normales (−5.7, no significativo).

**Por arquitectura:** la ganancia de robustez del fotométrico aparece en YOLOv8n y YOLO11n.
- Retención a +1: 80 % frente a 70 % y 67 %.
- En YOLOv10n casi no aparece (69 % frente a 67 %).

## 3. Decisión para la app

- **Recomendado:** YOLOv8n base (`mejor_base_yolov8n_s1.pt`), con el mejor rendimiento con luz normal (prueba 32.4).
  - Para el uso con sol fuerte se combina con el aviso "Mucha luz" y la recomendación de dar sombra.
- **Alternativa robusta:** `mejor_fotometrico_yolov8n_s2.pt`. Conviene si la prueba de campo muestra que la mayoría de las fotos reales caen por encima de +1 EV.
  - Hoy la mediana real al mediodía (3.6 % de píxeles saturados) cae entre 0 y +0.5 EV, donde el base es mejor.
- **Pendiente:** adaptar backend y app a 4 clases (hoy esperan 5) y convertir a TFLite antes de instalar.

## 4. Limitaciones

- Solo 3 semillas: las DE entre semillas (3–17 puntos) son del mismo orden que las diferencias.
  - Por eso se reportan pruebas pareadas y la consistencia en pares, no solo las medias.
- La sobreexposición sintética no reproduce reflejos especulares ni sombras duras. Falta la validación pareada en campo.
- Los controles de fuga y de sesiones (obs. 2 y 3) siguen pendientes. El script está listo (`entrenar_control.py`, ~2–2.5 h).
