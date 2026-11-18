# Matriz de respuesta a la revisión (reporte G6, 13/11/2026)

## Documentos
- **Versión nueva:** `Avance_corregido_revision_MDPI_KanaChullo.pdf` (17/11/2026, 13 páginas).
  - Incluye la v6 (etiquetas corregidas, 4 clases), la prueba de estrés por sobreexposición y los experimentos de mitigación.
- **Versiones anteriores:**
  - `Avance_final_antes_revision.pdf`: la que se revisó.
  - `main_antes_v6.tex`: corrección del 16/11, sin la v6.
- **Análisis y scripts:**
  - `D:\ProyectoInvestigacion\analisis_revision\` (a1–a10).
  - `app/tareas/` (docs 11–15, `entrenar_yolo_v6.py`, `entrenar_control.py`).
  - Repositorio `trabajoinvestigacionTomate`.

**Posición:** **A** = acepto; **AP** = acepto en parte; **D** = discrepo con sustento; **P** = pendiente.

## Numeración de la versión nueva
- **Secciones:**
  - 1 Introducción; 2 Trabajos relacionados.
  - 3 Métodos (3.1 Captura, 3.2 Anotación, 3.3 División, 3.4 Modelos, 3.5 Evaluación, 3.6 Sistema, 3.7 Disponibilidad).
  - 4 Resultados (4.1 P1 fuga, 4.2 P2 ampliación, 4.3 P3 exposición, 4.4 P4 mitigación, 4.5 P5 diagnóstico y teléfono, 4.6 Auditoría).
  - 5 Discusión (5.1 Limitaciones); 6 Conclusiones.
- **Tablas:** 1 relacionados; 2 dataset v6; 3 entorno; 4 recursos; 5 v4–v5; 6 mitigación; 7 por imagen.
- **Figuras:** 1 arquitectura; 2 v4–v5 por modelo; 3 por clase; 4 exposición; 5 prueba de estrés; 6 app; 7 auditoría.

## Respuesta a cada observación

| N.º | Observación | Posición | Qué se hizo y por qué | Ubicación | Evidencia |
|---|---|---|---|---|---|
| 1 | Radiación solar no estudiada | **A** (ruta a; campo P) | La radiación pasó a ser el aporte central: título, objetivo, P3 y P4. **(i) Medida:** al mediodía la saturación se cuadruplica (3.6 % frente a 0.8–1.0 %); el Bv del EXIF lo confirma. **(ii) Efecto causal:** prueba de estrés con la misma foto sobreexpuesta; +1 EV quita 32 % del mAP y dos tercios del AP de mosca blanca, y la subexposición no afecta. **(iii) Mitigación:** 27 corridas. El fotométrico eleva la retención a +2 EV (42 % frente a 29 %, 9/9 corridas, p = 0.004) a costa de −2.4 puntos con luz normal; CLAHE no ayuda. Ninguna estrategia rescata la mosca blanca por encima de +1.5 EV, así que la mitigación debe incluir la captura. **(iv) Aviso de la app validado:** no salta en ninguna foto real, salta en el 61 % a +1 EV y el 98 % desde +1.5 EV, detecta el 79 % de las fallas inducidas por la luz y, con aviso, el modelo falla el doble (48 % frente a 24 %); se mantienen los umbrales y el mensaje ahora dice "Mucha luz: dé sombra". **Pendiente:** prueba pareada de campo (mismas hojas en 3 franjas, luxómetro, sombra al mediodía, NASA POWER), con el protocolo listo. | Título, resumen, P3–P4, Sec. 3.5, 4.3, 4.4, Tabla 6, Fig. 5, Discusión | `a3_exposicion.py`, `a7_exif_luz.py`, `a8_estres_exposicion.py`, `a9_analisis_v6.py`, `a11_aviso_luz.py`, `estres_v6.csv`, `aviso_luz_resumen.txt`, `13_plan_radiacion.md`, `15_resultados_v6.md` |
| 2 | La caída v1→v4 no aísla la fuga | **A** (experimento P) | Se reescribió. El conteo de fotos hermanas (82/103) prueba que había fuga, pero la caída de mAP no se le atribuye, porque cambiaron a la vez la división, las etiquetas y el formato. El experimento controlado está programado y probado en `entrenar_control.py`: misma v6, división aleatoria por imagen frente a por grupo (95 % y 99 % de fuga medida), YOLOv8n, 3 semillas. No se ejecutó por falta de tiempo en la RTX (~2.5 h); se declara en Limitaciones. | Sec. 4.1, 5.1 (v), Conclusiones | `entrenar_control.py`, `14_entrenamiento_v6.md` |
| 3 | Comparación v4–v5 confundida | **A** | (i) Confirmado: los valores v4 del "test fijo" eran los de 102 imágenes con etiquetas v4. Se reevaluó todo con las mismas 97 imágenes y las etiquetas v5. (ii) Se declara la versión de etiquetas. (iii) Se reconocen los cambios de hiperparámetros. (iv) La conclusión cambió: la v5 no mejora en mayo y sí en julio (generalización temporal). La v6 confirma el patrón (prueba fija 20.2 %, nuevas 37.6 %). El control "solo mayo frente a mixto del mismo tamaño", que separa sesiones de cantidad, está en `entrenar_control.py` sin ejecutar. | Tabla 5, Fig. 2, Sec. 4.2, 4.4, Discusión | `metricas/map_ultralytics.csv`, `v4_vs_v5_pareado.csv`, `comparacion_v6.csv` |
| 4 | Texto en español para Sensors | **A** (P) | Se traducirá al inglés cuando el fondo esté cerrado, como sugiere el revisor. | — | — |
| 5 | Grupo por tiempo, no por planta | **A** | Se usa "grupo de captura" en lugar de "planta". Auditoría independiente del tiempo con pHash sobre los 856 785 pares entre subconjuntos: 0 pares casi idénticos (mínimo 12; referencia intra-ráfaga 30). Se declara que no descarta la misma planta en dos ráfagas. "Imágenes nuevas" (julio) funciona como prueba temporal parcial. La prueba de campo nueva se dividirá por planta marcada. | Sec. 3.3, 4.1, 5.1 | `duplicados/resumen.json`, `a4_duplicados.py`, `13_plan_radiacion.md` |
| 6 | Solo DE entre semillas | **A** | IC 95 % por bootstrap agrupado (1000 réplicas): unos ±6–8 puntos de mAP@0.5 en la prueba fija. Diferencias pareadas v4–v5 con IC; McNemar e IC de Wilson; n de cajas por clase. En la v6, las diferencias entre experimentos se prueban por pares de arquitectura y semilla (Wilcoxon) y se informa cuántos pares mejoran. Se declara que con 3 semillas la DE es del orden de varias diferencias. | Sec. 3.5, 4.2, 4.4, 4.5, Tablas 5–7, 5.1 (vi) | `map_con_ic.csv`, `mcnemar_v4_v5.csv`, `a9_analisis_v6.py`, `robustez_v6.csv` |
| 7 | Acierto de clase principal mal definido | **A** | Métricas por imagen: sensibilidad por plaga y tasa de falso sano (22 % v4; 28 % v5 instalado; 44 % v6 base, 33–60 % entre corridas). Líneas base "siempre mildiu" (60.8 % en la prueba fija, igual que los modelos) y "siempre sana". El falso sano se mide también en la prueba de estrés: sube de 44 % a 65 % con +2 EV. La discusión propone que la app muestre la confianza y pida revisar cuando no detecta nada. Falta calibración y regla de abstención. | Sec. 3.5, 4.3, 4.5, Tablas 6–7, Discusión | `metricas_por_imagen.csv`, `lineas_base.csv`, `estres_v6.csv` |
| 8 | Teléfono con modelos v1/v4; discordancia | **A** (campo P) | v5 convertido (6.1 MB; coincide con ONNX en 96/96, Δ ≤ 0.014) e instalado. La discordancia se explica por la interpolación del redimensionado (OpenCV frente a Dart). Se reporta el delegado (CPU, 4 hilos). Las capturas se declaran tomadas sobre un monitor. **v6 desplegado:** backend y app adaptados a 4 clases con el YOLOv8n v6 (servidor `best.pt`; TFLite float16 de 6.1 MB, coincide con ONNX en 96/96, Δ = 0.004; pruebas del backend 14/14 y app sin errores de análisis). **Medido en el Galaxy A56 con la v6** (192 imágenes, el mismo código de la app): 574 ± 186 ms por foto (mediana 596 ms; primera corrida 756 ± 497 ms; detecciones idénticas en ambas). El tiempo lo domina decodificar y reducir la foto: 271 ms con fotos < 1 MP frente a 678 ms con fotos de 50 MP. Misma clase principal que el servidor en el 82.8 % (IC 95 %: 76.8–87.5), igual que la v4. Reproduce el 77 % de las cajas del servidor. **Pendiente:** igualar la interpolación y probar en campo. | Sec. 3.6, 4.5, Tabla 3, Fig. 6, 5.1 (vii) | `comparar_onnx_tflite.py`, `v6/celular/comparacion_servidor_movil_v6.json` |
| 9 | Sin trabajos relacionados | **A** | Nueva sección con tabla. Solo se citan trabajos leídos en la fuente: Shehu 2025, Debnath 2023, Zhao 2025, Gerz y Jelali 2025. Se precisa el vacío: ninguno mide cómo la sobreexposición degrada al detector. Kamei 2023, Gookyi 2024 y HFA-Net no se pudieron leer (acceso bloqueado). | Sec. 2, Tabla 1 | — |
| 10 | Etiquetas sin validez experta | **AP** (resultados P) | **(i) Experiencia declarada:** el anotador principal tiene 7 años en el cultivo de tomate en El Pedregal (Majes, Arequipa), en un fundo con asistencia técnica de ingenieros de campo. Se declara como experiencia de campo, no como sustituto de una validación independiente. **(ii) Reglas y alcance:** reglas escritas completas (incluido 16 px); especie de oídio no confirmada; polilla como insecto + mina; identificación por síntomas visibles. **(iii) Auditoría y corrección:** 9 modelos como revisores encontraron 4 problemas. En la v6 se revisaron a mano 319 imágenes (mildiu 2698 → 2933 cajas) y se eliminó la hoja sana. Solo 2 cajas tenían la clase equivocada: el problema es de criterio, no de identificación. **(iv) Acuerdo entre anotadores (protocolo ciego, en curso):** muestra estratificada de 100 imágenes (3 sesiones, un grupo por imagen, 258 cajas), etiquetada por separado por los dos anotadores. Se reporta el kappa de Cohen por clase y el F1 por caja (IoU ≥ 0.5 y ≥ 0.3). El script está validado. **(v) Validación externa (en curso):** hoja de 50 fotos para un ingeniero agrónomo; kappa con las etiquetas v6. **Pendiente:** completar (iv) y (v) e incorporar las cifras. | Sec. 3.2, 3.5 (acuerdo), 4.6, Tabla 2, Fig. 7, 5.1 (i) | `errores/candidatos.csv`, `12_correccion_etiquetas.md`, `a12_muestra_acuerdo.py`, `a13_acuerdo_anotadores.py`, `a14_hoja_ingeniero.py`, `acuerdo_anotadores/` |
| 11 | Datos y código no publicados | **A** | Dataset público en Roboflow Universe (CC BY 4.0): versión 1 = v4, versión 2 = v5, versión 3 = v6. Código, listas de prueba, métricas y modelos en GitHub; tabla de entorno con versiones. Opcional: depósito en Zenodo con DOI. | Sec. 3.7, Tablas 3–4, Disponibilidad | Links verificados |
| 12 | Formato de informe de avance | **A** | Objetivo y preguntas P1–P5; resultados organizados por pregunta; sin "PPI", fases ni versión del servidor; párrafo de limitaciones. | Introducción, Sec. 4, 5.1 | — |
| 13 | Figuras | **A** | Se eliminaron el diagrama de fases, el mosaico de Ultralytics y la figura ilegible de curvas. Figuras nuevas: comparación con las mismas etiquetas, por clase, exposición, prueba de estrés (con banda de ±1 DE) y auditoría. | Figs. 1–7 | — |
| 14 | Cifras sin trazabilidad | **A** | "Hasta 6 puntos" se reemplazó por IC y rangos. "96 de 96" se explica como la comparación ONNX–TFLite con la misma entrada. Se retiró "más GPU no ayudan". Cada cifra de la v6 sale de `comparacion_v6.csv` / `estres_v6.csv`. | Sec. 4.2, 4.4, 4.5 | CSV citados |
| 15 | Referencias incompletas | **AP** | Se completó [JutePest] (vol. 12, 72938–72956). Se retiraron las referencias débiles o sin verificar (SENASA, Shafi, Ksibi, Das, Mengesha, Yadav, Feng). | Referencias | — |
| 16 | Terminología | **A** | Nombres comunes y científicos en el texto; sin identificadores de código. | Todo el texto | — |
| 17 | Título, resumen y palabras clave | **A** | Título nuevo centrado en la radiación. Resumen de menos de 200 palabras con objetivo, método y resultados cuantitativos. Palabras clave que no repiten el título (radiación solar, sobreexposición, aumento fotométrico, etc.). | Portada | — |
| Ap. 1 | Hoja sana como clase | **A** (cambio) | Se aceptó la sugerencia. En la v5 se reporta el mAP con y sin ella (baja 6–9 puntos). En la v6 se **eliminó**: estaba etiquetada en parte (212 casos sin caja) y no es una plaga. Las imágenes sin plaga quedan como fondo, y el falso sano mide el error de declarar sana una planta afectada. | Sec. 3.2, Tablas 2 y 5–6 | `map_4_vs_5.csv`, `comparacion_v6.csv` |
| Ap. 2 | Conservar la etapa v1 | **AP** | Reducida a un párrafo en P1, como caso medido de fuga. | Sec. 4.1 | — |
| Ap. 3 | Elegir YOLOv8n aunque YOLOv10n gane en prueba | **Mantengo** | Seleccionar con prueba sesga la estimación (Kapoor y Narayanan). En la v6, YOLOv8n gana en validación y también en prueba. | Discusión | `resumen_v6.txt` |
| Ap. 4 | Solo modelos nano | **Mantengo** | Restricción de despliegue en el teléfono; la conclusión se acota a modelos ligeros. El modelo `s` queda como experimento opcional. | Discusión | — |

## Resumen para el ingeniero
- **Atendidas por completo (15):** observaciones 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16 y 17; apreciaciones 1, 3 y 4. En la obs. 8 queda pendiente solo la prueba en campo.
- **Atendidas en parte, obs. 1:** falta la validación pareada en campo.
- **Atendidas en parte, obs. 2:** el experimento de control está listo, sin ejecutar.
- **Atendidas en parte, obs. 10:** experiencia declarada y auditoría hechas; el acuerdo entre anotadores y la validación externa están en curso con el protocolo listo.
- **Atendidas en parte, obs. 15 y apreciación 2.**
- **Pendiente por decisión del revisor:** obs. 4, traducción al final.

## Pendientes antes del envío a la revista
1. Prueba pareada de campo de la radiación (obs. 1), según el protocolo de `13_plan_radiacion.md`.
2. Ejecutar `entrenar_control.py` en la RTX, ~2.5 h (obs. 2 y 3).
3. Etiquetado ciego de las 100 imágenes por Fray y Hiram, y hoja de 50 fotos para un ingeniero agrónomo (obs. 10).
4. Igualar la interpolación servidor–teléfono y probar la app en campo (obs. 8).
5. Leer y citar Kamei 2023, Gookyi 2024 y HFA-Net (obs. 9).
6. Traducir al inglés (obs. 4).
