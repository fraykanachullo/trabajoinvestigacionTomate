# Observaciones del PPI

Documento revisado: *PPI — C8 — 2025-1 — Grupo 6* (34 págs.)
"Sistema de visión artificial basado en redes neuronales profundas con mecanismos de atención
para la detección de plagas y enfermedades del tomate en entornos de alta radiación solar en el sur del Perú".

**Veredicto:** aprobable y por encima del promedio en diseño de datos y arquitectura.
Lo que más hay que reforzar es la coherencia entre lo que promete el título y lo que se experimenta.

---

## Fortalezas (mantener)

- Problema y GAP claros: "sobreoptimismo algorítmico" (Jelali 2024) vs. radiación extrema de Majes.
- Área de estudio concreta (El Pedregal, 17 ha, 1410 msnm) y datos propios de campo.
- Protocolo anti fuga de datos (partición por planta/sesión, aumento solo en train). Muy bueno.
- Metadatos de captura (hora 08/12/16 h, plano general/detalle/macro, haz/envés).
- Arquitectura offline-first (Fig. 16): TFLite local + FastAPI remoto. Coincide con la app real.
- Aspectos éticos, cronograma y presupuesto completos.

---

## 🔴 Prioridad alta

### 1. Título/objetivo vs. metodología: "mecanismos de atención"
El título y el objetivo general prometen atención y preprocesamiento contra la saturación, pero la
Fase 3 solo compara YOLOv8n / YOLOv10n / YOLO11n tal como vienen. Las Figs. 9–12 (atención grupal,
G-MLP de Feng et al.) no indican cómo se integran en YOLO. Pregunta segura del jurado:
*"¿Dónde está su mecanismo de atención?"*

- [ ] Elegir una opción y escribirla en la Fase 3:
  - **Opción A (simple):** YOLO11 ya incluye el bloque de atención **C2PSA** (position-sensitive attention).
    Presentarlo como el mecanismo de atención y comparar contra YOLOv8 (sin C2PSA).
  - **Opción B (más aporte):** agregar un módulo CBAM / SE / ECA en el *neck* de YOLO11n
    y hacer **ablación**: con atención vs. sin atención.
- [ ] Quitar o reducir las Figs. 9–12 si no se implementa exactamente eso.

### 2. Experimento de saturación lumínica (responde la pregunta general)
- [ ] Separar un subconjunto de test "alta exposición" (capturas ~12:00 h) y otro normal.
- [ ] Medir la caída de mAP entre ambos.
- [ ] Comparar con / sin preprocesamiento (CLAHE, corrección gamma) y con / sin aumento fotométrico.
- [ ] Reportar en una tabla: configuración × subconjunto → mAP@0.5, F1.

### 3. Hipótesis y variables (enfoque cuantitativo experimental)
- [ ] Hipótesis general (ej.: "La integración de atención y preprocesamiento fotométrico en YOLO
      mejora el mAP@0.5 en imágenes de alta radiación respecto al modelo base").
- [ ] Hipótesis específicas (una por objetivo específico).
- [ ] Matriz de operacionalización:
      VI = arquitectura, módulo de atención, preprocesamiento, aumento;
      VD = mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 por clase, latencia (ms), tamaño (MB).

### 4. Ingeniería de software (pesa en Ing. de Sistemas)
La app y la API casi no tienen sustento en el documento.
- [ ] Metodología de desarrollo (Scrum o XP) con sprints alineados al cronograma (meses 7–10).
- [ ] Requisitos funcionales / no funcionales y diagrama de casos de uso.
- [ ] Diagrama de arquitectura de software real (Flutter + FastAPI + SQLite/PostgreSQL + TFLite).
- [ ] Plan de pruebas (unitarias, integración, campo) y evaluación con usuarios
      (ej.: escala SUS, tiempo de diagnóstico, % de aciertos en campo).

---

## 🟡 Prioridad media

### 5. Métricas incorrectas para detección
- [ ] ROC, *accuracy* y MCC (Figs. 17–18) son de **clasificación**. Para YOLO usar:
      mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 **por clase**, matriz de confusión de detección.
- [ ] Cambiar "grid search de funciones de activación" por ajuste de lr, batch, imgsz, épocas y aumentos.

### 6. Contradicciones de alcance
- [ ] UAV: la nota de alcance lo excluye, pero Fase 1 y Fase 2 dicen que se usarán imágenes de UAV. Unificar.
- [ ] Segmentación ROI (Fase 2 y cronograma): explicar cómo se hará o quitarla.
- [ ] Super-resolución, *few-shot learning* y multiespectral: se justifican pero no se usan.
      Pasarlos a "trabajos futuros".

### 7. Dataset "balanceado" sin números
Dataset actual (train): sana 1032, mildiu 759, oídio 459, mosca blanca 145, **polilla 84** cajas.
- [ ] Definir meta por clase (≥ 300–500 cajas en train).
- [ ] Criterios de etiquetado escritos (qué se encierra en cada clase, tamaño mínimo, oclusión).
- [ ] Definir la clase "sana": ¿se etiqueta con caja o significa *ausencia de detecciones*?
- [ ] Reportar la distribución final por clase y por subconjunto (train/valid/test).

---

## 🟢 Figuras y forma

### 8. Figuras a reemplazar o corregir
- [ ] Fig. 8: dron sobre un **olivar** (no es tomate y UAV está fuera de alcance) → quitar.
- [ ] Figs. 13, 14, 15, 17, 18: de otros cultivos/tareas (sépalos hiperespectrales, maní, clasificación)
      → reemplazar por resultados propios.
- [ ] Fig. 4: dice "daños causados por *Tuta absoluta*" pero muestra la **polilla adulta** → usar foto de minas en hoja.
- [ ] Figs. 1–3: agregar la fuente.

### 9. Formato
- [ ] "Contents / Figure / Table / References" en inglés → `\usepackage[spanish]{babel}` en LaTeX.
- [ ] Unificar estilo de citas (numérico [6] vs. autor-año en negrita "**Jelali (2024)**").
- [ ] Ref. [8]: falta la inicial del primer autor (Grbović).
- [ ] Ref. [16]: "vol. Volume 15 - 2024, 2025" → corregir volumen/año.
- [ ] p. 27: "realizara" → "realizará".
- [ ] Presupuesto: dice MongoDB Atlas, pero el sistema usa SQLite/PostgreSQL → alinear.
- [ ] Cronograma: verificar que las fechas reflejen el avance real.

---

## Datos que ya pueden usar en el documento

- **Baseline actual (YOLOv8n):** 68.0 % de acierto en clase principal sobre 103 imágenes de test
  (conf 0.30, iou 0.45). Antes de corregir el pipeline de colores era 42.7 %.
- Script de evaluación: `backend/scripts/evaluar_modelo.py` (ver `03_entrenamiento_modelo.md`).
