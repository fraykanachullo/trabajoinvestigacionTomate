# Conclusiones de los 3 entrenamientos de la versión 1

Revisado el 05/11/2026 a partir de:
- `notebokparaelentrenamiento/IATOMATOv1_de_los_3_entrenamientos.ipynb`
- `Resultados de los entrenamientos/train 1, 2 y 3` (`results.csv`, `args.yaml`, pesos)
- Reevaluación de los 3 `best.pt` en la PC con `plagas_tomate_majes/datasets`

Todos: YOLOv8n, `imgsz=640`, `batch=16`, semilla 0, GPU Tesla T4 de Colab, 720 / 205 / 103 imágenes.

## 1. Resultados reales

| | Exp. 1 | Exp. 2 | Exp. 3 |
|---|---|---|---|
| Épocas | 50 | 139 (de 150, parada temprana) | 100 |
| Tiempo | 707 s (11.8 min) | 2027 s (33.8 min) | 1462 s (24.4 min) |
| Cambios | por defecto | más épocas | `degrees=15`, HSV algo mayor (0.02/0.8/0.5) |
| Mejor mAP50 en valid (época) | 30.19 % (45) | 30.62 % (104) | 30.15 % (66) |
| Mejor mAP50-95 en valid (época) | 13.60 % (42) | 13.99 % (109) | 13.33 % (86) |
| Última época: mAP50 / P / R | 29.18 / 43.58 / 33.34 | 29.02 / 44.03 / 31.37 | 28.46 / 37.44 / 34.53 |

Por clase con `best.pt`, en valid. Exp. 1 y 3 salen del notebook; Exp. 2 no tiene salida guardada y se
reevaluó en la PC (con esa misma reevaluación, Exp. 1 da 28.95 % frente a 30.2 % del notebook, por lo que
hay ~1 punto de variación entre ambas fuentes).

| Clase | Instancias | AP50 Exp. 1 | AP50 Exp. 2 | AP50 Exp. 3 |
|---|---|---|---|---|
| polilla | 25 | 63.5 % | 66.1 % | 63.4 % |
| sana | 232 | 46.4 % | 46.0 % | 48.5 % |
| mildiu | 319 | 22.6 % | 16.5 % | 16.5 % |
| mosca_blanca | 81 | 10.0 % | 13.7 % | 9.4 % |
| oidio | 137 | 8.4 % | 6.5 % | 6.8 % |

## 2. Conclusiones

1. **Los tres experimentos empatan.** La diferencia es de menos de 1 punto, con una sola semilla. Ninguna
   configuración se puede declarar mejor.
2. **Entrenar más no ayuda.** En el Exp. 2, `val/box_loss` toca su mínimo en la época 14 (2.08) y luego
   sube (2.30), mientras `train/cls_loss` sigue bajando hasta 0.79 y `val/cls_loss` sube de 2.31 a 2.50:
   es sobreajuste. El mAP50 se estanca alrededor de 27–30 % desde la época ~50–60. El límite está en los
   datos, no en las épocas.
3. **El modelo falla más por no encontrar los objetos que por confundir las clases.** En la matriz de
   confusión del Exp. 2, quedan sin detectar (fondo) el 88 % de mosca blanca, el 83 % de oídio y el 63 %
   de mildiu; la confusión entre clases no pasa de 4 %. Hipótesis: cajas mal puestas o faltantes y
   objetos pequeños. La revisión de etiquetas (v4) permite comprobarla.
4. **El buen resultado de polilla no es confiable.** Es la clase con menos datos y la mejor nota, medida
   con solo 25 cajas (una por imagen) y con fuga de datos. Hay que ver si se mantiene sin fuga.
5. **No se puede atribuir la caída del Exp. 3 al color.** El cambio de HSV es pequeño frente al valor por
   defecto (0.015/0.7/0.4) y además se agregó rotación de 15°. Mildiu sí bajó (22.6 % → 16.5 %), pero
   ambos cambios van juntos y hay una sola semilla.
6. **El modelo que está en la app es el Exp. 3.** El `best.pt` del backend es idéntico al de `train 3`
   (mismo MD5). Sus 67–68 % de acierto de clase principal son de ese modelo.
7. **El test viejo no sirve para comparar:** da mAP50 de 34.8 / 32.2 / 34.4 %, más que valid, y tiene
   fuga de datos (82 de 103 imágenes con una foto de la misma planta en train o valid).

## 3. Errores que hay que corregir en el short paper y en el avance v1

| Dice | Lo correcto |
|---|---|
| Entrenamiento en CPU (i5-8300H) | GPU **Tesla T4** de Colab (`device=0`, `args.yaml`) |
| 707 / 2027 / 1462 **min** | Son **segundos**: 11.8 / 33.8 / 24.4 min |
| 9.0 ms/imagen en CPU, viable en móvil | Es en la T4. En el i5-8300H son ~120–135 ms. En el celular: 597 ms (medido) |
| Métricas como resultado final del modelo | Son de la **última época**, no del `best.pt` |
| Curva F1: F1=0.32 con confianza 0.450 | **F1=0.31 con confianza 0.476** (la curva del artículo es la del Exp. 2) |
| Tabla por clase del Exp. 2 (polilla 29.5 %, mildiu 29.9 %, mosca recall 1.9 %) | No se reproduce: polilla ~66 %, mildiu ~16 %, mosca recall ~7 %. Rehacerla con `val` |
| Listado del Exp. 3 sin `degrees` | Incluir `degrees=15` y `patience=20` |
| "Aumentación agresiva de HSV" degrada mildiu y oídio | Decir que combina HSV y rotación, y que no se aisló la causa |

## 4. Seguridad

El notebook tiene la **API key de Roboflow escrita en texto plano** (celda 3). Hay que regenerarla en
Roboflow (Settings → API Keys) y no compartir ese archivo hasta borrarla.
