# Qué está fallando en las anotaciones y cómo corregirlo (16/11/2026)

Análisis con los 9 modelos v5 como "segundos revisores" sobre validación y prueba (561 imágenes), más
una revisión estructural de las 1966 (`analisis_revision/a5_errores_etiquetas.py`).
Archivos para trabajar en Roboflow: carpeta [`revision_etiquetas/`](revision_etiquetas/)
(`imagenes_a_revisar.csv` con 319 imágenes priorizadas, `candidatos.csv` con cada caja sospechosa y las
hojas `hoja_*.jpg`: verde = etiqueta actual, rojo = lo que señalan los modelos).

## Diagnóstico: 5 problemas, de mayor a menor efecto

### 1. Oídio casi no está etiquetado en las sesiones de julio
| Lote | Cajas de oídio |
|---|---|
| Imágenes de la v4 (03/05, etiqueta previa revisada) | **869** |
| Imágenes nuevas (944, julio y resto de mayo) | **20** |

O en julio casi no había oídio, o no se etiquetó. Por eso oídio no creció (+8 %) y es la clase que peor
anda. Además el 40 % de las cajas de oídio de valid+test (118 de 295) no las toca ningún modelo: muchas son
puntos muy pequeños o están en zonas borrosas.
**Qué hacer:** revisar en Roboflow imágenes de julio (`tag:sesion-20260712`, `tag:sesion-20260729`)
buscando polvo blanco. Si hay oídio, etiquetarlo; si de verdad no hay, anotarlo (es un dato para el artículo).

### 2. El criterio de mildiu cambia según la sesión
Área mediana de las cajas de mildiu: **0.20 %** de la imagen en la sesión del 12/07, **0.94 %** en las
imágenes nuevas del 03/05 y **1.49 %** en el 29/07. En las hojas se ve: en unas imágenes la caja encierra
una mancha; en otras, la hoja entera con varias manchas (`hoja_NO_VISTA.jpg`, casos 4, 11 y 12).
(Esto corrige lo que dijo el análisis anterior por tamaño absoluto: la mezcla existe, pero es
**por sesión/anotador**, no dentro de una misma imagen.)
**Qué hacer:** elegir UNA regla y aplicarla a todas:
- **Recomendado: una caja por hoja (folíolo) afectada**, que encierre toda la zona dañada de esa hoja.
  Es más fácil de aplicar igual, coincide con cómo se marcan las hojas `sana` y le basta a la app.
- Alternativa: una caja por lesión. Más trabajo y más difícil de hacer consistente.

### 3. `sana` está etiquetada solo en parte (el error más frecuente)
212 casos en valid+test donde **los 9 modelos ven una hoja sana sin caja** (`hoja_FALTANTE.jpg`). Solo
el 34–41 % de las imágenes tiene alguna caja `sana`, y en unas se marcan todas las hojas sanas y en otras
unas pocas. Para el modelo eso es contradictorio: la misma hoja a veces es "sana" y a veces "fondo".
**Qué hacer (decisión importante):**
- **Opción A (recomendada): dejar de etiquetar `sana`.** Las hojas sanas pasan a ser fondo, y la app ya
  dice "sana" cuando no detecta nada. Se elimina el problema más frecuente y el revisor ya cuestionó que
  `sana` sea una clase de detección (43 % de las cajas, no es un objeto sino ausencia de síntoma).
  Requiere reentrenar con 4 clases y un ajuste chico en la app.
- Opción B: etiquetar **todas** las hojas sanas del primer plano en las 1966 imágenes. Mucho trabajo.

### 4. Mosca blanca demasiado pequeña
El 87 % de sus cajas (302 de 349) mide menos de 16 px a 640 px; el modelo no puede verlas.
**Qué hacer:** una caja por **grupo de moscas o zona con insectos** en la hoja, no por insecto. Y en el
entrenamiento, probar 1024 px.

### 5. Mildiu sin etiquetar y casos sueltos
- 30 lesiones de mildiu que los 9 modelos ven sin caja (están en `candidatos.csv`, tipo FALTANTE).
- 23 cajas de mildiu que ocupan más del 25 % de la imagen (tipo ENORME): revisar si son hojas lejanas
  o zonas enteras.
- Imágenes de paisaje con cajas en plantas lejanas (p. ej. `12__g017__20260503_122722`): la regla 10
  dice no etiquetar lo lejano o borroso.
- No hay confusión entre clases (solo 2 casos) ni cajas duplicadas: el problema no es poner la clase
  equivocada, sino **cuánto** y **cómo** se encierra.

## Orden de trabajo en Roboflow

1. **Decidir las reglas** (con el asesor si hace falta): `sana` sí o no (problema 3), caja por hoja o por
   lesión (problema 2), mosca por grupo (problema 4). Escribirlas en `05_etiquetado.md`.
2. **Empezar por prueba y validación** (98 y 205 imágenes de `imagenes_a_revisar.csv`): son las que
   miden el modelo. Luego entrenamiento.
3. **Oídio en julio** (problema 1).
4. **Aplicar las reglas a todo el dataset**, por clase: filtrar `class:mildiu`, `class:mosca_blanca`.
5. Poner el tag `corregida` a cada imagen tocada. No cambiar subconjuntos ni usar Rebalance.
6. Generar la versión 3 en Roboflow (misma configuración) y reentrenar con `entrenar_yolo.py`.

## Por qué esto debería subir el rendimiento

Hoy el modelo recibe señales contradictorias: la misma hoja sana a veces es objeto y a veces fondo; el
mildiu a veces es una mancha y a veces una hoja entera; el oídio existe en mayo y "no existe" en julio.
Con reglas consistentes, las clases con muchos datos (mildiu) deberían dejar de estar en ~25 % de AP.
Comparar v5 contra v6 (reetiquetada) en la misma prueba, **con las etiquetas nuevas para ambos
modelos**, mide el efecto del reetiquetado: es justo el experimento que pide el revisor (obs. 3 y 10).
