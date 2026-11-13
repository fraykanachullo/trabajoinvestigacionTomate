# App y arquitectura — mejoras por fases

## Arquitectura objetivo (offline-first, coincide con la Fig. 16 del PPI)

```
Celular: modelo TFLite ──► resultado inmediato (sin internet) ──► SQLite (pendiente)
                                                                      │
                                                      hay internet ───┘
                                                                      ▼
Servidor: /api/v1/diagnostic/detect (JWT) ──► best.pt ──► BD + imagen
```

## Ya hecho

- [x] Entorno Python único `venv311` (se eliminó `venv` vacío).
- [x] Registro e inicio de sesión contra el backend (antes el registro era solo local).
- [x] Contraseña mínima de 6 caracteres en front y back.
- [x] Admin maestro creado al arrancar (`ADMIN_*` en `backend/.env`).
- [x] Proyecto ordenado en `D:\ProyectoInvestigacion\app` (backend, frontend, documentación).
- [x] Detección: imagen en RGB (antes colores invertidos), rotación EXIF, conf 0.30, iou 0.45, NMS agnóstico.
- [x] Historial: guarda las cajas reales (antes dibujaba una caja fija inventada), un registro por escaneo.
- [x] Nombres y colores de clases en un solo archivo (`lib/core/constants/disease_classes.dart`).

## Fase 0 — Limpieza

- [x] Backend carga el modelo una sola vez (`main.py` usa la instancia de `yolo_engine`).
- [x] `requirements.txt` con las versiones realmente instaladas.
- [x] El APK solo incluye `best_float16.tflite` (los demás modelos quedan en disco pero no se empaquetan).
- [x] URL del backend configurable desde el login (tocar "Servidor: …"), guardada en el celular.
- [x] Eliminado código muerto (`settings_page`, `diagnostic_detail_page`, `tflite_service`) y la tabla `users` local con contraseñas en texto plano.

## Fase 1 — Detección offline en el celular ⭐

Modelo del celular: `best.pt` exportado a 640 px float16 (6 MB). En Windows el export directo de
Ultralytics a TFLite no funciona; se hizo `best.pt → ONNX → onnx2tf` (ver `03_entrenamiento_modelo.md`).

- [x] Nuevo `LocalDetector` (`lib/core/services/local_detector.dart`): salida `[1, 4+clases, N]`,
      letterbox, NMS agnóstico, conf 0.30 / iou 0.45, inferencia en un isolate (no congela la UI).
- [x] Captura y galería: con red usa el servidor; sin red, si el servidor no responde o falla, usa el celular.
- [x] Perfil → Motor de inferencia: "Automático" o "Solo en el celular" (se guarda en el celular).
- [x] Escaneo en tiempo real: cuadros YUV420 de la cámara → RGB rotado en un isolate → `LocalDetector`; cajas en vivo alineadas con la vista previa (cover) y con la foto (contain). Va tan rápido como el celular (salta cuadros mientras analiza).
- [x] Validado en test (103 img, conf 0.30): celular 67.0 % vs servidor 68.0 %. El modelo anterior a 320 px daba 57.3 %.
- [x] El resultado y el historial muestran si se analizó en el celular o en el servidor.
- [x] Instalado en el celular; se registró un análisis offline pendiente de subir.

## Fase 2 — Sincronización

- [x] Cola de diagnósticos pendientes (`is_synced = 0`): los análisis hechos en el celular quedan "Por subir".
- [x] `SyncService`: al abrir la app, al volver la conexión y con "Sincronizar Ahora" (Perfil) sube los pendientes a `/api/v1/diagnostic/detect` con JWT.
- [x] Historial muestra "Sincronizado" / "Por subir" en cada registro.
- [x] Descarga del historial del servidor (imagen + cajas) sin duplicados (`server_id`), al abrir la app y con el botón de nube del Historial.
- [x] Al cerrar sesión se borra del celular lo ya sincronizado (no se mezclan usuarios); lo pendiente de subir se conserva.
- [x] La app ya no usa `/detect` legacy: con internet analiza vía `/api/v1/diagnostic/detect` (queda guardado con el usuario).
- [x] Sesión guardada: no se pide contraseña al abrir (funciona sin internet); el token se renueva solo (access 8 h, refresh 30 días).

## Fase 3 — Modelo versionado

- [x] `metadata.json` junto al modelo (backend `app/models/` y app `assets/models/`): versión, arquitectura, clases, imgsz, conf, iou, fecha, métricas.
- [x] La app lee clases y umbrales del `metadata.json`; el backend valida que las clases coincidan al arrancar.
- [x] Perfil muestra la versión real del modelo; cada diagnóstico del servidor guarda `model_version` (ej. `YOLOv8n-v1.0.0`).

## Pruebas en celular real (25/10/2026)

Honor X8 (NTN-LX3, Android 13) conectado por USB (`adb reverse tcp:8000 tcp:8000` + servidor `http://localhost:8000`):
login ✅ · sesión guardada tras cerrar la app ✅ · escáner en vivo ✅ (~870 ms/cuadro) · análisis en
servidor ✅ · análisis sin internet ✅ (mismas 4 cajas, 83.5 % vs 83.8 %) · sincronización "Por subir" →
"Sincronizado" sin duplicados ✅ · historial con cajas reales ✅ · Perfil con versión real del modelo ✅.
Corregido: desborde de texto en Perfil en pantallas angostas.

## Pendientes menores (app)

- [x] Resultado: síntomas, condiciones que favorecen y tratamientos según la clase detectada
      (`lib/core/constants/disease_info.dart`); sin datos ambientales inventados; "Confianza" en vez de
      "Precisión"; "También se detectó: …" cuando hay varias clases. Verificado en el Honor X8.
- [ ] Revisar el contenido agronómico de `disease_info.dart` con un ingeniero agrónomo.

- [ ] Perfil: el interruptor "Sincronizar automáticamente en Wi-Fi" aún no hace nada (hoy sincroniza con cualquier red).
- [ ] Perfil: nombre/correo/ubicación de ejemplo cuando no hay usuario; foto de perfil fija de internet.
- [ ] Home y Analítica usan imágenes de ejemplo de internet (no cargan sin conexión).

## Fase 4 — Despliegue

- [x] Backend listo para desplegar: `Dockerfile` (torch CPU, `$PORT`), `.dockerignore`, `UPLOAD_DIR`, `docker-compose` con PostgreSQL.
- [ ] Desplegar en Railway (u otro) siguiendo [04_despliegue.md](04_despliegue.md) — requiere crear la cuenta.
- [ ] PostgreSQL (Supabase) en lugar de SQLite; imágenes en almacenamiento en la nube.
- [ ] `JWT_SECRET_KEY` y credenciales reales solo en el servidor, fuera del repositorio.
- [ ] Endurecer CORS (`CORS_ORIGINS`) y quitar `DEBUG=true`.
