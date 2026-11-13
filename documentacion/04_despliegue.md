# Despliegue del backend (Fase 4)

Objetivo: que la app funcione desde cualquier lugar (campo, universidad, sustentación) con una
URL fija `https://...`, en vez de depender de la IP de la PC en la misma red Wi-Fi.

## Qué ya está listo en el código

- `backend/Dockerfile`: imagen con torch **solo CPU** (sin CUDA) y puerto `$PORT`.
- `backend/.dockerignore`: no se suben `venv311/`, `.env` (secretos), bases `.db` ni `uploads/`.
- `UPLOAD_DIR` configurable: en el servidor las imágenes van a un disco persistente (`/data/uploads`).
- `docker-compose.yml`: prueba local igual al servidor (API + PostgreSQL).
- La app acepta `https://...` en "Servidor: …" (pantalla de login), sin recompilar.

## Opciones

| Opción | Costo aprox. | Pros | Contras |
|---|---|---|---|
| **Railway** (recomendado) | ~5 USD/mes (plan Hobby) | Despliega el `Dockerfile` desde GitHub, PostgreSQL y volumen en 2 clics, HTTPS automático | Requiere tarjeta para el plan Hobby |
| Render | Gratis / 7 USD/mes | Plan gratis disponible | El gratis tiene 512 MB de RAM (justo para torch) y se "duerme": la 1.ª petición tarda ~1 min |
| VPS (DigitalOcean, Contabo…) | 4–6 USD/mes | Control total, 2 GB RAM | Hay que instalar Docker, HTTPS (Caddy/Nginx) y mantenerlo |

> Memoria: el backend con torch + YOLOv8n usa ~500–700 MB de RAM. Pedir **≥ 1 GB**.

## Pasos con Railway

1. [ ] El backend ya tiene repo: `github.com/fraykanachullo/backentomate`. Hacer commit y push de
       los cambios recientes. Confirmar que el repo sea **privado** y que `.env` no se sube
       (ya está en `.gitignore`). Ojo: `.env.example` sí se sube y trae contraseñas de ejemplo.
2. [ ] En railway.app: *New Project → Deploy from GitHub repo* → elegir el repo
       (Root Directory: `backend` si el repo contiene más carpetas). Detecta el `Dockerfile`.
3. [ ] *New → Database → PostgreSQL*.
4. [ ] En el servicio de la API → *Variables*:
   ```
   DATABASE_URL=postgresql+asyncpg://<usuario>:<clave>@<host>:<puerto>/<db>   (de la BD de Railway)
   DATABASE_URL_SYNC=postgresql://<usuario>:<clave>@<host>:<puerto>/<db>
   JWT_SECRET_KEY=<cadena larga aleatoria NUEVA, no la del .env local>
   ADMIN_EMAIL=admin@tomate.com
   ADMIN_PASSWORD=<contraseña NUEVA y fuerte>
   UPLOAD_DIR=/data/uploads
   CONFIDENCE_THRESHOLD=0.30
   IOU_THRESHOLD=0.45
   DEBUG=false
   CORS_ORIGINS=*
   ```
   (Railway entrega la URL de PostgreSQL como `postgresql://...`; para `DATABASE_URL` cambiar el
   inicio a `postgresql+asyncpg://`.)
5. [ ] *Volumes → Add volume* montado en `/data` (imágenes persistentes).
6. [ ] *Settings → Networking → Generate Domain* → URL tipo `https://tomate-api.up.railway.app`.
7. [ ] Probar en el navegador: `https://.../health` y `https://.../docs`.
8. [ ] En la app: login → tocar "Servidor: …" → pegar la URL `https://...` → Guardar.
9. [ ] (Opcional) Cambiar `defaultBaseUrl` en `frontend/lib/core/constants/api_constants.dart`
       a la URL nueva y recompilar, para que las instalaciones nuevas ya apunten ahí.

## Probar localmente antes (opcional)

Con Docker Desktop abierto:

```bash
cd D:\ProyectoInvestigacion\app\backend
docker compose up --build
```

API en `http://localhost:8000` con PostgreSQL, como en el servidor.

## Seguridad antes de publicar

- [ ] `JWT_SECRET_KEY` y `ADMIN_PASSWORD` nuevos en el servidor (no reutilizar los locales).
- [ ] `DEBUG=false`.
- [ ] Las imágenes en `/uploads/...` se sirven sin login (la URL lleva un UUID difícil de adivinar).
      Para producción real: servirlas solo con token o usar URLs firmadas (Supabase Storage).
- [ ] Respaldo de la base PostgreSQL (Railway tiene backups en planes pagos).

## Mejora futura: backend más liviano

Servir el modelo con **ONNX Runtime** (`best.onnx`) en vez de torch reduciría la imagen de
~1.5 GB a ~300 MB y la RAM a ~200 MB, lo que permitiría el plan gratis de Render.
