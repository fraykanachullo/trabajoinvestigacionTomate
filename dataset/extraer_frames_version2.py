"""
EXTRACCIÓN DE FRAMES DESDE VIDEOS MP4 - Version 2
Dataset: Tomate - Majes, Arequipa
Videos de V3 (tarde) -> frames JPG 640x640 en dataset_procesado/version2/V3/frames_videos
"""

import cv2
import os
from pathlib import Path

# ========== CONFIGURACIÓN ==========
PROYECTO = Path(r"D:\ProyectoInvestigacion")
VIDEOS_DIR = PROYECTO / "datasetcrudo" / "V3" / "videos"   # Solo videos de V3
OUTPUT_DIR = PROYECTO / "dataset_procesado" / "version2" / "V3" / "frames_videos"  # Salida

# Cada cuántos frames guardamos (1 = guardar todos, 30 = guardar 1 por segundo si 30fps)
FRAME_SKIP = 30

# Tamaño de salida (opcional, mismo tamaño que tus JPGs)
RESIZE = (640, 640)
# ===================================

# Crear carpeta de salida
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("📹 EXTRACCIÓN DE FRAMES - VIDEOS V3")
print("="*70)
print(f"📁 Origen:  {VIDEOS_DIR}")
print(f"📁 Destino: {OUTPUT_DIR}")
print(f"⚙️  Frame skip: cada {FRAME_SKIP} frames")
print("="*70)

total_frames_extraidos = 0
videos_procesados = 0

# Buscar todos los MP4 en las subcarpetas
for mp4_path in VIDEOS_DIR.rglob("*.mp4"):
    print(f"\n🎬 Procesando: {mp4_path.name}")
    
    # Abrir video
    cap = cv2.VideoCapture(str(mp4_path))
    
    if not cap.isOpened():
        print(f"   ⚠️  No se pudo abrir: {mp4_path.name}")
        continue
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion = total_frames / fps if fps > 0 else 0
    
    print(f"   📊 Duración: {duracion:.1f}s | FPS: {fps:.1f} | Total frames: {total_frames}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Guardar solo cada N frames
        if frame_count % FRAME_SKIP == 0:
            # Redimensionar
            if RESIZE:
                frame_resized = cv2.resize(frame, RESIZE)
            else:
                frame_resized = frame
            
            # Nombre del archivo: video_nombre_frame_numero.jpg
            nombre_salida = f"{mp4_path.stem}_frame_{frame_count:06d}.jpg"
            ruta_salida = OUTPUT_DIR / nombre_salida
            
            # Guardar como JPG
            cv2.imwrite(str(ruta_salida), frame_resized)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"   ✅ Frames guardados: {saved_count} (de {frame_count} totales)")
    
    total_frames_extraidos += saved_count
    videos_procesados += 1

print("\n" + "="*70)
print("📊 RESUMEN FINAL")
print("="*70)
print(f"🎬 Videos procesados:     {videos_procesados}")
print(f"🖼️  Frames extraídos:      {total_frames_extraidos}")
print(f"📁 Carpeta de salida:     {OUTPUT_DIR}")
print("="*70)

# Mostrar algunos archivos de ejemplo
if total_frames_extraidos > 0:
    print("\n📷 Ejemplos de archivos generados:")
    for i, img in enumerate(OUTPUT_DIR.glob("*.jpg")):
        if i < 5:
            print(f"   - {img.name}")
        else:
            break