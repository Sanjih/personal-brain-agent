from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import subprocess
from ultralytics import YOLO

app = FastAPI(
    title="AI Combat Coach API",
    description="API d'analyse biomécanique et tactique pour sports de combat à la minute.",
    version="1.0.0"
)

# Chargement du modèle Pose au démarrage
model = YOLO("yolov8n-pose.pt")

def calculate_duration(file_path: str) -> float:
    """Calcule la durée de la vidéo via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return float(output.strip())

@app.get("/")
def home():
    return {"status": "online", "message": "Bienvenue sur l'API AI Combat Coach"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Reçoit une vidéo, calcule sa durée, simule le débit de crédits
    et lance l'analyse de posture YOLO.
    """
    temp_filename = f"temp_{file.filename}"
    
    # 1. Sauvegarder temporairement la vidéo reçue
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Calculer la durée et le coût
        duration_sec = calculate_duration(temp_filename)
        duration_min = duration_sec / 60
        cost_eur = round(duration_min * 0.50, 2) # Tarif : 0,50 € / min
        
        # 3. Exécuter la détection de posture YOLO
        results = model.predict(source=temp_filename, save=False, conf=0.5)
        detected_frames = len(results)
        
        # 4. Nettoyage du fichier temporaire
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return {
            "success": True,
            "filename": file.filename,
            "duration_seconds": round(duration_sec, 2),
            "cost_calculated_eur": cost_eur,
            "frames_analyzed": detected_frames,
            "message": "Analyse biomécanique réussie. En attente de connexion VLM Nebius pour le rapport textuel."
        }
        
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse : {str(e)}")