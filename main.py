from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
import math
import cv2
import numpy as np
from ultralytics import YOLO
import database as db

db.init_db()

app = FastAPI(
    title="Motion Analysis API",
    description="API d'analyse biomécanique horodatée pour Sports de Combat et Danse.",
    version="2.2.0"
)

model = YOLO("yolov8n-pose.pt")

def get_video_info(file_path: str):
    """Extrait la durée et le nombre d'images par seconde (FPS) de la vidéo."""
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  # Valeur par défaut si non détectée
    
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        duration = float(output.strip())
    except:
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        
    cap.release()
    return duration, fps

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def format_timestamp(seconds: float) -> str:
    """Convertit des secondes en format MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def analyze_timeline_biomechanics(results, fps: float, vid_stride: int, mode: str):
    """Analyse frame par frame et génère une timeline des erreurs."""
    timeline_events = []
    scores_history = []
    
    # Antirebond pour éviter de répéter la même erreur 10 fois par seconde
    last_guard_error_time = -5.0
    last_stance_error_time = -5.0
    
    for i, r in enumerate(results):
        # Index réel de la frame dans la vidéo d'origine
        real_frame_index = i * vid_stride
        timestamp_sec = real_frame_index / fps
        time_str = format_timestamp(timestamp_sec)
        
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0].cpu().numpy()
            
            if len(kpts) >= 17:
                shoulder_l, elbow_l, wrist_l = kpts[5][:2], kpts[7][:2], kpts[9][:2]
                ankle_l, ankle_r = kpts[15][:2], kpts[16][:2]
                
                # Check Coude / Garde
                if np.all(shoulder_l) and np.all(elbow_l) and np.all(wrist_l):
                    angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
                    
                    if mode == "combat" and angle > 130:
                        if timestamp_sec - last_guard_error_time > 2.0: # Délais de 2 sec min entre 2 alertes
                            timeline_events.append({
                                "timestamp": time_str,
                                "type": "urgent",
                                "message": "Garde ouverte : Coude gauche trop distant du corps."
                            })
                            last_guard_error_time = timestamp_sec
                    elif mode == "dance" and angle < 80:
                        if timestamp_sec - last_guard_error_time > 2.0:
                            timeline_events.append({
                                "timestamp": time_str,
                                "type": "warning",
                                "message": "Manque d'amplitude : Bras trop fléchi sur l'extension."
                            })
                            last_guard_error_time = timestamp_sec

                # Check Appuis / Stabilité
                if np.all(ankle_l) and np.all(ankle_r):
                    dist = np.linalg.norm(ankle_l - ankle_r)
                    if mode == "combat" and dist < 40:
                        if timestamp_sec - last_stance_error_time > 2.0:
                            timeline_events.append({
                                "timestamp": time_str,
                                "type": "warning",
                                "message": "Perte d'équilibre : Pieds trop proches (base étroite)."
                            })
                            last_stance_error_time = timestamp_sec

    # Calcul du score global basé sur le nombre d'erreurs détectées
    penalty = len(timeline_events) * 7
    global_score = max(100 - penalty, 45)
    
    if not timeline_events:
        timeline_events.append({
            "timestamp": "00:00",
            "type": "good",
            "message": "Exécution fluide et posture stable tout au long de la séquence !"
        })

    return global_score, timeline_events

def process_video_analysis(file: UploadFile, x_api_key: str, mode: str):
    user = db.get_user_by_api_key(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Clé API non autorisée ou invalide.")
    
    prefix = "combat" if mode == "combat" else "dance"
    temp_filename = f"temp_{prefix}_{file.filename}"
    
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        duration_sec, fps = get_video_info(temp_filename)
        duration_min = round(duration_sec / 60, 2)
        
        if user["balance_minutes"] < duration_min:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise HTTPException(status_code=402, detail="Solde insuffisant.")
            
        output_dir = f"output_{mode}"
        vid_stride = 3
        results = model.predict(
            source=temp_filename, 
            save=True, 
            project=output_dir, 
            name="result", 
            exist_ok=True, 
            conf=0.5 if mode == "combat" else 0.4, 
            vid_stride=vid_stride
        )
        
        # Analyse temporelle
        score, timeline = analyze_timeline_biomechanics(results, fps, vid_stride, mode)
        
        db.deduct_user_minutes(x_api_key, duration_min)
        updated_user = db.get_user_by_api_key(x_api_key)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return {
            "success": True,
            "mode": mode.upper(),
            "filename": file.filename,
            "duration_minutes": duration_min,
            "remaining_balance_minutes": updated_user["balance_minutes"],
            "frames_analyzed": len(results),
            "performance_score": score,
            "timeline_events": timeline,
            "annotated_video_url": f"/download-result/{mode}/{temp_filename}",
            "message": f"Analyse {mode} effectuée avec succès !"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse : {str(e)}")

@app.get("/")
def home():
    return {"status": "online", "message": "API Motion Analysis 2.2"}

@app.post("/users/register")
def register_user(email: str, api_key: str):
    success = db.create_user(email, api_key, initial_minutes=10.0)
    if not success:
        raise HTTPException(status_code=400, detail="Utilisateur déjà existant.")
    return {"message": f"Utilisateur {email} créé."}

@app.get("/users/balance")
def get_balance(x_api_key: str = Header(...)):
    user = db.get_user_by_api_key(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Clé API invalide.")
    return {"email": user["email"], "balance_minutes": user["balance_minutes"]}

@app.post("/analyze/combat")
async def analyze_combat(file: UploadFile = File(...), x_api_key: str = Header(...)):
    return process_video_analysis(file, x_api_key, mode="combat")

@app.post("/analyze/dance")
async def analyze_dance(file: UploadFile = File(...), x_api_key: str = Header(...)):
    return process_video_analysis(file, x_api_key, mode="dance")

@app.get("/download-result/{mode}/{filename}")
def download_result(mode: str, filename: str):
    file_path = os.path.join(f"output_{mode}", "result", filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="Vidéo introuvable.")