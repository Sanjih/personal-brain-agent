from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
import math
import numpy as np
from ultralytics import YOLO
import database as db

db.init_db()

app = FastAPI(
    title="Motion Analysis API",
    description="API d'analyse biomécanique pour Sports de Combat et Danse.",
    version="2.1.0"
)

model = YOLO("yolov8n-pose.pt")

def calculate_duration(file_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return float(output.strip())

# --- NOUVEAU : Fonctions de calculs géométriques biomécaniques ---

def calculate_angle(a, b, c):
    """Calcule l'angle formé par 3 points (ex: Épaule-Coude-Poignet)."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_biomechanics(results):
    """Analyse les angles et la posture moyenne sur l'ensemble des images."""
    left_arm_angles = []
    foot_distances = []
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0].cpu().numpy() # Premier combattant/danseur détecté
            
            # Vérifier qu'on a assez de points visibles (indices COCO)
            if len(kpts) >= 17:
                # Points : 5=Épaule G, 7=Coude G, 9=Poignet G
                shoulder_l, elbow_l, wrist_l = kpts[5][:2], kpts[7][:2], kpts[9][:2]
                # Points : 15=Cheville G, 16=Cheville D
                ankle_l, ankle_r = kpts[15][:2], kpts[16][:2]
                
                # Calcul de l'angle du coude gauche
                if np.all(shoulder_l) and np.all(elbow_l) and np.all(wrist_l):
                    angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
                    left_arm_angles.append(angle)
                
                # Écartement des pieds
                if np.all(ankle_l) and np.all(ankle_r):
                    dist = np.linalg.norm(ankle_l - ankle_r)
                    foot_distances.append(dist)
                    
    avg_elbow_angle = float(np.mean(left_arm_angles)) if left_arm_angles else 90.0
    avg_foot_dist = float(np.mean(foot_distances)) if foot_distances else 0.0
    
    return avg_elbow_angle, avg_foot_dist

def generate_coach_feedback(mode: str, avg_elbow_angle: float, avg_foot_dist: float):
    """Génère un rapport de coaching personnalisé et actionnable."""
    advices = []
    score = 85 # Score de base
    
    if mode == "combat":
        if avg_elbow_angle > 120:
            advices.append({"type": "urgent", "text": "Garde trop ouverte : Resserre ton coude gauche contre tes côtes pour fermer ton flanc."})
            score -= 10
        else:
            advices.append({"type": "good", "text": "Bonne protection du visage : Ton coude reste bien armé."})
            
        if avg_foot_dist < 50:
            advices.append({"type": "warning", "text": "Appuis trop serrés : Écarte légèrement tes pieds pour gagner en stabilité lors des frappes."})
            score -= 5
        else:
            advices.append({"type": "good", "text": "Excellente base d'appuis : Bonne répartition du poids au sol."})
            
    else: # Mode Danse
        if avg_elbow_angle < 100:
            advices.append({"type": "warning", "text": "Manque d'amplitude : Tends davantage les bras lors des extensions de ligne."})
            score -= 8
        else:
            advices.append({"type": "good", "text": "Excellente ouverture de bras : Mouvements très expressifs."})
            
        advices.append({"type": "good", "text": "Fluide et synchrone : Le rythme est bien tenu sur la séquence."})

    return max(score, 50), advices

# --- Fonctions de traitement inchangées avec ajout des métriques ---

def process_video_analysis(file: UploadFile, x_api_key: str, mode: str):
    user = db.get_user_by_api_key(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Clé API non autorisée ou invalide.")
    
    prefix = "combat" if mode == "combat" else "dance"
    temp_filename = f"temp_{prefix}_{file.filename}"
    
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        duration_sec = calculate_duration(temp_filename)
        duration_min = round(duration_sec / 60, 2)
        
        if user["balance_minutes"] < duration_min:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise HTTPException(status_code=402, detail=f"Solde insuffisant.")
            
        output_dir = f"output_{mode}"
        results = model.predict(
            source=temp_filename, 
            save=True, 
            project=output_dir, 
            name="result", 
            exist_ok=True, 
            conf=0.5 if mode == "combat" else 0.4, 
            vid_stride=3
        )
        
        # Calcul des métriques & conseils
        avg_elbow_angle, avg_foot_dist = analyze_biomechanics(results)
        score, coach_advices = generate_coach_feedback(mode, avg_elbow_angle, avg_foot_dist)
        
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
            "coach_advices": coach_advices,
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
    return {"status": "online", "message": "API Motion Analysis 2.1"}

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