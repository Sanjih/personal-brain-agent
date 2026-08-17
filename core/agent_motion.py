# core/agent_motion.py
import os
import cv2
import json
import subprocess
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT_COMBAT_COACH = """
Tu es un entraîneur principal de Boxe et de MMA professionnel.
Analyse l'action selon les 5 piliers :
1. POSTURE & GARDE (Hauteur des poignets / protection du menton)
2. BIOMÉCANIQUE & APPUIS (Transfert de poids, encrage au sol)
3. DYNAMIQUE DE FRAPPE / DÉFENSE (Extension des membres, retour en garde)
4. ERREURS TACTIQUES & OUVERTURES (Flanches, ouvertures identifiées)
5. CONSEIL DU COACH (1 consigne ou drill immédiat)
"""

def get_video_duration_seconds(video_path: str) -> float:
    """Calcule la durée exacte de la vidéo avec ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return float(output.strip())

print("✅ Script de combat prêt à recevoir la clé API Nebius !")
