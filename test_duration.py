import os
import subprocess

def get_video_duration(video_path: str) -> float:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Le fichier '{video_path}' est introuvable à la racine de votre espace de travail.")
        
    cmd = [
        "ffprobe", "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(output.strip())
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur FFprobe : {e.output.decode('utf-8', errors='ignore')}")
        raise

if __name__ == "__main__":
    video_file = "combat_test.mp4"
    try:
        duree = get_video_duration(video_file)
        prix_estime = (duree / 60) * 0.50  # Tarif : 0.50€ / minute
        print(f"✅ Durée exacte : {duree:.2f} secondes")
        print(f"💰 Coût facturé au client : {prix_estime:.3f} €")
    except Exception as err:
        print(err)