import os
import subprocess
import json

# Simulation d'une base de données d'utilisateurs (Ex: Supabase, PostgreSQL ou SQLite)
MOCK_USER_DATABASE = {
    "user_123": {
        "name": "Alice",
        "credit_balance_minutes": 45.0  # L'utilisateur a 45 minutes de crédits
    }
}

def get_video_duration_minutes(video_path: str) -> float:
    """
    Calcule la durée exacte d'un fichier vidéo en minutes
    en utilisant ffprobe (inclus avec FFmpeg).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Fichier introuvable : {video_path}")

    command = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        duration_seconds = float(output.strip())
        duration_minutes = round(duration_seconds / 60.0, 2)
        return duration_minutes
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la lecture de la durée vidéo : {str(e)}")

def process_video_with_credit_check(user_id: str, video_path: str):
    """
    Vérifie le solde de l'utilisateur, déduit le crédit et lance l'analyse.
    """
    # 1. Vérification de l'existence de l'utilisateur
    user = MOCK_USER_DATABASE.get(user_id)
    if not user:
        return {"status": "error", "message": "Utilisateur non trouvé."}

    # 2. Calcul de la durée de la vidéo
    try:
        video_duration = get_video_duration_minutes(video_path)
    except Exception as err:
        return {"status": "error", "message": str(err)}

    current_balance = user["credit_balance_minutes"]

    print(f"👤 Client : {user['name']}")
    print(f"⏱️ Durée de la vidéo : {video_duration} minute(s)")
    print(f"💳 Solde actuel : {current_balance} minute(s)")

    # 3. Contrôle du solde
    if current_balance < video_duration:
        missing_credits = round(video_duration - current_balance, 2)
        return {
            "status": "denied",
            "message": f"Solde insuffisant. Il vous manque {missing_credits} minutes de crédit.",
            "required_credits": video_duration,
            "current_balance": current_balance
        }

    # 4. Déduction des crédits avant le traitement
    new_balance = round(current_balance - video_duration, 2)
    user["credit_balance_minutes"] = new_balance
    print(f"✅ Déduction effectuée ! Nouveau solde : {new_balance} minute(s).")

    # 5. Lancement du pipeline Nebius (Importation de votre module)
    print("🚀 Lancement de l'analyse sur Nebius GPU...")
    # Remplacez cette ligne par l'appel réel à votre script :
    # result = process_video_scenes(video_path)
    
    return {
        "status": "success",
        "message": "Traitement démarré avec succès.",
        "processed_duration": video_duration,
        "remaining_balance": new_balance
    }

# --- DÉMONSTRATION D'UTILISATION ---
if __name__ == "__main__":
    sample_file = "sample_movie.mp4"
    
    # Simulation d'un appel API pour l'utilisateur 123
    if os.path.exists(sample_file):
        response = process_video_with_credit_check("user_123", sample_file)
        print("\n--- Réponse envoyée au Frontend ---")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print(f"⚠️ Créez un fichier de test '{sample_file}' pour tester la vérification.")
