import os
import json
import base64
from openai import OpenAI
from scenedetect import detect, ContentDetector, split_video_ffmpeg

# --- CONFIGURATION ---
VIDEO_PATH = "sample_movie.mp4"       # Chemin de votre fichier vidéo
OUTPUT_DIR = "output_scenes"          # Dossier de sortie
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "VOTRE_CLE_NEBIUS")

# Initialisation du client Nebius (API OpenAI-compatible)
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1",
    api_key=NEBIUS_API_KEY
)

def encode_image_to_base64(image_path: str) -> str:
    """Convertit une image locale en chaîne Base64 pour l'API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_keyframe_with_nebius(image_path: str) -> str:
    """Envoie l'image clé à un modèle multimodal sur Nebius Token Factory."""
    base64_image = encode_image_to_base64(image_path)
    
    prompt = (
        "Analyse cette image de scène de film. Indique : "
        "1. L'action principale. "
        "2. L'ambiance et l'éclairage. "
        "3. Le type de plan (gros plan, plan moyen, plan large). "
        "4. Les objets ou personnages clés."
    )

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",  # Modèle multimodal
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de l'analyse IA : {str(e)}"

def process_video_pipeline(video_path: str):
    print(f"🎬 Début du traitement de : {video_path}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Étape 1 : Détection automatique des scènes
    print("🔍 Détection des coupures de scènes...")
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    print(f"✅ {len(scene_list)} scènes détectées.")

    # Étape 2 : Découpage physique du fichier vidéo avec FFmpeg
    print("✂️ Découpage de la vidéo en fichiers individuels...")
    split_video_ffmpeg(video_path, scene_list, output_dir=OUTPUT_DIR)

    # Étape 3 : Extraction et analyse IA de chaque scène
    structured_data = []

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        
        # Nom de l'image de prévisualisation (extraite par PySceneDetect)
        keyframe_filename = f"{OUTPUT_DIR}/scene_{i+1:03d}-01.jpg"
        
        print(f"\n🧠 Analyse IA de la Scène {i+1}/{len(scene_list)} [{start_time} -> {end_time}]...")
        
        description = "Clé d'API non configurée"
        if NEBIUS_API_KEY != "VOTRE_CLE_NEBIUS" and os.path.exists(keyframe_filename):
            description = analyze_keyframe_with_nebius(keyframe_filename)

        scene_data = {
            "scene_id": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "keyframe_path": keyframe_filename,
            "ai_analysis": description
        }
        structured_data.append(scene_data)

    # Étape 4 : Sauvegarde du rapport JSON
    json_output_path = os.path.join(OUTPUT_DIR, "scenes_summary.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Pipeline terminé ! Rapport enregistré dans : {json_output_path}")

if __name__ == "__main__":
    # Assurez-vous d'avoir un fichier vidéo de test nommé "sample_movie.mp4"
    if os.path.exists(VIDEO_PATH):
        process_video_pipeline(VIDEO_PATH)
    else:
        print(f"⚠️ Veuillez placer un fichier vidéo nommé '{VIDEO_PATH}' dans le répertoire.")
