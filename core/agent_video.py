import os
import json
import base64
from openai import OpenAI
from scenedetect import detect, ContentDetector, split_video_ffmpeg

def encode_image_to_base64(image_path: str) -> str:
    """Convertit une image locale en chaîne Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_keyframe(image_path: str, client: OpenAI) -> str:
    """Envoie l'image clé au modèle VLM sur Nebius Token Factory."""
    base64_image = encode_image_to_base64(image_path)
    
    prompt = (
        "Analyse cette image de scène vidéo. Indique :"
        "\n1. L'action principale et le sujet."
        "\n2. L'ambiance visuelle et le décor."
        "\n3. Le type de cadrage (gros plan, plan moyen, plan large)."
    )

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de l'analyse VLM : {str(e)}"

def process_video_scenes(video_path: str, output_dir: str = "output_scenes") -> str:
    """Pipeline complet de découpage et d'analyse vidéo."""
    os.makedirs(output_dir, exist_ok=True)
    
    api_key = os.getenv("NEBIUS_API_KEY", "your_nebius_api_key_here")
    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1",
        api_key=api_key
    )

    print(f"🔍 Détection des scènes pour : {video_path}")
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    print(f"✅ {len(scene_list)} scènes détectées.")

    print("✂️ Découpage vidéo via FFmpeg...")
    split_video_ffmpeg(video_path, scene_list, output_dir=output_dir)

    results = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        keyframe_path = os.path.join(output_dir, f"scene_{i+1:03d}-01.jpg")

        print(f"🧠 Analyse de la scène {i+1}/{len(scene_list)} ({start_time} -> {end_time})...")
        
        description = "Clé API Nebius non configurée."
        if api_key != "your_nebius_api_key_here" and os.path.exists(keyframe_path):
            description = analyze_keyframe(keyframe_path, client)

        results.append({
            "scene_id": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "keyframe_path": keyframe_path,
            "vlm_analysis": description
        })

    json_output = os.path.join(output_dir, "video_analysis.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return json_output

if __name__ == "__main__":
    # Test stub
    sample_video = "sample.mp4"
    if os.path.exists(sample_video):
        process_video_scenes(sample_video)
    else:
        print(f"Placez un fichier '{sample_video}' pour tester le script.")
