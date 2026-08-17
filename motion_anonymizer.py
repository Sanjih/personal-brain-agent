import os
import cv2
import json
import base64
from openai import OpenAI
from ultralytics import YOLO

# Configuration Nebius
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "VOTRE_CLE_NEBIUS")
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1",
    api_key=NEBIUS_API_KEY
)

# Chargement du modèle de pose YOLOv8
pose_model = YOLO("yolov8n-pose.pt")

def process_and_anonymize_frame(image_path: str, output_masked_path: str):
    """
    Extrait les points de pose (squelette) et masque la zone du visage/tête
    pour garantir une analyse 100 % anonyme.
    """
    frame = cv2.imread(image_path)
    results = pose_model(frame)
    
    pose_data = []

    for result in results:
        if result.keypoints is not None:
            # Récupération des coordonnées X, Y des articulations
            keypoints = result.keypoints.xy.cpu().numpy()
            
            for person_pts in keypoints:
                person_pose = {}
                # Les 5 premiers points YOLO représentent : nez, œil G, œil D, oreille G, oreille D
                if len(person_pts) >= 5:
                    # Anonymisation : dessiner un cercle noir sur toute la zone de la tête
                    head_center = tuple(map(int, person_pts[0])) # Nez
                    if head_center != (0, 0):
                        cv2.circle(frame, head_center, 60, (0, 0, 0), -1)

                # Sauvegarde des articulations du corps (épaules, coudes, poignets, hanches, genoux)
                person_pose["body_joints"] = person_pts[5:].tolist()
                pose_data.append(person_pose)

    # Sauvegarde de l'image anonymisée avec visage masqué
    cv2.imwrite(output_masked_path, frame)
    return pose_data

def describe_anonymized_action(masked_image_path: str) -> str:
    """Envoie l'image anonymisée à Nebius Token Factory pour décrire le mouvement."""
    with open(masked_image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    prompt = (
        "Analyse le mouvement et l'action de la personne sur cette image anonymisée. "
        "Décris uniquement la posture, la dynamique du corps (marche, course, saut, position assise) "
        "et le contexte de la scène sans faire de référence à l'identité."
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        max_tokens=200
    )
    return response.choices[0].message.content

# Exemple d'exécution
if __name__ == "__main__":
    test_keyframe = "scene_001.jpg"
    masked_keyframe = "scene_001_anonymized.jpg"

    if os.path.exists(test_keyframe):
        print("1. Extraction de la pose et masquage du visage...")
        joints = process_and_anonymize_frame(test_keyframe, masked_keyframe)
        
        print("2. Envoi de l'image anonymisée à Nebius...")
        description = describe_anonymized_action(masked_keyframe)
        
        # Structure de sortie combinée
        output_payload = {
            "anonymized_image": masked_keyframe,
            "detected_bodies": len(joints),
            "pose_coordinates": joints,
            "action_description": description
        }
        
        print("\n--- RÉSULTAT OBTENU ---")
        print(json.dumps(output_payload, indent=2, ensure_ascii=False))
