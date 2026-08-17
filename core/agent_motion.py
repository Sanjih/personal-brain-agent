import os
import cv2
import json
import base64
from openai import OpenAI
from ultralytics import YOLO

def anonymize_frame_and_extract_pose(image_path: str, output_masked_path: str, model):
    """
    Détecte le squelette corporel, masque la tête/visage avec un disque noir
    et extrait les coordonnées des articulations.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    results = model(frame)
    pose_data = []

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()

            for person_pts in keypoints:
                # Masquage de la zone de la tête (Nez = index 0)
                if len(person_pts) > 0:
                    head_center = tuple(map(int, person_pts[0]))
                    if head_center != (0, 0):
                        cv2.circle(frame, head_center, 65, (0, 0, 0), -1)

                # Extraction des articulations principales du corps (épaules, coudes, genoux, etc.)
                body_joints = person_pts[5:].tolist() if len(person_pts) >= 5 else []
                pose_data.append({"joints_xy": body_joints})

    cv2.imwrite(output_masked_path, frame)
    return pose_data

def describe_anonymized_motion(masked_image_path: str, client: OpenAI) -> str:
    """Envoie l'image anonymisée à Nebius pour analyser la posture et l'action."""
    with open(masked_image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    prompt = (
        "Cette image a été anonymisée (visage masqué). "
        "Décris uniquement la dynamique du mouvement du corps (posture, marche, course, geste) "
        "et le contexte de l'action sans spéculer sur l'identité."
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
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur d'analyse du mouvement : {str(e)}"

def process_motion_pipeline(image_path: str, output_dir: str = "output_motion") -> dict:
    """Pipeline complet d'anonymisation et de capture de mouvement."""
    os.makedirs(output_dir, exist_ok=True)
    
    api_key = os.getenv("NEBIUS_API_KEY", "your_nebius_api_key_here")
    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1",
        api_key=api_key
    )

    # Modèle YOLOv8 Pose léger
    pose_model = YOLO("yolov8n-pose.pt")
    
    masked_path = os.path.join(output_dir, "anonymized_frame.jpg")
    
    print(f"👤 Detection de la pose et anonymisation de : {image_path}")
    joints_data = anonymize_frame_and_extract_pose(image_path, masked_path, pose_model)
    
    motion_description = "Clé API Nebius non configurée."
    if api_key != "your_nebius_api_key_here":
        print("🧠 Analyse sémantique de l'action sur Nebius...")
        motion_description = describe_anonymized_motion(masked_path, client)

    output_payload = {
        "original_image": image_path,
        "anonymized_image": masked_path,
        "privacy_status": "Anonymized (Face Redacted / Pose Estimation)",
        "detected_persons": len(joints_data),
        "pose_data": joints_data,
        "action_summary": motion_description
    }

    return output_payload

if __name__ == "__main__":
    sample_img = "test_frame.jpg"
    if os.path.exists(sample_img):
        res = process_motion_pipeline(sample_img)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(f"Placez une image '{sample_img}' pour tester le module de mouvement.")
