import cv2
import os

def extract_key_frames(video_path: str, output_folder: str = "frames", frame_interval: int = 30):
    """Découpe la vidéo en images-clés (ex: 1 image par seconde) pour le VLM."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        count += 1
        
    cap.release()
    print(f"✅ Extraction terminée : {saved_count} images clés sauvegardées dans '{output_folder}/'.")

if __name__ == "__main__":
    extract_key_frames("combat_test.mp4")