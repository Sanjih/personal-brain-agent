from ultralytics import YOLO

# Télécharge automatiquement le modèle de posture léger lors du premier lancement
model = YOLO("yolov8n-pose.pt")

# Analyse la vidéo et sauvegarde le résultat avec les squelettes dessinés
results = model.predict(source="combat_test.mp4", save=True, conf=0.5)
print("✅ Analyse de posture terminée !")