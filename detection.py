import cv2
import numpy as np #numpy est un package qui optimise les array 
from dotenv import load_dotenv
import os
import aiohttp
import asyncio
import torch

load_dotenv()

# Chargement du modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Définition des classes de YOLOv5
CLASSES = model.names

pc_object = {"laptop"}
mouse_object = {"mouse"}

async def display_camera_streams():

    #Initialisation de la capture vidéo par la webcam du pc
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur d'ouverture de la caméra réseau")
        return
    
    frame_count = 0
    while True:
        # Lire une image de la webcam
        ret, frame = cap.read()

        # Vérification de la réussite de la capture de l'image
        if not ret:
            print("erreur dans la capture vidéo")
            break

        # Utilisation de YOLOv5 pour la détection
        results = model(frame)

        # Comptage des occurrences des objets détectés
        counts = {}
        # Récupération des résultats
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            label = CLASSES[int(cls)]
            counts[label] = counts.get(label, 0) + 1

            if label in pc_object:
                color = (0, 255, 0)  # vert
            elif label in mouse_object:
                color = (0, 255, 255)  # jaune
            else:
                color = (255, 255, 255)  # blanc par défaut
            
            # Encadrement des objets détectés et étiquetage
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Affichage de nombre d'occurrences en haut à gauche de rendu vidéo
        y_offset = 30
        for label, count in counts.items():
            if count > 0:
                cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30

        # Affichage de l'image avec les détections
        cv2.imshow('Camera Stream - YOLOv5 Object Detection', frame)

        frame_count += 1

        # Arret du flux vidéo, si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération la capture vidéo et fermeture des fenptres ouvertes
    cap.release()
    cv2.destroyAllWindows()

async def main():
    await display_camera_streams()

if __name__ == "__main__":
    asyncio.run(main())