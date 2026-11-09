import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from utils.image_prep import load_pil_image, prepare_keras_tensor

# 1. Load Models Once on Server Startup
print("Initializing ML Service: Loading Models into memory...")
yolo_bouncer = YOLO("Models/yolov8n.pt")
identification_model = tf.keras.models.load_model("Models/weed_efficientnet.keras")
pathology_model = tf.keras.models.load_model("Models/soybean_pathology.keras")

# 2. Class Definitions
IDENTIFICATION_CLASSES = ['Broadleaf', 'Grass', 'Soil', 'Soybean']
PATHOLOGY_CLASSES = ['Bacterial_Viral', 'Fungal', 'Healthy']

# 3. Agricultural & Indoor Noise Blocklist (COCO Class IDs)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 7: truck/tractor
# 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow
# 24: backpack, 25: umbrella, 39: bottle, 41: cup
# 56: chair, 63: laptop, 67: cell phone, 73: book
BLOCKED_PARASITES = {
    0, 1, 2, 3, 7,
    14, 15, 16, 17, 18, 19,
    24, 25, 39, 41,
    56, 63, 67, 73
}
YOLO_CONF_THRESHOLD = 0.40

def run_diagnosis_pipeline(image_bytes: bytes) -> dict:
    """Executes the full YOLO -> Plant ID -> Pathology pipeline."""
    # Decode raw bytes once for the entire pipeline
    pil_image = load_pil_image(image_bytes)

    # STAGE 1: The YOLO Bouncer
    yolo_results = yolo_bouncer.predict(
        pil_image, 
        conf=YOLO_CONF_THRESHOLD, 
        verbose=False
    )
    
    for box in yolo_results[0].boxes:
        class_id = int(box.cls[0])
        if class_id in BLOCKED_PARASITES:
            return {
                "status": "rejected",
                "message": "Resource Parasite blocked. Human, animal, or non-plant object detected.",
                "is_weed": False
            }

    # STAGE 2: Preprocess for EfficientNet
    processed_tensor = prepare_keras_tensor(pil_image)

    # STAGE 3: Plant Identification
    id_predictions = identification_model.predict(processed_tensor, verbose=0)
    id_class_index = int(np.argmax(id_predictions[0]))
    plant_type = IDENTIFICATION_CLASSES[id_class_index]
    plant_confidence = float(id_predictions[0][id_class_index])

    # STAGE 4: Pathology Engine (Triggered strictly for Soybean)
    if plant_type == "Soybean":
        disease_preds = pathology_model.predict(processed_tensor, verbose=0)
        disease_index = int(np.argmax(disease_preds[0]))
        disease_type = PATHOLOGY_CLASSES[disease_index]
        disease_confidence = float(disease_preds[0][disease_index])

        return {
            "status": "success",
            "plant_type": plant_type,
            "identification_confidence": round(plant_confidence * 100, 2),
            "health_status": disease_type,
            "pathology_confidence": round(disease_confidence * 100, 2),
            "is_weed": False
        }

    # Non-Soybean Output (Broadleaf, Grass, or Soil)
    return {
        "status": "success",
        "plant_type": plant_type,
        "identification_confidence": round(plant_confidence * 100, 2),
        "health_status": "N/A - Not a Soybean",
        "pathology_confidence": 0.0,
        "is_weed": plant_type in ["Broadleaf", "Grass"]
    }