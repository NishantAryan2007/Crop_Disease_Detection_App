import gc
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from utils.image_prep import load_pil_image, prepare_keras_tensor

# Restrict TensorFlow thread pools to prevent per-core memory arena allocation on Render
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 1. Lazy-Loaded Model References (Allows Uvicorn to bind $PORT immediately on startup)
yolo_bouncer = None
identification_model = None
pathology_model = None

# 2. Class Definitions
IDENTIFICATION_CLASSES = ['Broadleaf', 'Grass', 'Soil', 'Soybean']
PATHOLOGY_CLASSES = ['Bacterial_Viral', 'Fungal', 'Healthy']

# 3. Agricultural & Indoor Noise Blocklist (COCO Class IDs)
BLOCKED_PARASITES = {
    0, 1, 2, 3, 7,
    14, 15, 16, 17, 18, 19,
    24, 25, 39, 41,
    56, 63, 67, 73
}
YOLO_CONF_THRESHOLD = 0.40


def get_yolo_bouncer():
    global yolo_bouncer
    if yolo_bouncer is None:
        print("Loading YOLOv8n into memory...")
        yolo_bouncer = YOLO("Models/yolov8n.pt")
        gc.collect()
    return yolo_bouncer


def get_identification_model():
    global identification_model
    if identification_model is None:
        print("Loading Weed EfficientNet into memory (compile=False)...")
        identification_model = tf.keras.models.load_model(
            "Models/weed_efficientnet.keras",
            compile=False
        )
        gc.collect()
    return identification_model


def get_pathology_model():
    global pathology_model
    if pathology_model is None:
        print("Loading Soybean Pathology Model into memory (compile=False)...")
        pathology_model = tf.keras.models.load_model(
            "Models/soybean_pathology.keras",
            compile=False
        )
        gc.collect()
    return pathology_model


def run_diagnosis_pipeline(image_bytes: bytes) -> dict:
    """Executes the full YOLO -> Plant ID -> Pathology pipeline."""
    pil_image = load_pil_image(image_bytes)

    # STAGE 1: The YOLO Bouncer
    bouncer = get_yolo_bouncer()
    yolo_results = bouncer.predict(
        pil_image,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False
    )

    for box in yolo_results[0].boxes:
        class_id = int(box.cls[0])
        if class_id in BLOCKED_PARASITES:
            gc.collect()
            return {
                "status": "rejected",
                "message": "Resource Parasite blocked. Human, animal, or non-plant object detected.",
                "is_weed": False
            }

    # STAGE 2: Preprocess for EfficientNet
    processed_tensor = prepare_keras_tensor(pil_image)

    # STAGE 3: Plant Identification (Direct forward pass avoids model.predict() memory leak)
    id_model = get_identification_model()
    id_predictions = id_model(processed_tensor, training=False).numpy()
    id_class_index = int(np.argmax(id_predictions[0]))
    plant_type = IDENTIFICATION_CLASSES[id_class_index]
    plant_confidence = float(id_predictions[0][id_class_index])

    # STAGE 4: Pathology Engine (Triggered strictly for Soybean)
    if plant_type == "Soybean":
        path_model = get_pathology_model()
        disease_preds = path_model(processed_tensor, training=False).numpy()
        disease_index = int(np.argmax(disease_preds[0]))
        disease_type = PATHOLOGY_CLASSES[disease_index]
        disease_confidence = float(disease_preds[0][disease_index])

        gc.collect()
        return {
            "status": "success",
            "plant_type": plant_type,
            "identification_confidence": round(plant_confidence * 100, 2),
            "health_status": disease_type,
            "pathology_confidence": round(disease_confidence * 100, 2),
            "is_weed": False
        }

    gc.collect()
    # Non-Soybean Output (Broadleaf, Grass, or Soil)
    return {
        "status": "success",
        "plant_type": plant_type,
        "identification_confidence": round(plant_confidence * 100, 2),
        "health_status": "N/A - Not a Soybean",
        "pathology_confidence": 0.0,
        "is_weed": plant_type in ["Broadleaf", "Grass"]
    }