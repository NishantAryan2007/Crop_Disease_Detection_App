import ctypes
import gc
import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO
from utils.image_prep import load_pil_image, prepare_keras_tensor

# Restrict CPU thread pools so PyTorch and TensorFlow don't allocate extra RAM arenas per core
torch.set_num_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Class Definitions
IDENTIFICATION_CLASSES = ['Broadleaf', 'Grass', 'Soil', 'Soybean']
PATHOLOGY_CLASSES = ['Bacterial_Viral', 'Fungal', 'Healthy']

# Agricultural & Indoor Noise Blocklist (COCO Class IDs)
BLOCKED_PARASITES = {
    0, 1, 2, 3, 7,
    14, 15, 16, 17, 18, 19,
    24, 25, 39, 41,
    56, 63, 67, 73
}
YOLO_CONF_THRESHOLD = 0.40


def release_ram():
    """Clears TensorFlow C++ graph, runs Python GC, and forces Linux glibc to return RAM to the OS."""
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def run_diagnosis_pipeline(image_bytes: bytes) -> dict:
    """Executes the YOLO -> Plant ID -> Pathology pipeline sequentially (1 model in RAM at a time)."""
    pil_image = load_pil_image(image_bytes)

    # STAGE 1: The YOLO Bouncer (Load -> Predict -> Unload)
    bouncer = YOLO("Models/yolov8n.pt")
    with torch.no_grad():
        yolo_results = bouncer.predict(
            pil_image,
            conf=YOLO_CONF_THRESHOLD,
            verbose=False
        )
    detected_classes = [int(box.cls[0]) for box in yolo_results[0].boxes]

    del yolo_results
    del bouncer
    release_ram()

    for class_id in detected_classes:
        if class_id in BLOCKED_PARASITES:
            return {
                "status": "rejected",
                "message": "Resource Parasite blocked. Human, animal, or non-plant object detected.",
                "is_weed": False
            }

    # STAGE 2 & 3: Plant Identification (Load -> Predict -> Unload)
    processed_tensor = prepare_keras_tensor(pil_image)

    id_model = tf.keras.models.load_model(
        "Models/weed_efficientnet.keras",
        compile=False
    )
    id_predictions = id_model(processed_tensor, training=False).numpy()
    id_class_index = int(np.argmax(id_predictions[0]))
    plant_type = IDENTIFICATION_CLASSES[id_class_index]
    plant_confidence = float(id_predictions[0][id_class_index])

    del id_predictions
    del id_model
    del processed_tensor
    release_ram()

    # STAGE 4: Pathology Engine (Triggered strictly for Soybean)
    if plant_type == "Soybean":
        processed_tensor = prepare_keras_tensor(pil_image)
        path_model = tf.keras.models.load_model(
            "Models/soybean_pathology.keras",
            compile=False
        )
        disease_preds = path_model(processed_tensor, training=False).numpy()
        disease_index = int(np.argmax(disease_preds[0]))
        disease_type = PATHOLOGY_CLASSES[disease_index]
        disease_confidence = float(disease_preds[0][disease_index])

        del disease_preds
        del path_model
        del processed_tensor
        del pil_image
        release_ram()

        return {
            "status": "success",
            "plant_type": plant_type,
            "identification_confidence": round(plant_confidence * 100, 2),
            "health_status": disease_type,
            "pathology_confidence": round(disease_confidence * 100, 2),
            "is_weed": False
        }

    del pil_image
    release_ram()

    # Non-Soybean Output (Broadleaf, Grass, or Soil)
    return {
        "status": "success",
        "plant_type": plant_type,
        "identification_confidence": round(plant_confidence * 100, 2),
        "health_status": "N/A - Not a Soybean",
        "pathology_confidence": 0.0,
        "is_weed": plant_type in ["Broadleaf", "Grass"]
    }