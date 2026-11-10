import numpy as np
import onnxruntime as ort
import tensorflow as tf
from PIL import ImageOps
from utils.image_prep import load_pil_image, prepare_keras_tensor

# Restrict thread pools to keep memory footprint flat
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 1. Load All Models Once on Server Startup (~270 MB total RAM)
print("Initializing ML Service: Loading ONNX & Keras Models into memory...")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

yolo_session = ort.InferenceSession(
    "Models/yolov8n.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)
yolo_input_name = yolo_session.get_inputs()[0].name

identification_model = tf.keras.models.load_model("Models/weed_efficientnet.keras", compile=False)
pathology_model = tf.keras.models.load_model("Models/soybean_pathology.keras", compile=False)

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


def run_yolo_bouncer_onnx(pil_image) -> bool:
    """Runs YOLOv8n via ONNX Runtime and returns True if a blocked parasite is detected."""
    # Letterbox resize to 640x640 with standard YOLO grey padding (114, 114, 114)
    padded_img = ImageOps.pad(pil_image.convert("RGB"), (640, 640), color=(114, 114, 114))
    img_array = np.asarray(padded_img, dtype=np.float32) / 255.0
    # Convert HWC (640, 640, 3) -> NCHW (1, 3, 640, 640)
    input_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]

    # Output shape is (1, 84, 8400): first 4 rows are box coords, remaining 80 rows are COCO class scores
    outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
    class_scores = outputs[0][0][4:, :]

    max_scores = np.max(class_scores, axis=0)
    class_ids = np.argmax(class_scores, axis=0)
    detected_classes = set(class_ids[max_scores >= YOLO_CONF_THRESHOLD].tolist())

    return bool(detected_classes & BLOCKED_PARASITES)


def run_diagnosis_pipeline(image_bytes: bytes) -> dict:
    """Executes the full YOLO (ONNX) -> Plant ID -> Pathology pipeline."""
    pil_image = load_pil_image(image_bytes)

    # STAGE 1: The YOLO Bouncer
    if run_yolo_bouncer_onnx(pil_image):
        return {
            "status": "rejected",
            "message": "Resource Parasite blocked. Human, animal, or non-plant object detected.",
            "is_weed": False
        }

    # STAGE 2: Preprocess for EfficientNet
    processed_tensor = prepare_keras_tensor(pil_image)

    # STAGE 3: Plant Identification
    id_predictions = identification_model(processed_tensor, training=False).numpy()
    id_class_index = int(np.argmax(id_predictions[0]))
    plant_type = IDENTIFICATION_CLASSES[id_class_index]
    plant_confidence = float(id_predictions[0][id_class_index])

    # STAGE 4: Pathology Engine (Triggered strictly for Soybean)
    if plant_type == "Soybean":
        disease_preds = pathology_model(processed_tensor, training=False).numpy()
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