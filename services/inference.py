import ctypes
import gc
import threading
import numpy as np
import onnxruntime as ort
from PIL import ImageOps
from utils.image_prep import load_pil_image, prepare_keras_tensor

# Lock ensures multiple simultaneous users never collide
inference_lock = threading.Lock()

# Disable dynamic memory arena growth so RAM stays flat (~115 MB total) across unlimited scans
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.enable_cpu_mem_arena = False
sess_options.enable_mem_pattern = False

print("Initializing ML Service: Loading all 3 ONNX Models into memory...")
yolo_session = ort.InferenceSession(
    "Models/yolov8n.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)
yolo_input_name = yolo_session.get_inputs()[0].name

id_session = ort.InferenceSession(
    "Models/weed_efficientnet.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)
id_input_name = id_session.get_inputs()[0].name

pathology_session = ort.InferenceSession(
    "Models/soybean_pathology.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)
pathology_input_name = pathology_session.get_inputs()[0].name

IDENTIFICATION_CLASSES = ['Broadleaf', 'Grass', 'Soil', 'Soybean']
PATHOLOGY_CLASSES = ['Bacterial_Viral', 'Fungal', 'Healthy']
BLOCKED_PARASITES = {
    0, 1, 2, 3, 7,
    14, 15, 16, 17, 18, 19,
    24, 25, 39, 41,
    56, 63, 67, 73
}
YOLO_CONF_THRESHOLD = 0.40


def force_ram_cleanup():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def run_yolo_bouncer_onnx(pil_image) -> bool:
    padded_img = ImageOps.pad(pil_image.convert("RGB"), (640, 640), color=(114, 114, 114))
    img_array = np.asarray(padded_img, dtype=np.float32) / 255.0
    input_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]

    outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
    class_scores = outputs[0][0][4:, :]

    max_scores = np.max(class_scores, axis=0)
    class_ids = np.argmax(class_scores, axis=0)
    detected_classes = set(class_ids[max_scores >= YOLO_CONF_THRESHOLD].tolist())

    return bool(detected_classes & BLOCKED_PARASITES)


def run_diagnosis_pipeline(image_bytes: bytes) -> dict:
    with inference_lock:
        pil_image = load_pil_image(image_bytes)

        # STAGE 1: The YOLO Bouncer
        if run_yolo_bouncer_onnx(pil_image):
            del pil_image
            force_ram_cleanup()
            return {
                "status": "rejected",
                "message": "Resource Parasite blocked. Human, animal, or non-plant object detected.",
                "is_weed": False
            }

        # STAGE 2: Preprocess for EfficientNet
        processed_tensor = prepare_keras_tensor(pil_image)
        del pil_image

        # STAGE 3: Plant Identification
        id_predictions = id_session.run(None, {id_input_name: processed_tensor})[0][0]
        id_class_index = int(np.argmax(id_predictions))
        plant_type = IDENTIFICATION_CLASSES[id_class_index]
        plant_confidence = float(id_predictions[id_class_index])
        soybean_prob = float(id_predictions[3])  # Index 3 is 'Soybean'

        # STAGE 4: Pathology Engine
        # Run if Soybean is #1, OR if classified as Broadleaf/Soil when Soybean has >=15% probability
        if plant_type == "Soybean" or (plant_type in ["Broadleaf", "Soil"] and soybean_prob >= 0.15):
            disease_preds = pathology_session.run(None, {pathology_input_name: processed_tensor})[0][0]
            disease_index = int(np.argmax(disease_preds))
            disease_type = PATHOLOGY_CLASSES[disease_index]
            disease_confidence = float(disease_preds[disease_index])

            # Accept as Soybean if it was already #1, or if pathology strongly detects a disease (>=75%)
            if plant_type == "Soybean" or (disease_type != "Healthy" and disease_confidence >= 0.75):
                del processed_tensor
                force_ram_cleanup()

                return {
                    "status": "success",
                    "plant_type": "Soybean",
                    "identification_confidence": round(max(plant_confidence, disease_confidence) * 100, 2),
                    "health_status": disease_type,
                    "pathology_confidence": round(disease_confidence * 100, 2),
                    "is_weed": False
                }

        del processed_tensor
        force_ram_cleanup()

        return {
            "status": "success",
            "plant_type": plant_type,
            "identification_confidence": round(plant_confidence * 100, 2),
            "health_status": "N/A - Not a Soybean",
            "pathology_confidence": 0.0,
            "is_weed": plant_type in ["Broadleaf", "Grass"]
        }