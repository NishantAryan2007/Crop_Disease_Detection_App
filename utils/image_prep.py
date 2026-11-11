import io
import numpy as np
from PIL import Image

def load_pil_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def prepare_keras_tensor(pil_image: Image.Image) -> np.ndarray:
    resized = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(img_array, axis = 0)