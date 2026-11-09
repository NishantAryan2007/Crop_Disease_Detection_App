import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from keras.utils import img_to_array

TARGET_SIZE = (224, 224)

def load_pil_image(image_bytes: bytes) -> Image.Image:
    """Safely converts raw request bytes into an RGB PIL Image."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Invalid image file. Please upload a valid JPG or PNG.")

def prepare_keras_tensor(pil_image: Image.Image) -> np.ndarray:
    """Resizes a PIL image and formats it into a batch tensor (1, 224, 224, 3)."""
    resized_image = pil_image.resize(TARGET_SIZE)
    img_array = img_to_array(resized_image)
    return np.expand_dims(img_array, axis = 0)