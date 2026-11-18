from fastapi import APIRouter, File, UploadFile, HTTPException
from services.inference import run_diagnosis_pipeline

router = APIRouter()

@router.post("/scan")
def scan_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        image_bytes = file.file.read()
        result = run_diagnosis_pipeline(image_bytes)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        file.file.close()