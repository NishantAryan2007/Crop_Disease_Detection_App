from fastapi import APIRouter, UploadFile, File, HTTPException
from services.inference import run_diagnosis_pipeline

router = APIRouter()

@router.post("/api/v1/scan")
async def predict_plant(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        result = run_diagnosis_pipeline(contents)
        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")