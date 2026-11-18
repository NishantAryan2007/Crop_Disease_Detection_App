from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import weed_router

app = FastAPI(title = "Crop Disease Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], 
    allow_credentials = False,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(weed_router.router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "online"}