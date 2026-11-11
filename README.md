# Crop Disease Detection & Weed Identification System

A production-grade agricultural computer vision API and web application engineered for real-time crop identification, weed segmentation, and foliar pathology diagnostics. Built on a **100% ONNX Runtime inference engine**, the system unifies multi-framework deep learning models into a deterministic, memory-bounded microservice architecture.

---

## System Architecture: 3-Stage Cascaded Inference Pipeline

Incoming imagery passes through a sequential, multi-stage inference pipeline designed to maximize diagnostic precision while minimizing unnecessary compute on invalid inputs:

1. **Stage 1 : Out-of-Distribution (OOD) Object Gatekeeper (`YOLOv8n`)**
   * Screens incoming images for non-agricultural objects and environmental noise before triggering downstream classifiers.
   * Immediately halts pipeline execution and returns an early rejection payload if non-botanical anomalies are detected.

2. **Stage 2 : Taxonomic & Weed Classification (`EfficientNet`)**
   * Evaluates validated field imagery across four primary agronomy classes: **`Soybean`**, **`Broadleaf`** (weed), **`Grass`** (weed), and **`Soil`**.
   * Features confidence-aware cross-stage routing to ensure heavily symptomatic crop foliage with severe lesion coverage is accurately routed for pathology inspection rather than prematurely flagged as broadleaf vegetation.

3. **Stage 3 : Foliar Pathology Diagnostics (`Custom CNN / EfficientNet`)**
   * Triggered selectively on verified crop foliage to classify plant health into **`Healthy`**, **`Fungal`**, or **`Bacterial_Viral`** states, returning class probabilities and actionable agronomic metadata.

---

## Systems Engineering & Runtime Optimizations

To achieve sub-second CPU inference and zero memory drift in resource-constrained cloud environments, the backend incorporates several systems-level optimizations:

* **Unified ONNX Execution Graph:** Heterogeneous models trained across PyTorch and TensorFlow/Keras are compiled into static ONNX graphs (`opset 15`), eliminating heavy training framework dependencies and maintaining a flat **~115 MB RAM footprint**.
* **Deterministic Graph Sanitization:** Training-specific stochastic augmentation layers and dropout nodes are stripped prior to serialization, ensuring consistent, deterministic tensor outputs across platforms.
* **Zero-Drift Memory Management:** Configured with disabled CPU memory arena expansion (`enable_cpu_mem_arena = False`) and coupled with explicit Python garbage collection and Linux `libc` heap trimming (`malloc_trim`) after every request lifecycle.
* **Thread-Safe Request Serialization:** Uses mutex-locked inference execution (`threading.Lock`) and pinned single-thread operator pools to guarantee predictable latency and stability under concurrent multi-user traffic.
* **Signal-Preserving Preprocessing:** Implements lightweight, dependency-free image decoding and Bicubic resampling via `Pillow` and `NumPy` to preserve fine-grained foliar lesion textures.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Inference Engine** | ONNX Runtime (`CPUExecutionProvider`), YOLOv8n, EfficientNet |
| **Backend API** | Python 3.11, FastAPI, Uvicorn, Pillow, NumPy |
| **Frontend Client** | HTML5, CSS3, Vanilla JavaScript (Fetch API) |
| **Cloud Infrastructure** | Render (Backend Web Service), Vercel (Frontend Edge Network) |

---

## Project Structure

```text
Crop_Disease_Detection_App/
├── Frontend/
│   ├── index.html               # Client UI layout
│   ├── style.css                # Responsive styling & animations
│   └── app.js                   # API integration & dynamic DOM rendering
├── Models/
│   ├── yolov8n.onnx             # Stage 1: OOD anomaly filter
│   ├── weed_efficientnet.onnx   # Stage 2: Crop & weed classifier
│   └── soybean_pathology.onnx   # Stage 3: Foliar disease classifier
├── routers/
│   └── weed_router.py           # FastAPI route controllers & payload validation
├── services/
│   └── inference.py             # Thread-safe ONNX session manager & pipeline logic
├── utils/
│   └── image_prep.py            # Bicubic tensor preprocessing utilities
├── main.py                      # FastAPI application entrypoint & CORS configuration
├── requirements.txt             # Production Python dependencies
└── .python-version              # Runtime version pin (3.11.9)
```

---

## Local Development Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/NishantAryan2007/Crop_Disease_Detection_App.git](https://github.com/NishantAryan2007/Crop_Disease_Detection_App.git)
cd Crop_Disease_Detection_App
```

### 2. Configure Virtual Environment & Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Start the FastAPI Inference Server
```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Launch the Frontend
Open `Frontend/index.html` in any modern browser (or via VS Code Live Server). For local testing, point `API_ENDPOINT` in `Frontend/app.js` to `http://127.0.0.1:8000/api/v1/scan`.

---

## API Reference

### Health Check
```http
GET /health
```
* **Response (`200 OK`):** `{"status": "online"}`

### Execute Diagnostic Scan
```http
POST /api/v1/scan
Content-Type: multipart/form-data
```
* **Form Parameter:** `file` (Image binary: `.jpg`, `.jpeg`, `.png`)
* **Success Response (`200 OK`):**
```json
{
  "status": "success",
  "plant_type": "Soybean",
  "identification_confidence": 94.62,
  "health_status": "Fungal",
  "pathology_confidence": 91.18,
  "is_weed": false
}
```
* **Filtered Anomaly Response (`200 OK`):**
```json
{
  "status": "rejected",
  "message": "Resource Parasite blocked. Human, animal, or non-plant object detected.",
  "is_weed": false
}
```