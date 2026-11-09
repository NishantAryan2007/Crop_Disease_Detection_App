let stream = null;
let currentBlob = null;
let currentMode = 'upload';

// Initialize Drag & Drop Listeners once DOM loads
document.addEventListener('DOMContentLoaded', () => {
    const uploadUI = document.getElementById('uploadUI');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadUI.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadUI.addEventListener(eventName, () => uploadUI.classList.add('drag-over'));
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadUI.addEventListener(eventName, () => uploadUI.classList.remove('drag-over'));
    });

    uploadUI.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            currentBlob = file;
            showPreview(URL.createObjectURL(file));
        } else {
            alert("Please drop a valid JPG or PNG image.");
        }
    });
});

async function switchMode(mode) {
    currentMode = mode;
    const btnUpload = document.getElementById('btnUploadMode');
    const btnCamera = document.getElementById('btnCameraMode');
    const uploadUI = document.getElementById('uploadUI');
    const cameraUI = document.getElementById('cameraUI');
    
    clearCurrentScan(); 

    if (mode === 'upload') {
        btnUpload.classList.add('active');
        btnCamera.classList.remove('active');
        uploadUI.classList.add('active');
        cameraUI.classList.remove('active');
        stopCamera();
    } else {
        btnCamera.classList.add('active');
        btnUpload.classList.remove('active');
        cameraUI.classList.add('active');
        uploadUI.classList.remove('active');
        await startCamera();
    }
}

function clearCurrentScan() {
    currentBlob = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('preview').classList.remove('active');
    document.getElementById('actionBar').classList.remove('active');
    document.getElementById('mainContent').classList.remove('expanded');
    document.getElementById('rConfBar').style.width = '0%';
}

function resetView() {
    clearCurrentScan();
    if (currentMode === 'camera') {
        document.getElementById('cameraUI').classList.add('active');
        startCamera();
    } else {
        document.getElementById('uploadUI').classList.add('active');
        stopCamera();
    }
}

async function startCamera() {
    if (stream) stopCamera();
    const video = document.getElementById('webcam');
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
    } catch (err) {
        alert("Camera unavailable or permission denied.");
        switchMode('upload');
    }
}

function stopCamera() {
    if (stream) { 
        stream.getTracks().forEach(track => track.stop()); 
        stream = null; 
    }
    const video = document.getElementById('webcam');
    if (video) {
        video.srcObject = null; 
    }
}

function capturePhoto() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth; 
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(blob => {
        currentBlob = blob;
        showPreview(URL.createObjectURL(blob));
    }, 'image/jpeg', 0.9);
}

function handleFileUpload() {
    const file = document.getElementById('imageInput').files[0];
    if (file) {
        currentBlob = file;
        showPreview(URL.createObjectURL(file));
    }
}

function showPreview(url) {
    const preview = document.getElementById('preview');
    preview.src = url;
    
    document.getElementById('uploadUI').classList.remove('active');
    document.getElementById('cameraUI').classList.remove('active');
    preview.classList.add('active');
    document.getElementById('actionBar').classList.add('active');
    
    stopCamera();
}

async function runScan() {
    if (!currentBlob) return;

    const btn = document.getElementById('scanBtn');
    const loader = document.getElementById('loader');
    const mainContent = document.getElementById('mainContent');
    const resultCard = document.querySelector('.result-card');
    const statusBadge = document.querySelector('.status-badge');
    const riskEl = document.getElementById('rRisk');
    const confBar = document.getElementById('rConfBar');

    btn.disabled = true;
    loader.classList.add('active'); 
    confBar.style.width = '0%';

    const formData = new FormData();
    formData.append('file', currentBlob, 'capture.jpg');

    try {
        const response = await fetch('http://127.0.0.1:8000/api/v1/scan', {
            method: 'POST',
            body: formData
        });
        
        const json = await response.json();

        if (!response.ok) {
            alert("API Error: " + (json.detail || JSON.stringify(json)));
            return;
        }
        
        // 1. Handle YOLO Bouncer Rejection
        if (json.status === "rejected") {
            document.getElementById('rPathology').innerText = "Scan Terminated";
            document.getElementById('rConfText').innerText = '100%';
            document.getElementById('rClass').innerText = "Human/Non-Plant";
            riskEl.innerText = 'N/A';

            resultCard.classList.remove('state-danger', 'state-warning');
            resultCard.classList.add('state-danger');
            statusBadge.innerText = 'RESOURCE PARASITE BLOCKED';
            riskEl.style.color = '#ef4444';

            mainContent.classList.add('expanded');
            setTimeout(() => {
                confBar.style.width = '100%';
            }, 600);
            return;
        }

        // 2. Handle Valid Plant / Soil Result
        document.getElementById('rPathology').innerText = json.health_status;
        document.getElementById('rConfText').innerText = json.identification_confidence + '%';
        document.getElementById('rClass').innerText = json.plant_type;
        riskEl.innerText = json.pathology_confidence + '%';

        resultCard.classList.remove('state-danger', 'state-warning');
        
        if (json.plant_type === "Soil" || json.is_weed) {
            resultCard.classList.add('state-warning');
            statusBadge.innerText = json.plant_type === "Soil" ? 'SOIL SAMPLE' : 'NON-TARGET PLANT';
            riskEl.style.color = '#f59e0b';
        } else if (json.health_status === "Fungal" || json.health_status === "Bacterial_Viral") {
            resultCard.classList.add('state-danger');
            statusBadge.innerText = 'DISEASE DETECTED';
            riskEl.style.color = '#ef4444';
        } else {
            statusBadge.innerText = 'DIAGNOSIS COMPLETE';
            riskEl.style.color = '#00b87c';
        }

        mainContent.classList.add('expanded');
        
        setTimeout(() => {
            confBar.style.width = json.identification_confidence + '%';
        }, 600);

    } catch (err) {
        alert("Server connection failed. Make sure FastAPI is running on port 8000.");
        console.error("Actual error details:", err);
    } finally {
        btn.disabled = false;
        loader.classList.remove('active');
    }
}