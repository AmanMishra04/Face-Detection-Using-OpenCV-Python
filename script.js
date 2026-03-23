const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const engineStatus = document.getElementById('engine-status');
const detectionInfo = document.getElementById('detection-info');
const logList = document.getElementById('log-list');

const scaleInput = document.getElementById('scale-factor');
const neighborsInput = document.getElementById('min-neighbors');
const scaleDisplay = document.getElementById('scale-val');
const neighborsDisplay = document.getElementById('min-val');

const latencyVal = document.getElementById('latency-val');
const targetCount = document.getElementById('target-count');

let isRunning = false;
let stream = null;
let animationId = null;

// Initialize
async function init() {
    addLog("System Initialized. Awaiting User Engagement.");
    setupListeners();
    checkEngine();
}

function setupListeners() {
    startBtn.addEventListener('click', startTracking);
    stopBtn.addEventListener('click', stopTracking);
    
    scaleInput.addEventListener('input', (e) => {
        scaleDisplay.textContent = e.target.value;
    });
    
    neighborsInput.addEventListener('input', (e) => {
        neighborsDisplay.textContent = e.target.value;
    });
}

function addLog(msg) {
    const p = document.createElement('p');
    p.className = 'log-entry';
    p.textContent = `> ${new Date().toLocaleTimeString()} :: ${msg}`;
    logList.prepend(p);
}

async function checkEngine() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.status) {
            engineStatus.textContent = 'ONLINE';
            engineStatus.style.color = '#00ffaa';
            addLog("Neural Engine Connection Established.");
        }
    } catch (e) {
        engineStatus.textContent = 'WARNING: OFFLINE';
        engineStatus.style.color = '#ff4b4b';
        addLog("Neural Engine Connection Error. Using local fallback mode.");
    }
}

async function startTracking() {
    if (isRunning) return;
    
    try {
        addLog("Requesting Camera Access...");
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isRunning = true;
            addLog("Camera Synchronized. Resolving at " + video.videoWidth + "x" + video.videoHeight);
            processFrame();
        };
        
        startBtn.textContent = "TRACKING ACTIVE";
        startBtn.style.background = "#7000ff";
        startBtn.style.color = "white";

    } catch (err) {
        addLog("CRITICAL: Camera Access Denied.");
        console.error(err);
    }
}

function stopTracking() {
    isRunning = false;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (animationId) cancelAnimationFrame(animationId);
    
    startBtn.textContent = "ENGAGE NEURAL TRACKING";
    startBtn.style.background = "#00f2ff";
    startBtn.style.color = "#000";
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    addLog("Tactical Session Terminated.");
}

async function processFrame() {
    if (!isRunning) return;

    const startTime = performance.now();
    
    // Capture frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);
    
    const base64Img = tempCanvas.toDataURL('image/jpeg', 0.5);

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Img,
                scaleFactor: scaleInput.value,
                minNeighbors: neighborsInput.value
            })
        });

        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);
        
        drawOverlay(data.faces);
        updateTelemetry(data.faces.length, latency);

    } catch (e) {
        console.error("Detection error:", e);
    }

    if (isRunning) {
        setTimeout(processFrame, 100); // 10 FPS to save bandwidth/credits
    }
}

function drawOverlay(faces) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (faces.length > 0) {
        detectionInfo.textContent = `${faces.length} TARGETS IDENTIFIED`;
        detectionInfo.style.borderLeftColor = '#00ffaa';
    } else {
        detectionInfo.textContent = `NO TARGETS DETECTED`;
        detectionInfo.style.borderLeftColor = '#ff4b4b';
    }

    faces.forEach(face => {
        // Draw main box
        ctx.strokeStyle = '#00f2ff';
        ctx.lineWidth = 3;
        ctx.strokeRect(face.x, face.y, face.width, face.height);
        
        // Draw "corners" for extra tech look
        ctx.fillStyle = '#00f2ff';
        const s = 15;
        ctx.fillRect(face.x - 2, face.y - 2, s, 4);
        ctx.fillRect(face.x - 2, face.y - 2, 4, s);
        
        // Draw ID label
        ctx.font = '12px JetBrains Mono';
        ctx.fillStyle = '#00f2ff';
        ctx.fillText(`ID: T-${Math.floor(face.x)}`, face.x, face.y - 10);
    });
}

function updateTelemetry(count, latency) {
    targetCount.textContent = count;
    latencyVal.textContent = latency + "ms";
    
    if (count > 0) {
        targetCount.style.color = '#00ffaa';
    } else {
        targetCount.style.color = '#ff4b4b';
    }
}

init();
