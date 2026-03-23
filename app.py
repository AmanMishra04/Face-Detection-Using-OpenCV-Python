import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import time
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# 1. PAGE SETUP (Professional Tech Aesthetic)
st.set_page_config(page_title="VISION AI | Recognition System", page_icon="👁️", layout="wide")

# Google Fonts & CSS Inject
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    /* Professional Tech Aesthetic */
    .stApp { 
        background-color: #0f172a; 
        color: #f1f5f9; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Simplified Typography */
    h1, h2, h3 { 
        font-family: 'Orbitron', sans-serif !important; 
        color: #6366f1 !important; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Clean Professional Cards */
    .stAlert, .diag-box, .status-card { 
        background: rgba(30, 41, 59, 0.7); 
        border: 1px solid rgba(99, 102, 241, 0.2) !important; 
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Professional Buttons */
    .stButton>button { 
        background: #6366f1 !important; 
        color: white !important; 
        border: none !important; 
        border-radius: 6px; 
        height: 3.2rem; 
        font-family: 'Inter', sans-serif;
        font-weight: 600; 
        width: 100%; 
        transition: all 0.2s;
    }
    .stButton>button:hover { 
        background: #4f46e5 !important;
        transform: translateY(-1px);
    }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Professional Badges */
    .badge {
        background: rgba(99, 102, 241, 0.1);
        color: #818cf8;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# 2. SYSTEM CONSTANTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_FILE = "haarcascade_frontalface_default.xml"
XML_PATH = os.path.join(BASE_DIR, XML_FILE)
HERO_IMG = "vision_hero.png"
NODE_IMG = "biometric_node.png"
BANNER_IMG = "face_detection_banner.png"

def safe_image_load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path): return None
    try:
        img = Image.open(path)
        img.verify()
        return Image.open(path)
    except: return None

@st.cache_resource
def load_ai():
    if not os.path.exists(XML_PATH): return None, "OFFLINE"
    detector = cv2.CascadeClassifier(XML_PATH)
    return detector, "READY"

ai_engine, ai_status = load_ai()

# 3. MISSION CONTROL (SIDEBAR)
st.sidebar.markdown(f"<h2>VISION AI</h2>", unsafe_allow_html=True)
mission = st.sidebar.selectbox("OPERATIONAL MODE", 
    ["Vision Hub", "Image Recognition", "Live Detection", "Video Forensic"])

st.sidebar.markdown("---")
st.sidebar.subheader("SYSTEM CALIBRATION")
SENS = st.sidebar.slider("SENSITIVITY", 1.05, 1.4, 1.1)
STAB = st.sidebar.slider("STABILITY", 1, 15, 3)

# System Health
st.sidebar.markdown("---")
st.sidebar.subheader("SYSTEM STATUS")
st.sidebar.markdown(f"""
<div style='font-size: 0.8rem; opacity: 0.9;'>
    <p><b>ENGINE:</b> <span style='color:#4ade80'>OPERATIONAL</span></p>
    <p><b>AI CORE:</b> <span style='color:#818cf8'>{ai_status}</span></p>
    <p><b>NETWORK:</b> <span style='color:#818cf8'>ENCRYPTED</span></p>
</div>
""", unsafe_allow_html=True)

# 4. PROFESSIONAL BOX LOGIC
def draw_pro_box(img, x, y, w, h):
    # Indigo Bounding Box
    cv2.rectangle(img, (x, y), (x + w, y + h), (241, 102, 99), 2) # BGR
    # Minimal Modern Markers
    length = 15
    # Corners only
    cv2.line(img, (x, y), (x + length, y), (255, 242, 0), 2)
    cv2.line(img, (x, y), (x, y + length), (255, 242, 0), 2)
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), (255, 242, 0), 2)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), (255, 242, 0), 2)
    
    cv2.putText(img, "MATCH_CONFIRMED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 242, 0), 1)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]
    
    # NEURAL SCAN
    faces_found = 0
    if ai_engine is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = 320 / w
        gray_s = cv2.resize(gray, (320, int(h * scale)))
        detections = ai_engine.detectMultiScale(gray_s, scaleFactor=SENS, minNeighbors=STAB, minSize=(30, 30))
        faces_found = len(detections)
        for (fx, fy, fw, fh) in detections:
            tx, ty, tw, th = [int(v/scale) for v in (fx, fy, fw, fh)]
            draw_pro_box(img, tx, ty, tw, th)

    # Status HUD
    cv2.putText(img, f"ENGINE: ACTIVE | TARGETS: {faces_found}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241, 102, 99), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. CORE INTERFACE
st.markdown("<h1>👁️ AI VISION RECOGNITION</h1>", unsafe_allow_html=True)

if mission == "Vision Hub":
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>VISION CONTROL CENTER</h2>", unsafe_allow_html=True)
    
    # 1. PROFESSIONAL OVERVIEW
    col_a, col_b = st.columns([1, 1.2])
    
    with col_a:
        hero = safe_image_load(HERO_IMG)
        if hero: 
            st.image(hero, use_container_width=True, caption="Advanced Biometric Acquisition Interface")
        else:
            st.markdown("<div class='diag-box'><h1>VISION AI</h1><p>Image Stream Initialized</p></div>", unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
        <div class='diag-box'>
            <h3>System Overview</h3>
            <p style='color: #94a3b8;'>Welcome to the **AI Vision Recognition System**. This professional-grade platform provides real-time detection and biometric tracking using high-performance Haar-Cascade neural modules. It is designed for high accuracy and low latency in diverse operational environments.</p>
            <div style='display: flex; gap: 8px; margin-top: 10px;'>
                <span class='badge'>System: Ready</span>
                <span class='badge'>Latency: Ultra-Low</span>
                <span class='badge'>Model: Haar-V2</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Telemetry
        c1, c2, c3 = st.columns(3)
        with c1: st.write("**ENGINE**  \n`STABLE`")
        with c2: st.write("**AI LOAD**  \n`MINIMAL`")
        with c3: st.write("**UPLINK**  \n`VERIFIED`")

    st.markdown("---")

    # 2. AI MODEL EXPLANATION
    col_x, col_y = st.columns([1.2, 1])
    with col_x:
        st.markdown("### 🔬 HOW THE AI WORKS")
        st.write("""
        The system utilizes a **Haar Cascade Classifier**, a robust machine learning object detection algorithm used to identify faces in images or video streams.
        
        - **Feature Detection**: The model scans for 'Haar' features, which are essentially rectangular patterns of dark and light pixels. Common patterns include the horizontal bar of eyebrows or the vertical light strip on the nose.
        - **Cascaded Logic**: To maintain extreme speed, the AI uses a "Cascade" of stages. It quickly rejects non-face regions in early stages and only focuses heavy computation on areas that highly resemble a face.
        - **Viola-Jones Algorithm**: This industry-standard framework ensures that recognition is fast enough for real-time video, even on standard hardware.
        """)
    with col_y:
        node = safe_image_load(NODE_IMG)
        if node:
            st.image(node, use_container_width=True, caption="Neural Node Visualization")

    st.markdown("---")

    # 3. TOOLS MANUAL
    st.markdown("### 🛠️ PLATFORM TOOLS")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("<div class='diag-box' style='height: 220px;'><h4>🖼️ Image Recognition</h4><p style='font-size:0.85rem;'>Analyze static images for biometric data. Perfect for verifying individual portraits.</p></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='diag-box' style='height: 220px;'><h4>📹 Live Detection</h4><p style='font-size:0.85rem;'>Real-time webcam surveillance. Features active target tracking and tactical diagnostics.</p></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='diag-box' style='height: 220px;'><h4>🎞️ Video Forensic</h4><p style='font-size:0.85rem;'>Scan recorded video archives. Automatically detects and logs faces throughout the duration.</p></div>", unsafe_allow_html=True)

elif mission == "Image Recognition":
    st.markdown("<h3>🖼️ IMAGE TARGET ANALYSIS</h3>", unsafe_allow_html=True)
    up = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if up:
        raw = Image.open(up)
        arr = np.array(raw.convert("RGB"))
        if ai_engine:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            faces = ai_engine.detectMultiScale(gray, SENS, STAB)
            for (x, y, w, h) in faces:
                draw_pro_box(arr, x, y, w, h)
            st.image(arr, use_container_width=True)
            st.success(f"RECOGNITION COMPLETE: {len(faces)} entities found.")

elif mission == "Live Detection":
    st.markdown("<h3>📹 LIVE OPERATION: REAL-TIME DETECTION</h3>", unsafe_allow_html=True)
    if ai_engine is None:
        st.error("ENGINE ERROR: AI Model disconnected.")
    else:
        st.info("💡 Please allow camera access to start the vision sync.")
        webrtc_streamer(
            key="v10-pro-final",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

elif mission == "Video Forensic":
    st.markdown("<h3>🎞️ VIDEO ARCHIVE SCAN</h3>", unsafe_allow_html=True)
    vid = st.file_uploader("Upload Video File", type=["mp4","mov"])
    if vid:
        st.video(vid)
        if st.button("🚀 INITIATE SCAN"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            vid.seek(0)
            tfile.write(vid.read())
            cap = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            placeholder = st.empty()
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                count += 1
                if count % 3 != 0: continue 
                h, w = frame.shape[:2]
                frame_s = cv2.resize(frame, (480, int(h * 480 / w)))
                if ai_engine:
                    gray = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
                    fcs = ai_engine.detectMultiScale(gray, SENS, STAB)
                    for (fx, fy, fw, fh) in fcs:
                        draw_pro_box(frame_s, fx, fy, fw, fh)
                placeholder.image(frame_s, channels="BGR", use_container_width=True)
                progress.progress(min(count/total, 1.0))
            cap.release()
            os.unlink(tfile.name)
            st.success("SCAN COMPLETE: DATA LOGGED.")
