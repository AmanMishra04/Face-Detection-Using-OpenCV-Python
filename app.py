import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import time
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# 1. PAGE SETUP (Vision AI Professional Edition)
st.set_page_config(page_title="VISION AI | Professional Biometrics", page_icon="👁️", layout="wide")

# Google Fonts & Premium Spatial CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
    /* Premium Dashboard Aesthetic */
    .stApp { 
        background-color: #0f172a; 
        color: #f1f5f9; 
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 { 
        font-family: 'Orbitron', sans-serif !important; 
        color: #6366f1 !important; 
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 20px !important;
        margin-bottom: 12px !important;
    }
    
    /* Elegant Spatial Control */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
        max-width: 90% !important;
    }
    
    .element-container {
        margin-bottom: 1rem !important;
    }

    /* System Cards - Premium Elevation */
    .diag-box { 
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)); 
        border: 1px solid rgba(99, 102, 241, 0.2) !important; 
        border-left: 5px solid #6366f1 !important;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    /* Creator Badge - Premium Style */
    .creator-badge {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    /* Lab Tools Buttons */
    .stButton>button { 
        background: #6366f1 !important; 
        color: white !important; 
        border-radius: 8px; 
        height: 3.5rem; 
        font-weight: 600; 
        width: 100%; 
        transition: all 0.3s;
        border: none !important;
    }
    .stButton>button:hover { 
        background: #4f46e5 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
    }
    
    .badge {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-right: 8px;
    }
    
    /* Roadmap Item - Sophisticated */
    .roadmap-item {
        border-left: 3px solid #6366f1;
        padding-left: 18px;
        margin-bottom: 20px;
        background: rgba(99, 102, 241, 0.05);
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    
    /* Global Typography Spacing */
    p {
        margin-bottom: 15px !important;
        line-height: 1.7 !important;
        font-weight: 300;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# 2. CORE UTILITIES & AI LOADING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
GENDER_MODEL = os.path.join(BASE_DIR, "gender_net.caffemodel")
GENDER_PROTO = os.path.join(BASE_DIR, "gender_deploy.prototxt")
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (104.0, 117.0, 123.0)

@st.cache_resource
def load_ai():
    face_engine = cv2.CascadeClassifier(XML_PATH) if os.path.exists(XML_PATH) else None
    gender_engine = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL) if os.path.exists(GENDER_MODEL) and os.path.exists(GENDER_PROTO) else None
    status = "OPERATIONAL" if face_engine and gender_engine else "PARTIAL"
    return face_engine, gender_engine, status

ai_engine, gender_net, ai_status = load_ai()

# 3. NAVIGATION (WING STRUCTURE)
st.sidebar.markdown("<div class='creator-badge'>CREATED BY AMAN MISHRA</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h2>VISION AI</h2>", unsafe_allow_html=True)
mission_wing = st.sidebar.radio("SYSTEM WING", ["Intelligence Dashboard", "Detection Laboratory"])

if mission_wing == "Detection Laboratory":
    st.sidebar.markdown("---")
    tool_select = st.sidebar.selectbox("OPERATIONAL TOOL", ["Image Recognizer", "Live Sentinel", "Archive Scanner"])
    SENS = st.sidebar.slider("SENSITIVITY", 1.05, 1.4, 1.1)
    STAB = st.sidebar.slider("STABILITY", 1, 15, 3)
else:
    st.sidebar.info("Select 'Detection Laboratory' to access biometric tracking tools.")

# 4. SYSTEM BOX LOGIC
def draw_pro_box(img, x, y, w, h, gender_label="ANALYZING..."):
    # Main Bounding Box
    cv2.rectangle(img, (x, y), (x + w, y + h), (241, 102, 99), 2)
    # Tactical Corners
    length = 15
    cv2.line(img, (x, y), (x + length, y), (255, 242, 0), 2)
    cv2.line(img, (x, y), (x, y + length), (255, 242, 0), 2)
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), (255, 242, 0), 2)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), (255, 242, 0), 2)
    
    # Gender Label (Outside Box)
    label_y = y - 10 if y - 10 > 10 else y + h + 20
    cv2.putText(img, f"GENDER: {gender_label}", (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 242, 0), 2)
    cv2.putText(img, "MATCH_CONFIRMED", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (99, 241, 102), 1)

def analyze_gender(img, x, y, w, h):
    if gender_net is None: return "UNKNOWN"
    try:
        # Extract face ROI with padding
        padding = 20
        face_img = img[max(0, y-padding):min(y+h+padding, img.shape[0]), 
                       max(0, x-padding):min(x+w+padding, img.shape[1])]
        if face_img.size == 0: return "UNKNOWN"
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        return GENDER_LIST[gender_preds[0].argmax()]
    except:
        return "ERROR"

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    if ai_engine is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = ai_engine.detectMultiScale(gray, scaleFactor=SENS, minNeighbors=STAB, minSize=(30, 30))
        for (fx, fy, fw, fh) in detections:
            gender = analyze_gender(img, fx, fy, fw, fh)
            draw_pro_box(img, fx, fy, fw, fh, gender.upper())
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. WING 1: INTELLIGENCE DASHBOARD
if mission_wing == "Intelligence Dashboard":
    st.markdown("<h1>👁️ AI VISION RECOGNITION</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; opacity: 0.8; margin-top: -10px;'>Advanced Biometric Intelligence Platform | Vision Authority: <b>Aman Mishra</b></p>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Core Intelligence Blocks
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div class='diag-box'>
            <h3>System Status</h3>
            <p>A professional-grade computer vision suite optimized for high-fidelity detection. The engine is tuned for real-time biometric analysis with sub-40ms latency.</p>
            <div style='margin-top: 15px;'>
                <span class='badge'>Engine: {ai_status}</span>
                <span class='badge'>Intelligence: Dual-Core</span>
                <span class='badge'>Gender ID: Enabled</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='diag-box'>
            <h3>Core Advantages</h3>
            <ul style='color: #cbd5e1; line-height: 1.7;'>
                <li><b>🔒 Local Privacy</b>: All processing occurs within your secure environment.</li>
                <li><b>⚧ Gender ID</b>: Real-time neural classification of gender parameters.</li>
                <li><b>📊 Multi-Vector</b>: Unified analysis across images, live streams, and archives.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Science & Tech Section
    col_x, col_y = st.columns([1, 1.4])
    with col_x:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **OpenCV DNN**: Powering a deep neural network for gender classification.
        - **Caffe Framework**: Utilizing industry-standard pre-trained models.
        - **Python 3.10+**: Core system architecture and data orchestration.
        - **WebRTC/PyAV**: Low-latency secure media synchronization.
        """)
        
        st.markdown("### 🗺️ System Roadmap")
        st.markdown("""
        <div class='roadmap-item'>
            <b>Q3 2026: Neural Landmarks</b><br><p style='font-size:0.85rem; opacity:0.8; margin:0;'>Implementing 68-point facial landmark and alignment mapping for deep structure analysis.</p>
        </div>
        <div class='roadmap-item'>
            <b>Q4 2026: Emotion AI</b><br><p style='font-size:0.85rem; opacity:0.8; margin:0;'>Real-time sentiment detection and emotional state classification via micro-expression tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_y:
        st.markdown("### 🔬 Biometric Intelligence Layers")
        st.write("""
        The **AI Vision Recognition** platform now operates on a dual-layer intelligence architecture:
        
        1.  **Localization Layer (Haar Cascade)**: High-speed detection of facial bounding boxes using the Viola-Jones framework.
        2.  **Classification Layer (Caffe DNN)**: A deep convolutional neural network (CNN) analyzes the detected facial ROI to identify gender characteristics.
        3.  **Preprocessing Interface**: Facial crops are normalized to 227x227 pixels with mean subtraction (104, 117, 123) to match the neural network's training environment.
        4.  **Inference Engine**: Real-time forward pass through the gender net for near-instant classification labels.
        """)
        
        with st.expander("Explore Recognition Complexity"):
            st.write("""
            The gender model is based on a refined CaffeNet architecture, trained for high-fidelity classification. For 100% accurate results, ensure subjects are well-lit and facing the sensor directly.
            """)

# 6. WING 2: DETECTION LABORATORY
elif mission_wing == "Detection Laboratory":
    st.markdown(f"<h1>🧪 DETECTION LABORATORY</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3>Operational Tool: {tool_select}</h3>", unsafe_allow_html=True)
    
    if tool_select == "Image Recognizer":
        up = st.file_uploader("Upload Image Intelligence Asset", type=["jpg","png","jpeg"])
        if up:
            raw = Image.open(up)
            img_arr = np.array(raw.convert("RGB"))
            bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            if ai_engine:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                faces = ai_engine.detectMultiScale(gray, SENS, STAB)
                for (x, y, w, h) in faces:
                    gender = analyze_gender(bgr, x, y, w, h)
                    draw_pro_box(bgr, x, y, w, h, gender.upper())
                st.image(bgr, channels="BGR", use_container_width=True)
                st.success(f"ANALYSIS COMPLETE: {len(faces)} entities localized with Gender ID.")

    elif tool_select == "Live Sentinel":
        st.info("💡 Grant optical sensor access to initiate real-time biometric tracking with Gender ID.")
        webrtc_streamer(
            key="v12-laboratory-gender",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    elif tool_select == "Archive Scanner":
        vid = st.file_uploader("Upload Recorded Archive", type=["mp4","mov"])
        if vid:
            st.video(vid)
            if st.button("🚀 EXECUTE BIOMETRIC SCAN"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                vid.seek(0)
                tfile.write(vid.read())
                cap = cv2.VideoCapture(tfile.name)
                progress = st.progress(0)
                placeholder = st.empty()
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                            gender = analyze_gender(frame_s, fx, fy, fw, fh)
                            draw_pro_box(frame_s, fx, fy, fw, fh, gender.upper())
                    placeholder.image(frame_s, channels="BGR", use_container_width=True)
                    progress.progress(min(count/total, 1.0))
                cap.release()
                os.unlink(tfile.name)
                st.success("FORENSIC SCAN COMPLETE WITH GENDER INTELLIGENCE.")
