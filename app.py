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

# Google Fonts & Ultra-Dense Professional CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    /* Professional Dashboard Aesthetic */
    .stApp { 
        background-color: #0f172a; 
        color: #f1f5f9; 
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 { 
        font-family: 'Orbitron', sans-serif !important; 
        color: #6366f1 !important; 
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0px !important;
        margin-bottom: 5px !important;
    }
    
    /* Aggressive Gap Reduction */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    .element-container {
        margin-bottom: 0.2rem !important;
    }
    .stVerticalBlock {
        gap: 0.5rem !important;
    }
    
    /* System Cards */
    .diag-box { 
        background: rgba(30, 41, 59, 0.7); 
        border: 1px solid rgba(99, 102, 241, 0.2) !important; 
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 5px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Creator Badge */
    .creator-badge {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Lab Tools Buttons */
    .stButton>button { 
        background: #6366f1 !important; 
        color: white !important; 
        border-radius: 6px; 
        height: 3rem; 
        font-weight: 600; 
        width: 100%; 
    }
    
    .badge {
        background: rgba(99, 102, 241, 0.1);
        color: #818cf8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Roadmap Item */
    .roadmap-item {
        border-left: 2px solid #6366f1;
        padding-left: 10px;
        margin-bottom: 8px;
        background: rgba(99, 102, 241, 0.05);
        padding-top: 3px;
        padding-bottom: 3px;
    }
</style>
""", unsafe_allow_html=True)

# 2. CORE UTILITIES & AI LOADING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
HERO_IMG = "vision_hero.png"
NODE_IMG = "biometric_node.png"

def safe_image_load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path): return None
    try:
        return Image.open(path)
    except: return None

@st.cache_resource
def load_ai():
    if not os.path.exists(XML_PATH): return None, "OFFLINE"
    return cv2.CascadeClassifier(XML_PATH), "OPERATIONAL"

ai_engine, ai_status = load_ai()

# 3. NAVIGATION (WING STRUCTURE)
st.sidebar.markdown("<div class='creator-badge'>CREATED BY AMAN MISHRA</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h2>VISION AI</h2>", unsafe_allow_html=True)
mission_wing = st.sidebar.radio("SYSTEM WING", ["Intelligence Dashboard", "Detection Laboratory"])

if mission_wing == "Detection Laboratory":
    st.sidebar.markdown("---")
    tool_select = st.sidebar.selectbox("OPERATIONAL TOOL", ["Image Recognizer", "Live Sentinel", "Archive Scanner"])
    SENS = st.sidebar.slider("SENSITIVITY", 1.05, 1.4, 1.1)
    STAB = st.sidebar.slider("STABILITY", 1, 15, 3)

# 4. SYSTEM BOX LOGIC
def draw_pro_box(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (241, 102, 99), 2)
    length = 15
    cv2.line(img, (x, y), (x + length, y), (255, 242, 0), 2)
    cv2.line(img, (x, y), (x, y + length), (255, 242, 0), 2)
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), (255, 242, 0), 2)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), (255, 242, 0), 2)
    cv2.putText(img, "IDENTITY_SYNC", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 242, 0), 1)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    if ai_engine is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = ai_engine.detectMultiScale(gray, scaleFactor=SENS, minNeighbors=STAB, minSize=(30, 30))
        for (fx, fy, fw, fh) in detections: draw_pro_box(img, fx, fy, fw, fh)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. WING 1: INTELLIGENCE DASHBOARD
if mission_wing == "Intelligence Dashboard":
    st.markdown("<h1>👁️ AI VISION RECOGNITION</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1rem; opacity: 0.8; margin-top: -10px;'>Advanced Biometric Detection & Analysis Platform</p>", unsafe_allow_html=True)
    
    # Hero Section
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        hero = safe_image_load(HERO_IMG)
        if hero: st.image(hero, use_container_width=True)
        else: st.markdown("<div class='diag-box'><h3>VISION AI</h3><p>System Ready.</p></div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class='diag-box'>
            <h3>System Status</h3>
            <p style='font-size: 0.9rem;'>A professional-grade computer vision suite optimized for high-fidelity detection.</p>
            <div style='margin-bottom: 10px;'>
                <span class='badge'>Engine: {ai_status}</span>
                <span class='badge'>Latency: 35ms</span>
                <span class='badge'>Protocol: Secure-Local</span>
            </div>
            <p style='font-size: 0.85rem; color: #94a3b8;'>Vision Authority: <b>Aman Mishra</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🏆 Core Advantages")
        st.markdown("""
        - **🔒 Local Privacy**: All processing occurs locally.
        - **⚡ Efficiency**: Real-time performance on standard hardware.
        - **📊 Multi-Stream**: Supports static, live, and archive data.
        """)

    # Science & Tech Section
    col_x, col_y = st.columns([1, 1.2])
    with col_x:
        node = safe_image_load(NODE_IMG)
        if node: st.image(node, use_container_width=True)
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Python 3.10+**: Core logic.
        - **OpenCV 4.10**: Vision kernels.
        - **Streamlit**: UI High-performance components.
        - **WebRTC/PyAV**: Low-latency streaming.
        """)
    with col_y:
        st.markdown("### 🔬 Algorithmic Foundation")
        with st.expander("The Science of Haar Cascades", expanded=True):
            st.write("""
            Built upon the **Viola-Jones Framework**:
            
            1.  **Haar Features**: Edge and shadow identification.
            2.  **Integral Images**: Constant-time pixel calculation.
            3.  **Adaboost Learning**: Critical feature selection.
            4.  **Attentional Cascade**: Instant non-face region rejection.
            """)
        
        st.markdown("### 🗺️ Future Roadmap")
        st.markdown("""
        <div class='roadmap-item'>
            <b>Q3 2026: Neural Landmarks</b><br><p style='font-size:0.75rem; opacity:0.8;'>68-point facial landmark mapping for deep structure alignment.</p>
        </div>
        <div class='roadmap-item'>
            <b>Q4 2026: Emotion AI</b><br><p style='font-size:0.75rem; opacity:0.8;'>Sentiment detection through micro-expression tracking.</p>
        </div>
        <div class='roadmap-item'>
            <b>2027: Neural Pose Estimation</b><br><p style='font-size:0.75rem; opacity:0.8;'>Full crowd analytics and behavior modeling.</p>
        </div>
        """, unsafe_allow_html=True)

# 6. WING 2: DETECTION LABORATORY
elif mission_wing == "Detection Laboratory":
    st.markdown(f"<h1>🧪 DETECTION LABORATORY</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3>Tool: {tool_select}</h3>", unsafe_allow_html=True)
    
    if tool_select == "Image Recognizer":
        up = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if up:
            raw = Image.open(up)
            arr = np.array(raw.convert("RGB"))
            if ai_engine:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                faces = ai_engine.detectMultiScale(gray, SENS, STAB)
                for (x, y, w, h) in faces: draw_pro_box(arr, x, y, w, h)
                st.image(arr, use_container_width=True)
                st.success(f"ANALYSIS COMPLETE: {len(faces)} entities found.")

    elif tool_select == "Live Sentinel":
        st.info("💡 Grant optical sensor access.")
        webrtc_streamer(
            key="v11-laboratory",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    elif tool_select == "Archive Scanner":
        vid = st.file_uploader("Upload Archive", type=["mp4","mov"])
        if vid:
            st.video(vid)
            if st.button("🚀 EXECUTE SCAN"):
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
                        for (fx, fy, fw, fh) in fcs: draw_pro_box(frame_s, fx, fy, fw, fh)
                    placeholder.image(frame_s, channels="BGR", use_container_width=True)
                    progress.progress(min(count/total, 1.0))
                cap.release()
                os.unlink(tfile.name)
                st.success("SCAN COMPLETE.")
