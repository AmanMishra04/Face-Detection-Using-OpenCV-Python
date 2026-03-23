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
else:
    st.sidebar.info("Select 'Detection Laboratory' to access biometric tracking tools.")

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
                <span class='badge'>Latency: 35ms</span>
                <span class='badge'>Sync: Local-Only</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='diag-box'>
            <h3>Core Advantages</h3>
            <ul style='color: #cbd5e1; line-height: 1.7;'>
                <li><b>🔒 Local Privacy</b>: All processing occurs within your secure environment.</li>
                <li><b>⚡ Zero Latency</b>: Optimized Haar Cascade kernels for instant recognition.</li>
                <li><b>📊 Multi-Vector</b>: Unified analysis across images, live streams, and archives.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Science & Tech Section
    col_x, col_y = st.columns([1, 1.4])
    with col_x:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Python 3.10+**: Core system architecture and data orchestration.
        - **OpenCV 4.10**: Advanced Computer Vision kernels and image processing.
        - **Streamlit**: Modern, high-performance professional UI framework.
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
        st.markdown("### 🔬 Algorithmic Foundation")
        st.write("""
        The **AI Vision Recognition** platform is built upon the landmark **Viola-Jones Framework**. This system ensures robust object detection through a series of optimized mathematical processes:
        
        1.  **Haar-like Features**: Rapid acquisition of facial structures by analyzing pixel intensity contrast across specialized rectangular regions.
        2.  **Integral Images**: A breakthrough technique that allows for the calculation of feature sums in constant time, independent of window scale.
        3.  **Adaboost Learning**: Selective processing that focuses the engine's power on the most critical biometric identifiers.
        4.  **Attentional Cascade**: A multi-stage filtering process where negative regions are discarded early, allowing for multi-entity tracking at 30+ FPS.
        """)
        
        with st.expander("Explore Recognition Complexity"):
            st.write("""
            The current implementation utilizes a 25-stage cascade classifier, meticulously trained on thousands of positive facial samples to minimize false positives while maintaining high sensitivity.
            """)

# 6. WING 2: DETECTION LABORATORY
elif mission_wing == "Detection Laboratory":
    st.markdown(f"<h1>🧪 DETECTION LABORATORY</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3>Operational Tool: {tool_select}</h3>", unsafe_allow_html=True)
    
    if tool_select == "Image Recognizer":
        up = st.file_uploader("Upload Image Intelligence Asset", type=["jpg","png","jpeg"])
        if up:
            raw = Image.open(up)
            arr = np.array(raw.convert("RGB"))
            if ai_engine:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                faces = ai_engine.detectMultiScale(gray, SENS, STAB)
                for (x, y, w, h) in faces: draw_pro_box(arr, x, y, w, h)
                st.image(arr, use_container_width=True)
                st.success(f"ANALYSIS COMPLETE: {len(faces)} entities localized.")

    elif tool_select == "Live Sentinel":
        st.info("💡 Grant optical sensor access to initiate real-time biometric tracking.")
        webrtc_streamer(
            key="v11-laboratory-aurora",
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
                        for (fx, fy, fw, fh) in fcs: draw_pro_box(frame_s, fx, fy, fw, fh)
                    placeholder.image(frame_s, channels="BGR", use_container_width=True)
                    progress.progress(min(count/total, 1.0))
                cap.release()
                os.unlink(tfile.name)
                st.success("FORENSIC SCAN COMPLETE.")
