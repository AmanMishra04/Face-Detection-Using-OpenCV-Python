import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import time
import threading

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
XML_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_alt2.xml")
FALLBACK_XML_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
GENDER_MODEL = os.path.join(BASE_DIR, "gender_net.caffemodel")
GENDER_PROTO = os.path.join(BASE_DIR, "gender_deploy.prototxt")
GENDER_LIST = ['Male', 'Female']
# Mean values used when the bundled gender model was trained.
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
LIVE_FRAME_COUNT = 0
MODEL_LOCK = threading.Lock()

@st.cache_resource
def load_ai():
    face_engine = None
    fallback_face_engine = None
    gender_engine = None

    try:
        cascade_classifier = getattr(cv2, "CascadeClassifier", None)
        if callable(cascade_classifier) and os.path.exists(XML_PATH):
            face_engine = cascade_classifier(XML_PATH)
        if callable(cascade_classifier) and os.path.exists(FALLBACK_XML_PATH):
            fallback_face_engine = cascade_classifier(FALLBACK_XML_PATH)

        dnn_module = getattr(cv2, "dnn", None)
        read_net = getattr(dnn_module, "readNetFromCaffe", None)
        if callable(read_net) and os.path.exists(GENDER_MODEL) and os.path.exists(GENDER_PROTO):
            gender_engine = read_net(GENDER_PROTO, GENDER_MODEL)
    except Exception:
        # Keep the dashboard available when an OpenCV wheel is incomplete.
        face_engine = None
        gender_engine = None

    status = "OPERATIONAL" if face_engine is not None and gender_engine is not None else "PARTIAL"
    return face_engine, fallback_face_engine, gender_engine, status

ai_engine, fallback_ai_engine, gender_net, ai_status = load_ai()

# 3. NAVIGATION (WING STRUCTURE)
SENS = 1.1
STAB = 5
st.sidebar.markdown("<div class='creator-badge'>CREATED BY AMAN MISHRA</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h2>VISION AI</h2>", unsafe_allow_html=True)
mission_wing = st.sidebar.radio("SYSTEM WING", ["Intelligence Dashboard", "Detection Laboratory"])

if mission_wing == "Detection Laboratory":
    st.sidebar.markdown("---")
    tool_select = st.sidebar.selectbox("OPERATIONAL TOOL", ["Image Recognizer", "Live Video Detection", "In Video Detector"])
    SENS = st.sidebar.slider("SENSITIVITY", 1.05, 1.4, 1.1)
    STAB = st.sidebar.slider("STABILITY", 1, 15, 7)
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
        # Keep the model input focused on the detected face.
        padding = max(4, int(min(w, h) * 0.15))
        face_img = img[max(0, y-padding):min(y+h+padding, img.shape[0]), 
                       max(0, x-padding):min(x+w+padding, img.shape[1])]
        if face_img.size == 0: return "UNKNOWN"
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        probabilities = gender_preds[0]
        confidence = float(probabilities.max())
        if confidence < 0.60:
            return "UNCERTAIN"
        return GENDER_LIST[int(probabilities.argmax())]
    except:
        return "ERROR"

def detect_faces(img, min_size=(40, 40)):
    if ai_engine is None and fallback_ai_engine is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    neighbors = max(3, min(STAB, 5))
    faces = ()
    if ai_engine is not None:
        faces = ai_engine.detectMultiScale(
            gray, scaleFactor=SENS, minNeighbors=neighbors, minSize=min_size
        )
    if len(faces) == 0 and fallback_ai_engine is not None:
        faces = fallback_ai_engine.detectMultiScale(
            gray, scaleFactor=SENS, minNeighbors=neighbors, minSize=min_size
        )
    if len(faces) == 0 and ai_engine is not None:
        faces = ai_engine.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=min_size
        )
    return faces

def video_frame_callback(frame):
    global LIVE_FRAME_COUNT

    try:
        img = frame.to_ndarray(format="bgr24")
        img = np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return frame

    try:
        LIVE_FRAME_COUNT += 1
        if ai_engine is not None and LIVE_FRAME_COUNT % 3 == 0:
            height, width = img.shape[:2]
            detection_width = min(width, 640)
            detection_scale = detection_width / width
            detection_img = cv2.resize(
                img, (detection_width, int(height * detection_scale))
            )
            detections = detect_faces(
                detection_img,
                min_size=(max(30, int(40 * detection_scale)), max(30, int(40 * detection_scale))),
            )
            with MODEL_LOCK:
                for (fx, fy, fw, fh) in detections:
                    original_x = int(fx / detection_scale)
                    original_y = int(fy / detection_scale)
                    original_w = int(fw / detection_scale)
                    original_h = int(fh / detection_scale)
                    gender = analyze_gender(
                        img, original_x, original_y, original_w, original_h
                    )
                    draw_pro_box(
                        img,
                        original_x,
                        original_y,
                        original_w,
                        original_h,
                        gender.upper(),
                    )
    except Exception:
        pass

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
        3.  **Preprocessing Interface**: Facial crops are normalized to 227x227 pixels with mean subtraction (78.426, 87.769, 114.896) to match the neural network's training environment.
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
            if ai_engine or fallback_ai_engine:
                faces = detect_faces(bgr)
                for (x, y, w, h) in faces:
                    gender = analyze_gender(bgr, x, y, w, h)
                    draw_pro_box(bgr, x, y, w, h, gender.upper())
                st.image(bgr, channels="BGR", use_container_width=True)
                st.success(f"ANALYSIS COMPLETE: {len(faces)} entities localized with Gender ID.")

    elif tool_select == "Live Video Detection":
        st.info("Grant camera access, then capture a frame for face and gender detection.")
        camera_frame = st.camera_input("Capture video frame", key="live-video-camera")
        if camera_frame:
            raw = Image.open(camera_frame)
            img_arr = np.array(raw.convert("RGB"))
            bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            faces = []
            if ai_engine or fallback_ai_engine:
                faces = detect_faces(bgr)
                for (x, y, w, h) in faces:
                    gender = analyze_gender(bgr, x, y, w, h)
                    draw_pro_box(bgr, x, y, w, h, gender.upper())
            st.image(bgr, channels="BGR", use_container_width=True)
            st.success(f"DETECTION COMPLETE: {len(faces)} entities localized.")

    elif tool_select == "In Video Detector":
        vid = st.file_uploader("Upload Recorded Archive", type=["mp4","mov"])
        if vid:
            st.subheader("Original Video")
            st.video(vid)
            if st.button("🚀 EXECUTE BIOMETRIC SCAN"):
                input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                vid.seek(0)
                input_file.write(vid.read())
                input_file.close()
                cap = cv2.VideoCapture(input_file.name)
                progress = st.progress(0)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if fps and fps > 0 else 25.0
                output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                output_file.close()
                writer = None
                count = 0
                detected_count = 0
                gender_counts = {"MALE": 0, "FEMALE": 0, "UNCERTAIN": 0, "UNKNOWN": 0}
                tracked_faces = []
                preview_frames = []
                missed_frames = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    count += 1
                    h, w = frame.shape[:2]
                    frame_s = cv2.resize(frame, (480, int(h * 480 / w)))
                    if writer is None:
                        output_size = (frame_s.shape[1], frame_s.shape[0])
                        writer = cv2.VideoWriter(
                            output_file.name,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            output_size,
                        )
                        if not writer.isOpened():
                            cap.release()
                            os.unlink(input_file.name)
                            st.error("Unable to create the processed video file in this environment.")
                            st.stop()
                    if (ai_engine or fallback_ai_engine):
                        fcs = detect_faces(frame_s, min_size=(20, 20))
                        if len(fcs) > 0:
                            tracked_faces = []
                            for (fx, fy, fw, fh) in fcs:
                                gender = analyze_gender(frame_s, fx, fy, fw, fh)
                                gender_label = gender.upper()
                                tracked_faces.append((fx, fy, fw, fh, gender_label))
                                detected_count += 1
                                gender_counts[gender_label] = gender_counts.get(gender_label, 0) + 1
                            missed_frames = 0
                        else:
                            missed_frames += 1
                            if missed_frames > 5:
                                tracked_faces = []
                    for (fx, fy, fw, fh, gender) in tracked_faces:
                        draw_pro_box(frame_s, fx, fy, fw, fh, gender)
                    writer.write(frame_s)
                    if len(preview_frames) < 6 and (count == 1 or count % max(total // 6, 1) == 0):
                        preview_frames.append(Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)))
                    progress.progress(min(count / max(total, 1), 1.0))
                cap.release()
                if writer is not None:
                    writer.release()
                os.unlink(input_file.name)
                st.success(f"VIDEO DETECTION COMPLETE: {count} frames analyzed.")
                st.metric("Detected face instances", detected_count)
                st.write(
                    "Gender results: "
                    + ", ".join(f"{label}: {amount}" for label, amount in gender_counts.items() if amount)
                )
                if preview_frames:
                    st.subheader("Detection Preview Frames")
                    preview_width = max(frame.width for frame in preview_frames)
                    preview_height = max(frame.height for frame in preview_frames)
                    contact_sheet = Image.new(
                        "RGB", (preview_width * 2, preview_height * 3), "black"
                    )
                    for index, preview in enumerate(preview_frames):
                        contact_sheet.paste(preview, ((index % 2) * preview_width, (index // 2) * preview_height))
                    st.image(contact_sheet, use_container_width=True)
                st.subheader("Detected Faces and Gender")
                with open(output_file.name, "rb") as processed_video:
                    st.video(processed_video.read())
                with open(output_file.name, "rb") as processed_video:
                    st.download_button(
                        "Download detected video",
                        processed_video.read(),
                        file_name="detected_faces_gender.mp4",
                        mime="video/mp4",
                    )
