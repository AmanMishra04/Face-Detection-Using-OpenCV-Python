import cv2
import os
import numpy as np

# Configuration
FACE_XML = "haarcascade_frontalface_default.xml"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_LIST = ['Male', 'Female']
MEAN_VALUES = (104.0, 117.0, 123.0)

# Test Subjects
SUBJECT_MALE = r"C:\Users\AMAN MISHRA\.gemini\antigravity\brain\53bddd5d-b4d7-4742-819e-ad31a2ad2482\man_test_subject_1774291815128.png"
SUBJECT_FEMALE = r"C:\Users\AMAN MISHRA\.gemini\antigravity\brain\53bddd5d-b4d7-4742-819e-ad31a2ad2482\woman_test_subject_1774291833966.png"
OUT_MALE = r"C:\Users\AMAN MISHRA\.gemini\antigravity\brain\53bddd5d-b4d7-4742-819e-ad31a2ad2482\biometric_verify_male.png"
OUT_FEMALE = r"C:\Users\AMAN MISHRA\.gemini\antigravity\brain\53bddd5d-b4d7-4742-819e-ad31a2ad2482\biometric_verify_female.png"

def run_audit(img_path, out_path):
    if not os.path.exists(img_path): return False
    img = cv2.imread(img_path)
    if img is None: return False
    
    face_cascade = cv2.CascadeClassifier(FACE_XML)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Padded ROI
        padding = 20
        face_roi = img[max(0, y-padding):min(y+h+padding, img.shape[0]), 
                       max(0, x-padding):min(x+w+padding, img.shape[1])]
        
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()
        gender = GENDER_LIST[preds[0].argmax()]
        
        # Draw HUD (Cyberpunk Style)
        cv2.rectangle(img, (x, y), (x+w, y+h), (241, 102, 99), 2)
        cv2.putText(img, f"GENDER: {gender.upper()}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 242, 0), 2)
        cv2.putText(img, "MATCH_CONFIRMED", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99, 241, 102), 1)

    cv2.imwrite(out_path, img)
    return True

# Execute
print("Starting Biometric Audit...")
if run_audit(SUBJECT_MALE, OUT_MALE): print(f"Male Verification Exported: {OUT_MALE}")
if run_audit(SUBJECT_FEMALE, OUT_FEMALE): print(f"Female Verification Exported: {OUT_FEMALE}")
print("Audit Complete.")
