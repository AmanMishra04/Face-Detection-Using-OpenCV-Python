import cv2
import os
import sys

def detect_faces():
    # Load the Haar Cascade classifier
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        print(f"Error: Classifier file not found at {cascade_path}")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    print("""
    \033[94m
    ╔════════════════════════════════════════════════════════════╗
    ║        ⚡ REAL-TIME FACE DETECTION SYSTEM v2.0 ⚡          ║
    ║           Digital Vision Intelligence Engine               ║
    ╚════════════════════════════════════════════════════════════╝
    \033[0m
    """)
    print("\033[92m[SYSTEM]\033[0m Initializing Computer Vision Modules...")
    print("\033[92m[SYSTEM]\033[0m Scanning for Haar Cascade Classifiers...")
    
    # Load the Haar Cascade classifier
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        print(f"\033[91m[ERROR]\033[0m Classifier file not found at {cascade_path}")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("\033[92m[SYSTEM]\033[0m Engine Ready.\n")

    print("SELECT VISION MODE:")
    print(" [1] 🖼️ Static Image Analysis")
    print(" [2] 🎞️ Video Stream Processing")
    print(" [3] 📹 Real-Time Neural Tracking (Webcam)")
    print(" [4] 🚫 Terminate System")
    
    choice = input("\n\033[96mCMD > \033[0m")

    if choice == '1':
        img_path = input("Enter the path to the input image: ").strip('"')
        if not os.path.exists(img_path):
            print("Error: Image file not found.")
            return
        
        image = cv2.imread(img_path)
        if image is None:
            print("Error: Could not read image.")
            return
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        cv2.imshow('Detected Faces - Pres "q" to close', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choice == '2':
        video_path = input("Enter the path to the video file: ").strip('"')
        if not os.path.exists(video_path):
            print("Error: Video file not found.")
            return
        process_video(video_path, face_cascade)

    elif choice == '3':
        print("Starting webcam... Press 'q' to exit.")
        process_video(0, face_cascade)

    elif choice == '4':
        print("Exiting...")
        sys.exit()
    else:
        print("Invalid choice.")

def process_video(source, face_cascade):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Detection - Press 'q' to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
