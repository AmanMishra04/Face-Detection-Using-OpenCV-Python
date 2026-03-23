import cv2
import os

def test_headless():
    cascade_path = 'face_detection/haarcascade_frontalface_default.xml'
    img_path = 'face_detection/sample.jpg'
    output_path = 'face_detection/test_output.jpg'

    if not os.path.exists(cascade_path):
        print(f"Error: Cagecade file not found.")
        return
    if not os.path.exists(img_path):
        print(f"Error: Sample image not found.")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    print(f"Detected {len(faces)} faces.")
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    test_headless()
