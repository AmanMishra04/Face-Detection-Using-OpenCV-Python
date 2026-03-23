from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# Load the Haar Cascade classifier
# Path relative to the deployment environment
cascade_path = os.path.join(os.path.dirname(__file__), '..', 'haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_path):
    # Try current directory as well
    cascade_path = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(cascade_path)

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode the base64 image
        header, encoded = image_data.split(",", 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Detection parameters from request or defaults
        scale_factor = float(data.get('scaleFactor', 1.1))
        min_neighbors = int(data.get('minNeighbors', 5))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

        results = []
        for (x, y, w, h) in faces:
            results.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            })

        return jsonify({'faces': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'Neural Engine Online', 'model_loaded': not face_cascade.empty()})

if __name__ == '__main__':
    app.run(debug=True)
