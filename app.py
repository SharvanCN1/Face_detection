# app.py
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os
import torch
from werkzeug.utils import secure_filename
from face_detection_module import FaceDetector
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#haar, retinaface, mtcnn
face_detector = FaceDetector(detection_method='haar', embedding_model='dinov2')

DB_PATH = 'face_database.json'

def save_face_database():
    serializable_db = {}
    for name, embedding in face_detector.face_database.items():
        serializable_db[name] = embedding.tolist()
    
    with open(DB_PATH, 'w') as f:
        json.dump(serializable_db, f)
    
    print(f"Saved {len(serializable_db)} faces to database")

def load_face_database():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, 'r') as f:
                serialized_db = json.load(f)
                
            for name, embedding_list in serialized_db.items():
                face_detector.face_database[name] = torch.tensor(embedding_list)
                
            print(f"Loaded {len(face_detector.face_database)} faces from database")
        except Exception as e:
            print(f"Error loading face database: {e}")

load_face_database()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    if file:
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img, face_locations = face_detector.detect_faces(filepath)
            result_img, results = face_detector.process_image(img)
            
            result_filename = f"result_{filename}"
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_filepath, result_img)
            
            return jsonify({
                'original': f'/static/uploads/{filename}',
                'processed': f'/static/uploads/{result_filename}',
                'faces': len(results),
                'results': results
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/webcam', methods=['POST'])
def process_webcam():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data'})
        
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'})
        
        result_img, results = face_detector.process_image(img)
        
        result_base64 = face_detector.image_to_base64(result_img)
        
        return jsonify({
            'processed': f'data:image/jpeg;base64,{result_base64}',
            'faces': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': f'Error processing webcam image: {str(e)}'})

@app.route('/register', methods=['POST'])
def register_face():
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Missing file or name'})
        
    file = request.files['file']
    name = request.form['name'].strip()
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if name == '':
        return jsonify({'error': 'Name cannot be empty'})
        
    if file:
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            success = face_detector.register_face(filepath, name)
            
            if success:
                save_face_database()
                return jsonify({'success': True, 'message': f'Face registered as {name}'})
            else:
                return jsonify({'success': False, 'message': 'No face detected in the image'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error registering face: {str(e)}'})

@app.route('/faces', methods=['GET'])
def get_registered_faces():
    return jsonify({'faces': list(face_detector.face_database.keys())})

@app.route('/capture_webcam', methods=['GET'])
def capture_webcam():
    try:
        frame = face_detector.capture_from_webcam(camera_id=0, num_frames=3)
        
        if frame is None:
            return jsonify({'error': 'Failed to capture from webcam'})

        result_img, results = face_detector.process_image(frame)

        result_base64 = face_detector.image_to_base64(result_img)
        
        return jsonify({
            'captured': f'data:image/jpeg;base64,{result_base64}',
            'faces': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': f'Error capturing from webcam: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
