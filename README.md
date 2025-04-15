# Face Recognition Web App

A Flask-based face recognition web application that supports face detection, registration, and recognition using multiple detection models (Haar Cascade, MTCNN, RetinaFace) and embedding models (DINOv2, FaceNet). The app supports both image upload and live webcam input.

---

## Features

- Face Detection using:
  - Haar Cascade (default)
  - MTCNN (`facenet-pytorch`)
  - RetinaFace (ONNX-based)
- Embedding Extraction using:
  - DINOv2 (default)
  - FaceNet (`facenet-pytorch`)
- Image upload and webcam capture support
- Face registration with persistent storage
- Face recognition with configurable similarity threshold
- REST API built using Flask
- Embeddings stored in `face_database.json` for persistence

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/SharvanCN1/Face_detection.git
cd Face_detection 
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Flask app
```bash
python app.py
```

The server will start at `http://0.0.0.0:5000/`.

### 2. Available Endpoints

- `GET /` – Main index (requires `index.html` in `templates/`)
- `POST /upload` – Upload an image for face detection and recognition
- `POST /webcam` – Send a base64 webcam frame for processing
- `POST /register` – Register a face with a name
- `GET /faces` – List all registered faces
- `GET /capture_webcam` – Capture and process a frame from the webcam

---

## Notes

- DINOv2 and RetinaFace models will be downloaded at runtime if not available.
- For MTCNN and FaceNet support, install `facenet-pytorch` manually.
- Detected and processed images are stored under `static/uploads/`.

---

## License

This project is licensed under the MIT License.
