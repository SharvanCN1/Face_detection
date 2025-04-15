import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import base64
import logging
import onnxruntime
import os
import time
from PIL import Image
import urllib.request
import gdown

logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, detection_method='haar', embedding_model='dinov2'):
        logger.info(f"Initializing FaceDetector with {detection_method} detector and {embedding_model} embeddings...")

        self.detection_method = detection_method
        self.embedding_model_name = embedding_model
        self.model_dir = os.path.join(os.path.expanduser("~"), ".face_models")
        os.makedirs(self.model_dir, exist_ok=True)
        self._init_face_detector()
        self._init_embedding_model()
        self.face_database = {}  
        logger.info("FaceDetector initialization complete")

    def _init_face_detector(self):
        try:
            if self.detection_method == 'retinaface':
                self._init_retinaface()
            elif self.detection_method == 'mtcnn':
                self._init_mtcnn()
            else:
                logger.info("Using Haar cascade classifier as fallback")
                self.detection_method = 'haar'
                self._init_haar_cascade()
        except Exception as e:
            logger.error(f"Failed to initialize {self.detection_method} detector: {e}")
            logger.info("Falling back to Haar cascade classifier")
            self.detection_method = 'haar'
            self._init_haar_cascade()

    def _init_retinaface(self):
        model_path = os.path.join(self.model_dir, "retinaface_resnet50.onnx")

        if not os.path.exists(model_path):
            logger.info("Downloading RetinaFace model...")
            try:
                url = "1https://github.com/discipleofhamilton/RetinaFace/raw/refs/heads/master/FaceDetector.onnx?download="
                urllib.request.urlretrieve(url, model_path)
                logger.info("RetinaFace model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download RetinaFace model: {e}")
                raise
    
        self.retinaface_session = onnxruntime.InferenceSession(model_path)
        self.retinaface_input_name = self.retinaface_session.get_inputs()[0].name
        logger.info("RetinaFace model loaded successfully")

    def _init_mtcnn(self):
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn_detector = MTCNN(keep_all=True, device='cpu', post_process=False)
            logger.info("MTCNN detector initialized successfully")
        except ImportError:
            logger.error("facenet_pytorch package not found. Install with: pip install facenet-pytorch")
            raise ImportError("Missing dependency: facenet_pytorch")

    def _init_haar_cascade(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.face_cascade.empty():
            logger.error("Failed to load Haar cascade classifier. Check OpenCV installation.")
            raise Exception("Failed to load face detector")
        logger.info("Haar cascade classifier loaded successfully")

    def _init_embedding_model(self):
        try:
            if self.embedding_model_name == 'dinov2':
                logger.info("Loading DINOv2 model...")
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                self.model = AutoModel.from_pretrained('facebook/dinov2-base')
                logger.info("DINOv2 model loaded successfully")
            elif self.embedding_model_name == 'facenet':
                from facenet_pytorch import InceptionResnetV1
                self.model = InceptionResnetV1(pretrained='vggface2').eval()
                logger.info("FaceNet model loaded successfully")
            else:
                logger.info("Unknown embedding model, defaulting to DINOv2")
                self.embedding_model_name = 'dinov2'
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def detect_faces(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise ValueError("Image must be a path or numpy array")

        if self.detection_method == 'retinaface':
            return img, self._detect_retinaface(img)
        elif self.detection_method == 'mtcnn':
            return img, self._detect_mtcnn(img)
        else: 
            return img, self._detect_haar(img)

    def _detect_haar(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
        return faces  

    def _detect_mtcnn(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        boxes, _ = self.mtcnn_detector.detect(img_pil)

        if boxes is None:
            return []

        faces = []
        for box in boxes:
            x1, y1, x2, y2 = [int(p) for p in box]
            faces.append((x1, y1, x2-x1, y2-y1))

        return faces

    def _detect_retinaface(self, img):
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640))
        blob = cv2.dnn.blobFromImage(
                resized, 1.0, (640, 640), 
                (104, 117, 123), swapRB=False, crop=False
                )

        outputs = self.retinaface_session.run(None, {self.retinaface_input_name: blob})
        boxes = outputs[0]
        scores = outputs[1]  
        faces = []
        if len(scores.shape) == 3: 
            face_scores = scores[0, :, 1]  
        elif len(scores.shape) == 2:  
            face_scores = scores[:, 1] 
        else:
            logger.warning(f"Unexpected scores shape: {scores.shape}")
            return []
    
        for i in range(len(face_scores)):
            score = face_scores[i]
        
            if score > 0.7: 
                if len(boxes.shape) == 3:  
                    box = boxes[0, i]  
                else:  
                    box = boxes[i]
                
                x1, y1, x2, y2 = box
            
                x1 = int(x1 * w / 640)
                y1 = int(y1 * h / 640)
                x2 = int(x2 * w / 640)
                y2 = int(y2 * h / 640)

                faces.append((x1, y1, x2-x1, y2-y1))

        return faces

    def get_face_embeddings(self, image, face_locations=None):
        if face_locations is None:
            _, face_locations = self.detect_faces(image)

        if len(face_locations) == 0:
            return [], []

        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from path")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_embeddings = []
        face_images = []

        for (x, y, w, h) in face_locations:
            padding = int(min(w, h) * 0.2)  
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            face_img = image_rgb[y1:y2, x1:x2]

            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                logger.warning(f"Face too small: {face_img.shape}")
                continue

            face_images.append(face_img)

            try:
                if self.embedding_model_name == 'dinov2':
                    embedding = self._get_dinov2_embedding(face_img)
                elif self.embedding_model_name == 'facenet':
                    embedding = self._get_facenet_embedding(face_img)
                else:
                    embedding = self._get_dinov2_embedding(face_img)

                face_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                continue

        return face_embeddings, face_images

    def _get_dinov2_embedding(self, face_img):
        inputs = self.processor(images=face_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).flatten()
        embedding = embedding / torch.norm(embedding, p=2)

        return embedding

    def _get_facenet_embedding(self, face_img):
        face_pil = Image.fromarray(face_img)
        face_pil = face_pil.resize((160, 160))
        face_tensor = torch.from_numpy(np.array(face_pil)).permute(2, 0, 1).float()
        face_tensor = face_tensor.unsqueeze(0) / 255.0  # Add batch dim and normalize

        with torch.no_grad():
            embedding = self.model(face_tensor)

        embedding = embedding / torch.norm(embedding, p=2)

        return embedding.flatten()

    def compare_faces(self, embedding1, embedding2):
        similarity = torch.dot(embedding1, embedding2)
        return similarity.item()

    def register_face(self, image, name, face_locations=None):
        try:
            if face_locations is None:
                _, face_locations = self.detect_faces(image)

            if len(face_locations) == 0:
                logger.warning(f"No faces found in the image for registration")
                return False

            if len(face_locations) > 1:
                logger.warning(f"Multiple faces found in registration image. Using first detected face.")

            embeddings, _ = self.get_face_embeddings(image, [face_locations[0]])
            if embeddings:
                self.face_database[name] = embeddings[0]
                logger.info(f"Face registered successfully for {name}")
                return True
            else:
                logger.warning(f"Failed to get face embedding for {name}")
                return False
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False

    def recognize_face(self, embedding, threshold=0.7):
        best_match = None
        best_score = -1

        for name, stored_embedding in self.face_database.items():
            similarity = self.compare_faces(embedding, stored_embedding)
            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = name

        return best_match, best_score

    def process_image(self, image, draw_faces=True, recognize=True):
        start_time = time.time()
        img, face_locations = self.detect_faces(image)
        detect_time = time.time() - start_time

        result_img = img.copy()

        if len(face_locations) == 0:
            logger.info(f"No faces detected (took {detect_time:.3f}s)")
            return result_img, []

        logger.info(f"Detected {len(face_locations)} faces (took {detect_time:.3f}s)")

        start_time = time.time()
        face_embeddings, face_images = self.get_face_embeddings(img, face_locations)
        embed_time = time.time() - start_time

        results = []
        for i, ((x, y, w, h), embedding) in enumerate(zip(face_locations, face_embeddings)):
            info = {
                    "box": (int(x), int(y), int(w), int(h)),
                    "confidence": 1.0  
                    }

            if recognize and self.face_database:
                name, score = self.recognize_face(embedding)
                info["name"] = name
                info["score"] = float(score)  

            results.append(info)

            if draw_faces:
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if recognize and "name" in info and info["name"]:
                    text = f"{info['name']} ({info['score']:.2f})"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_img, (x, y - 25), (x + text_size[0], y), (0, 255, 0), -1)
                    cv2.putText(result_img, text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        total_time = detect_time + embed_time
        logger.info(f"Process completed in {total_time:.3f}s (detection: {detect_time:.3f}s, embedding: {embed_time:.3f}s)")
        return result_img, results

    def image_to_base64(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def capture_from_webcam(self, camera_id=0, num_frames=1):
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Cannot open camera {camera_id}")
                return None

            logger.info(f"Camera {camera_id} opened successfully")

            best_frame = None
            max_faces = 0

            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Can't receive frame from camera")
                    continue

                _, faces = self.detect_faces(frame)

                if len(faces) > max_faces:
                    max_faces = len(faces)
                    best_frame = frame.copy()
                time.sleep(0.1)
            cap.release()

            return best_frame
        except Exception as e:
            logger.error(f"Error capturing from webcam: {e}")
            return None
