#!/usr/bin/env python3
"""
Improved Face Recognition for Robot Pet
- DNN face detector (more accurate than Haar cascade)
- Histogram equalization for lighting normalization
- LBPH recognizer for fast on-device recognition
"""

import cv2
import numpy as np
import os
import pickle
import time

FACES_DIR = "/home/bo/robot_pet/data/known_faces"
MODEL_FILE = "/home/bo/robot_pet/data/face_model.yml"
LABELS_FILE = "/home/bo/robot_pet/data/face_labels.pkl"

# DNN model paths
DNN_MODEL = "/home/bo/robot_pet/data/models/opencv_face_detector.caffemodel"
DNN_CONFIG = "/home/bo/robot_pet/data/models/opencv_face_detector.prototxt"


class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}
        self.camera = None
        self.trained = False
        
        # Try DNN detector first (more accurate), fall back to Haar
        self.use_dnn = False
        if os.path.exists(DNN_MODEL) and os.path.exists(DNN_CONFIG):
            try:
                self.face_net = cv2.dnn.readNetFromCaffe(DNN_CONFIG, DNN_MODEL)
                self.use_dnn = True
                print("[FaceRecog] Using DNN detector (high accuracy)")
            except Exception as e:
                print(f"[FaceRecog] DNN failed: {e}, using Haar cascade")
        
        if not self.use_dnn:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("[FaceRecog] Using Haar cascade detector")

        # Load existing model if available
        if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
            try:
                self.recognizer.read(MODEL_FILE)
                with open(LABELS_FILE, 'rb') as f:
                    self.labels = pickle.load(f)
                self.trained = True
                print(f"Loaded face model with {len(self.labels)} people")
            except:
                pass

    def open_camera(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.camera.isOpened()

    def close_camera(self):
        if self.camera:
            self.camera.release()
            self.camera = None

    def capture_frame(self):
        if not self.open_camera():
            return None
        # Clear buffer
        self.camera.grab()
        ret, frame = self.camera.read()
        return frame if ret else None

    def detect_faces_dnn(self, frame):
        """Detect faces using DNN (more accurate)."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 50% confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                # Convert to (x, y, w, h) format
                x, y = max(0, x1), max(0, y1)
                w_box, h_box = min(w, x2) - x, min(h, y2) - y
                if w_box > 30 and h_box > 30:
                    faces.append((x, y, w_box, h_box))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return faces, gray

    def detect_faces_haar(self, frame):
        """Detect faces using Haar cascade (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
        return list(faces), gray

    def detect_faces(self, frame):
        """Detect faces in frame, return list of (x, y, w, h)."""
        if self.use_dnn:
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)

    def preprocess_face(self, gray, x, y, w, h):
        """Extract and preprocess face region."""
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        # Histogram equalization for lighting normalization
        face_roi = cv2.equalizeHist(face_roi)
        return face_roi

    def enroll_face(self, name, num_samples=20):
        """Capture face samples with guidance for varied angles."""
        os.makedirs(FACES_DIR, exist_ok=True)
        person_dir = os.path.join(FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        print(f"Enrolling {name}...")
        print(f"Capturing {num_samples} samples. Follow the prompts!")
        print()

        samples = []
        sample_count = 0
        
        # Prompts for varied angles
        prompts = [
            "Look straight at camera",
            "Look straight at camera", 
            "Look straight at camera",
            "Look straight at camera",
            "Tilt head slightly LEFT",
            "Tilt head slightly LEFT",
            "Tilt head slightly RIGHT",
            "Tilt head slightly RIGHT",
            "Look slightly UP",
            "Look slightly UP",
            "Look slightly DOWN",
            "Look slightly DOWN",
            "Move a bit CLOSER",
            "Move a bit CLOSER",
            "Move a bit FARTHER",
            "Move a bit FARTHER",
            "Look straight again",
            "Look straight again",
            "Smile!",
            "Neutral face",
        ]

        current_prompt = ""
        no_face_count = 0

        while sample_count < num_samples:
            frame = self.capture_frame()
            if frame is None:
                continue

            faces, gray = self.detect_faces(frame)

            if len(faces) >= 1:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi = self.preprocess_face(gray, x, y, w, h)

                # Save sample
                sample_path = os.path.join(person_dir, f"{sample_count}.jpg")
                cv2.imwrite(sample_path, face_roi)
                samples.append(face_roi)
                sample_count += 1
                
                # Show progress with prompt
                prompt = prompts[sample_count - 1] if sample_count <= len(prompts) else "Hold still"
                print(f"  ✓ {sample_count}/{num_samples} - Next: {prompt}")
                no_face_count = 0
                time.sleep(0.4)  # Give time to adjust
            else:
                no_face_count += 1
                if no_face_count % 10 == 1:  # Don't spam
                    print("  ... waiting for face (move closer or adjust lighting)")

            time.sleep(0.05)

        print(f"\n✅ Captured {len(samples)} samples for {name}")
        self._train()
        return True

    def _train(self):
        """Train recognizer on all enrolled faces."""
        faces = []
        labels = []
        label_map = {}
        current_label = 0

        if not os.path.exists(FACES_DIR):
            return False

        for person_name in os.listdir(FACES_DIR):
            person_dir = os.path.join(FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            label_map[current_label] = person_name

            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    # Apply same preprocessing
                    img = cv2.equalizeHist(img)
                    faces.append(img)
                    labels.append(current_label)

            current_label += 1

        if len(faces) == 0:
            return False

        print(f"Training on {len(faces)} samples, {len(label_map)} people...")
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.write(MODEL_FILE)

        self.labels = label_map
        with open(LABELS_FILE, 'wb') as f:
            pickle.dump(self.labels, f)

        self.trained = True
        print("Training complete!")
        return True

    def recognize(self):
        """Recognize face in current camera view."""
        if not self.trained:
            return None, 0, "not_trained"

        frame = self.capture_frame()
        if frame is None:
            return None, 0, "no_camera"

        faces, gray = self.detect_faces(frame)

        if len(faces) == 0:
            return None, 0, "no_face"

        # Use the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = self.preprocess_face(gray, x, y, w, h)

        label, confidence = self.recognizer.predict(face_roi)

        # Lower confidence = better match (it's a distance metric)
        # Threshold 120 = lenient, allows some variation
        if confidence < 120:
            name = self.labels.get(label, "unknown")
            # Convert to 0-100% scale where 120=0% and 0=100%
            certainty = max(0, min(100, (120 - confidence) / 120 * 100))
            return name, certainty, "recognized"
        else:
            return None, 0, "unknown_face"

    def who_is_there(self):
        """Simple check - returns name or status."""
        name, certainty, status = self.recognize()
        if status == "recognized":
            return name, certainty
        elif status == "unknown_face":
            return "stranger", 0
        elif status == "no_face":
            return None, 0
        else:
            return None, 0


def enroll_user():
    """Interactive enrollment."""
    recognizer = FaceRecognizer()
    name = input("Enter your name: ").strip()
    if name:
        recognizer.enroll_face(name)
    recognizer.close_camera()


def test_recognition():
    """Test recognition."""
    recognizer = FaceRecognizer()

    print("Testing recognition... (Ctrl+C to stop)")
    try:
        while True:
            name, certainty = recognizer.who_is_there()
            if name:
                print(f"  Detected: {name} ({certainty:.0f}% sure)")
            else:
                print("  No one detected")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    recognizer.close_camera()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "enroll":
        enroll_user()
    else:
        test_recognition()
