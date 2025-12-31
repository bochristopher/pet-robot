#!/usr/bin/env python3
"""
Simple Face Recognition for Robot Pet
Uses OpenCV's LBPH recognizer - works on Jetson without dlib
"""

import cv2
import numpy as np
import os
import pickle
import time

FACES_DIR = "/home/bo/robot_pet/data/known_faces"
MODEL_FILE = "/home/bo/robot_pet/data/face_model.yml"
LABELS_FILE = "/home/bo/robot_pet/data/face_labels.pkl"

class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}
        self.camera = None
        self.trained = False

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

    def detect_faces(self, frame):
        """Detect faces in frame, return list of (x, y, w, h)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        return faces, gray

    def enroll_face(self, name, num_samples=10):
        """Capture face samples and train recognizer."""
        os.makedirs(FACES_DIR, exist_ok=True)
        person_dir = os.path.join(FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        print(f"Enrolling {name}...")
        print("Look at the camera. Capturing 10 samples...")

        samples = []
        sample_count = 0

        while sample_count < num_samples:
            frame = self.capture_frame()
            if frame is None:
                continue

            faces, gray = self.detect_faces(frame)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))

                # Save sample
                sample_path = os.path.join(person_dir, f"{sample_count}.jpg")
                cv2.imwrite(sample_path, face_roi)
                samples.append(face_roi)
                sample_count += 1
                print(f"  Captured {sample_count}/{num_samples}")
                time.sleep(0.3)
            elif len(faces) == 0:
                print("  No face detected, move closer...")
            else:
                print("  Multiple faces, need just one...")

            time.sleep(0.1)

        print(f"Captured {len(samples)} samples for {name}")
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
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))

        label, confidence = self.recognizer.predict(face_roi)

        # Lower confidence = better match (it's a distance metric)
        if confidence < 70:
            name = self.labels.get(label, "unknown")
            certainty = max(0, 100 - confidence)  # Convert to percentage
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
