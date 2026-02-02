import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

class CameraThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    status_signal = pyqtSignal(int)
    sign_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mp_holistic = mp.solutions.holistic
        self.face_idx = [1, 33, 133, 362, 263, 61, 291, 70, 300]
        
        self.sequence = [] 
        self.model = None
        self.actions = []
        self.last_sent = None
        self.predictions = [] 

        # --- ABSOLUTE PATH UPDATE ---
        # Using raw string to handle Windows backslashes
        base_path = r"C:\Users\Sailos R Mapfumo\Desktop\Sign language project\train"
        model_path = os.path.join(base_path, 'sign_model.h5')
        actions_path = os.path.join(base_path, 'actions.npy')

        try:
            if os.path.exists(model_path) and os.path.exists(actions_path):
                # Load with compile=False for better compatibility with custom training
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.actions = np.load(actions_path)
                print(f"✔️ AI BRAIN ONLINE: {len(self.actions)} signs loaded from {base_path}")
            else:
                print(f"❌ MISSING FILES at {base_path}")
        except Exception as e:
            print(f"⚠️ LOAD ERROR: {e}")

    def extract_keypoints(self, results):
        # 1. Pose (Shoulder-Relative) - CRITICAL: Must see your shoulders!
        if results.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark])
            center = (pose[11] + pose[12]) / 2 
            pose = (pose - center).flatten()
        else: 
            pose = np.zeros(33 * 4)

        # 2. Face (Nose-Relative)
        if results.face_landmarks:
            f_raw = np.array([[results.face_landmarks.landmark[i].x, 
                               results.face_landmarks.landmark[i].y, 
                               results.face_landmarks.landmark[i].z] for i in self.face_idx])
            face = (f_raw - f_raw[0]).flatten() 
        else: 
            face = np.zeros(len(self.face_idx) * 3)

        # 3. Hands (Wrist-Relative & Scaled)
        def norm_hand(hand):
            if not hand: return np.zeros(21 * 3)
            raw = np.array([[l.x, l.y, l.z] for l in hand.landmark])
            rel = raw - raw[0]
            scale = np.linalg.norm(raw[0] - raw[9]) + 1e-6
            return (rel / scale).flatten()

        return np.concatenate([pose, face, norm_hand(results.left_hand_landmarks), norm_hand(results.right_hand_landmarks)])

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        with self.mp_holistic.Holistic(
            model_complexity=0, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        ) as holistic:
            while self._run_flag:
                ret, frame = self.cap.read()
                if not ret: continue

                frame = cv2.flip(frame, 1)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_image)
                
                hands_detected = results.left_hand_landmarks or results.right_hand_landmarks
                
                # Progress bar updates
                score = 100 if hands_detected else (40 if results.pose_landmarks else 10)
                self.status_signal.emit(score)

                # Process features
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:] 

                # PREDICTION LOGIC
                if len(self.sequence) == 30 and self.model is not None:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                    idx = np.argmax(res)
                    confidence = res[idx]

                    # TERMINAL DEBUGGING: Shows you what the AI is thinking right now
                    if hands_detected:
                        print(f"Predicted: {self.actions[idx]} | Conf: {confidence*100:.1f}%", end='\r')

                    # GATE: Confidence & Stability
                    if confidence > 0.75 and hands_detected: # Lowered to 75% for easier detection
                        self.predictions.append(idx)
                        self.predictions = self.predictions[-3:] 
                        
                        if len(self.predictions) == 3 and np.unique(self.predictions).size == 1:
                            predicted_word = self.actions[idx]
                            if self.last_sent != predicted_word:
                                self.sign_signal.emit(predicted_word)
                                self.last_sent = predicted_word

                if not hands_detected:
                    self.last_sent = None
                    self.predictions = []

                # Render to GUI
                h, w, ch = rgb_image.shape
                q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.change_pixmap.emit(q_img)
        
        if self.cap:
            self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()