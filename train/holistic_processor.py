import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import mediapipe as mp

# --- 1. THE BRAIN MANAGER (Single-File Spirit) ---
class BrainManager:
    """Handles the loading, saving, and registry of the AI's intelligence."""
    def __init__(self, filename="sign_brain"):
        self.model_path = f"{filename}.h5"
        self.label_path = f"{filename}.npy"
        
    def save(self, model, labels):
        model.save(self.model_path)
        np.save(self.label_path, np.array(labels))
        
    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.label_path):
            return load_model(self.model_path), np.load(self.label_path).tolist()
        return None, None

# --- 2. THE VISION ENGINE (285-Feature Optimized) ---
class HolisticProcessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=0, # Essential for zero-lag MP4
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.face_idx = [1, 33, 133, 362, 263, 61, 291, 70, 300] # Key expression points

    def extract_keypoints(self, results):
        # Pose (Shoulder-Relative)
        if results.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark])
            center = (pose[11] + pose[12]) / 2
            pose = (pose - center).flatten()
        else: pose = np.zeros(33 * 4)

        # Face (Nose-Relative)
        if results.face_landmarks:
            f_raw = np.array([[results.face_landmarks.landmark[i].x, 
                               results.face_landmarks.landmark[i].y, 
                               results.face_landmarks.landmark[i].z] for i in self.face_idx])
            face = (f_raw - f_raw[0]).flatten()
        else: face = np.zeros(len(self.face_idx) * 3)

        # Hands (Wrist-Relative & Scale-Normalized)
        def norm_hand(hand):
            if not hand: return np.zeros(21 * 3)
            raw = np.array([[l.x, l.y, l.z] for l in hand.landmark])
            rel = raw - raw[0]
            scale = np.linalg.norm(raw[0] - raw[9]) + 1e-6
            return (rel / scale).flatten()

        return np.concatenate([pose, face, norm_hand(results.left_hand_landmarks), norm_hand(results.right_hand_landmarks)])

# --- 3. THE INTERFACE & LOGIC ---
class MasterSignAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = HolisticProcessor()
        self.brain = BrainManager()
        self.base_path = "SignData"
        self.live_sequence = []; self.recording_buffer = []
        self.sentence = []; self.is_recording = False
        
        self.init_ui()
        self.refresh_brain()
        self.showMaximized()

    def init_ui(self):
        self.setWindowTitle("Gemini Neural Sign AI")
        self.setStyleSheet("QMainWindow { background: #050505; } QGroupBox { color: #00FFCC; border: 1px solid #333; background: #0A0A0A; }")
        
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # HUD
        hud = QHBoxLayout()
        self.sig_bar = QProgressBar(); self.sig_bar.setFixedWidth(150)
        hud.addWidget(QLabel("NEURAL ENGINE V2.0")); hud.addStretch(); hud.addWidget(QLabel("SIGNAL:")); hud.addWidget(self.sig_bar)
        layout.addLayout(hud)

        # Monitors
        mons = QHBoxLayout()
        
        # Source (Capture)
        src_box = QGroupBox("SOURCE / DATA CAPTURE"); src_v = QVBoxLayout()
        self.mon_src = QLabel(); self.mon_src.setFixedSize(480, 320); self.mon_src.setStyleSheet("background: black;")
        self.mode_sel = QComboBox(); self.mode_sel.addItems(["SIGN (Dynamic)", "GESTURE (Static)"])
        src_btns = QHBoxLayout()
        btn_load = QPushButton("📁 LOAD MP4"); btn_load.clicked.connect(self.open_mp4)
        btn_cap = QPushButton("✂ CAPTURE"); btn_cap.clicked.connect(self.trigger_capture)
        src_btns.addWidget(btn_load); src_btns.addWidget(btn_cap)
        src_v.addWidget(self.mon_src); src_v.addWidget(self.mode_sel); src_v.addLayout(src_btns); src_box.setLayout(src_v)

        # Live Monitor
        live_box = QGroupBox("LIVE NEURAL FEED"); live_v = QVBoxLayout()
        self.mon_live = QLabel(); self.mon_live.setFixedSize(480, 320); self.mon_live.setStyleSheet("background: black;")
        self.sent_lbl = QLabel("IDLE"); self.sent_lbl.setStyleSheet("font-size: 18px; color: #00FFCC; background: #111; padding: 10px;")
        live_btns = QHBoxLayout()
        btn_go = QPushButton("🚀 START"); btn_go.clicked.connect(self.start_live)
        btn_stop = QPushButton("🛑 STOP"); btn_stop.clicked.connect(self.stop_live)
        live_btns.addWidget(btn_go); live_btns.addWidget(btn_stop)
        live_v.addWidget(self.mon_live); live_v.addWidget(self.sent_lbl); live_v.addLayout(live_btns); live_box.setLayout(live_v)

        mons.addWidget(src_box); mons.addWidget(live_box)
        layout.addLayout(mons)

        # Brain Sync Button
        self.btn_sync = QPushButton("🧠 SYNC & REBUILD BRAIN (DELETE BADS)"); self.btn_sync.setFixedHeight(50)
        self.btn_sync.setStyleSheet("background: #00FFCC; color: black; font-weight: bold;")
        self.btn_sync.clicked.connect(self.sync_and_train)
        layout.addWidget(self.btn_sync)

        self.timer_f = QTimer(); self.timer_f.timeout.connect(self.proc_file)
        self.timer_l = QTimer(); self.timer_l.timeout.connect(self.proc_live)

    # --- BRAIN LOGIC ---
    def sync_and_train(self):
        """The 'Delete/Upgrade' core. Scans disk and rebuilds brain from scratch."""
        self.btn_sync.setText("⏳ CLEANING & TRAINING...")
        QApplication.processEvents()
        
        actions, sequences, labels = [], [], []
        
        # Scan filesystem (Single source of truth)
        for mode in ["Signs", "Gestures"]:
            p = os.path.join(self.base_path, mode)
            if not os.path.exists(p): continue
            for action in os.listdir(p):
                actions.append(action)
                a_path = os.path.join(p, action)
                for seq in os.listdir(a_path):
                    # Data Cleaner: Ensure 30 frames exist
                    try:
                        window = [np.load(os.path.join(a_path, seq, f"{i}.npy")) for i in range(30)]
                        # Validation: Check if hands were detected
                        hand_data = np.array(window)[:, 159:]
                        if np.count_nonzero(np.sum(hand_data, axis=1)) > 25:
                            sequences.append(window); labels.append(actions.index(action))
                    except: continue

        if not sequences: return
        
        # Build Architecture
        X = np.array(sequences); y = to_categorical(labels).astype(int)
        model = Sequential([
            Input(shape=(30, 285)),
            Dense(128, activation='relu'),
            LSTM(128, return_sequences=True, activation='tanh'),
            LSTM(64, return_sequences=False, activation='tanh'),
            Dense(64, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(X, y, epochs=100, verbose=0)
        self.brain.save(model, actions)
        self.refresh_brain()
        self.btn_sync.setText("✅ BRAIN REBUILT")

    def refresh_brain(self):
        self.ai_model, self.actions = self.brain.load()

    # --- VIDEO LOGIC ---
    def open_mp4(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open MP4", "", "*.mp4")
        if p: self.cap_f = cv2.VideoCapture(p); self.timer_f.start(20)

    def trigger_capture(self):
        name, ok = QInputDialog.getText(self, "New Sign", "Label:")
        if ok and name: self.curr_label = name.upper(); self.recording_buffer = []; self.is_recording = True

    def proc_file(self):
        ret, frame = self.cap_f.read()
        if ret:
            results = self.processor.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp = self.processor.extract_keypoints(results)
            if self.is_recording:
                self.recording_buffer.append(kp)
                if len(self.recording_buffer) == 30: self.finalize_data()
            self.draw_frame(frame, self.mon_src)

    def finalize_data(self):
        self.is_recording = False
        mode = "Gestures" if "GESTURE" in self.mode_sel.currentText() else "Signs"
        path = os.path.join(self.base_path, mode, self.curr_label)
        idx = len(os.listdir(path)) if os.path.exists(path) else 0
        save_p = os.path.join(path, str(idx)); os.makedirs(save_p, exist_ok=True)
        for i, k in enumerate(self.recording_buffer): np.save(os.path.join(save_p, f"{i}.npy"), k)
        QApplication.beep()

    def start_live(self):
        if not self.actions: return
        self.cap_l = cv2.VideoCapture(0); self.timer_l.start(20)

    def stop_live(self): self.timer_l.stop(); self.cap_l.release()

    def proc_live(self):
        ret, frame = self.cap_l.read()
        if ret:
            frame = cv2.flip(frame, 1)
            results = self.processor.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp = self.processor.extract_keypoints(results)
            self.live_sequence.append(kp); self.live_sequence = self.live_sequence[-30:]
            
            if len(self.live_sequence) == 30 and self.ai_model:
                res = self.ai_model.predict(np.expand_dims(self.live_sequence, axis=0), verbose=0)[0]
                # Confidence HUD
                for i, p in enumerate(res):
                    cv2.rectangle(frame, (0, 40+(i*35)), (int(p*150), 65+(i*35)), (0, 255, 204), -1)
                    cv2.putText(frame, f"{self.actions[i]}: {p:.2f}", (5, 58+(i*35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                if res[np.argmax(res)] > 0.90:
                    word = self.actions[np.argmax(res)]
                    if not self.sentence or word != self.sentence[-1]:
                        self.sentence.append(word); self.sent_lbl.setText(" ".join(self.sentence[-5:]))
            self.draw_frame(frame, self.mon_live)

    def draw_frame(self, frame, label):
        h, w, _ = frame.shape
        qi = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, w, h, w*3, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv); win = MasterSignAI(); win.show(); sys.exit(app.exec())