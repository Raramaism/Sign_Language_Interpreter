import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import mediapipe as mp

# --- 1. OPTIMIZED ENGINE ---
class HolisticProcessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_idx = [1, 33, 133, 362, 263, 61, 291, 70, 300]

    def extract_keypoints(self, results):
        hands_detected = results.left_hand_landmarks or results.right_hand_landmarks
        
        if results.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark])
            pose = (pose - (pose[11] + pose[12]) / 2).flatten()
        else: pose = np.zeros(132)

        if results.face_landmarks:
            f_raw = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in self.face_idx])
            face = (f_raw - f_raw[0]).flatten()
        else: face = np.zeros(27)

        def norm_hand(hand):
            if not hand: return np.zeros(63)
            raw = np.array([[l.x, l.y, l.z] for l in hand.landmark])
            return ((raw - raw[0]) / (np.linalg.norm(raw[0] - raw[9]) + 1e-6)).flatten()

        keypoints = np.concatenate([pose, face, norm_hand(results.left_hand_landmarks), norm_hand(results.right_hand_landmarks)])
        return keypoints, hands_detected

# --- 2. DICTIONARY POPUP ---
class DictionaryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Neural Dictionary")
        self.setFixedSize(400, 500)
        self.setStyleSheet("background: #050505; color: #00FFCC; border: 1px solid #333;")
        layout = QVBoxLayout(self)
        self.list_view = QListWidget()
        self.list_view.setStyleSheet("QListWidget { background: #0A0A0A; border: 1px solid #00FFCC; padding: 10px; font-size: 14px; color: #00FFCC; }")
        layout.addWidget(QLabel("COLLECTED NEURAL DATA:"))
        layout.addWidget(self.list_view)
        
        path = "SignData/Signs"
        if os.path.exists(path):
            for label in sorted(os.listdir(path)):
                samples = len(os.listdir(os.path.join(path, label)))
                self.list_view.addItem(f"▶ {label.ljust(20)} | {samples} Sequences")

# --- 3. MASTER INTERFACE ---
class CyberSignAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = HolisticProcessor()
        self.base_path = "SignData"
        self.live_seq = []; self.rec_buffer = []; self.sentence = []
        self.is_recording = False; self.is_paused = False
        self.cap_f = None; self.cap_l = None
        self.model = None; self.actions = []
        
        self.init_ui()
        self.set_dull_monitors()
        self.showMaximized()

    def init_ui(self):
        self.setWindowTitle("Sign Language AI: Master Control")
        self.setStyleSheet("""
            QMainWindow { background-color: #000; }
            QGroupBox { color: #00FFCC; font-family: 'Segoe UI Black'; border: 1px solid #222; border-radius: 8px; background: #080808; margin-top: 30px; padding-top: 20px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 15px; top: 0px; }
            QLabel#Monitor { background-color: #0F0F0F; border: 1px solid #1a1a1a; }
            QPushButton { background-color: #151515; color: #EEE; border-radius: 4px; padding: 10px; font-weight: bold; }
            QPushButton#ActionBtn { background-color: #00FFCC; color: black; font-size: 16px; border: none; }
            QPushButton#DangerBtn { background-color: #800; color: white; border: none; }
            QProgressBar { border: 1px solid #333; height: 25px; text-align: center; color: white; background: #111; border-radius: 5px; }
            QProgressBar::chunk { background-color: #00FFCC; }
        """)

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(20, 20, 20, 20)
        
        head = QHBoxLayout()
        title = QLabel("NEURAL SYNC INTERFACE")
        title.setStyleSheet("font-size: 35px; color: #00FFCC; font-family: 'Segoe UI Black';")
        self.progress = QProgressBar(); self.progress.setFixedWidth(400)
        head.addWidget(title); head.addStretch(); head.addWidget(self.progress)
        main_layout.addLayout(head)

        cols = QHBoxLayout()
        
        # SOURCE CONTROL
        s_box = QGroupBox("SOURCE CONTROL"); sv = QVBoxLayout()
        self.mon_src = QLabel(); self.mon_src.setObjectName("Monitor"); self.mon_src.setMinimumSize(540, 380)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        s_btns = QHBoxLayout()
        btn_load = QPushButton("📁 LOAD MP4"); btn_load.clicked.connect(self.load_video)
        self.btn_live_t = QPushButton("🔥 LIVE FEED"); self.btn_live_t.setCheckable(True); self.btn_live_t.clicked.connect(self.toggle_source_live)
        btn_pause = QPushButton("⏯ PAUSE"); btn_pause.clicked.connect(self.toggle_pause)
        btn_cap = QPushButton("✂ CAPTURE"); btn_cap.clicked.connect(self.trigger_capture)
        s_btns.addWidget(btn_load); s_btns.addWidget(self.btn_live_t); s_btns.addWidget(btn_pause); s_btns.addWidget(btn_cap)
        sv.addWidget(self.mon_src); sv.addWidget(self.scrub); sv.addLayout(s_btns); s_box.setLayout(sv)

        # ANALYSIS
        a_box = QGroupBox("NEURAL ANALYSIS"); av = QVBoxLayout()
        self.fig, self.ax = plt.subplots(); self.fig.patch.set_facecolor('#080808')
        self.ax.set_facecolor('#080808')
        self.ax.tick_params(axis='both', colors='#00FFCC', labelsize=9)
        for spine in self.ax.spines.values(): spine.set_color('#222')
        self.canvas = FigureCanvas(self.fig); av.addWidget(self.canvas); a_box.setLayout(av)

        # LIVE FEED & TEXT
        l_box = QGroupBox("LIVE FEED"); lv = QVBoxLayout()
        self.mon_live = QLabel(); self.mon_live.setObjectName("Monitor"); self.mon_live.setMinimumSize(540, 380)
        self.text_out = QLabel("TEXT: "); self.text_out.setWordWrap(True)
        self.text_out.setStyleSheet("background: #001010; color: #00FFCC; padding: 15px; font-size: 22px; border: 1px solid #00FFCC; min-height: 100px;")
        
        l_btns = QHBoxLayout()
        self.btn_ai = QPushButton("🚀 START AI"); self.btn_ai.setCheckable(True); self.btn_ai.clicked.connect(self.toggle_interpreter)
        btn_space = QPushButton("⌨ SPACE"); btn_space.clicked.connect(self.add_space)
        btn_back = QPushButton("🔙 BACKSPACE"); btn_back.clicked.connect(self.backspace_text)
        btn_clear = QPushButton("🧹 CLEAR ALL"); btn_clear.clicked.connect(self.clear_text)
        l_btns.addWidget(self.btn_ai); l_btns.addWidget(btn_space); l_btns.addWidget(btn_back); l_btns.addWidget(btn_clear)
        lv.addWidget(self.mon_live); lv.addWidget(self.text_out); lv.addLayout(l_btns); l_box.setLayout(lv)

        cols.addWidget(s_box, 2); cols.addWidget(a_box, 1); cols.addWidget(l_box, 2)
        main_layout.addLayout(cols)

        foot = QHBoxLayout()
        btn_dict = QPushButton("📋 DICTIONARY"); btn_dict.clicked.connect(lambda: DictionaryDialog(self).exec())
        btn_sync = QPushButton("🧠 SYNC TRAIN SYSTEM"); btn_sync.setObjectName("ActionBtn"); btn_sync.setFixedHeight(55); btn_sync.clicked.connect(self.sync_brain)
        btn_term = QPushButton("🛑 TERMINATE APP"); btn_term.setObjectName("DangerBtn"); btn_term.clicked.connect(self.close)
        foot.addWidget(btn_dict); foot.addWidget(btn_sync, 2); foot.addWidget(btn_term)
        main_layout.addLayout(foot)

        self.timer_f = QTimer(); self.timer_f.timeout.connect(self.update_source)
        self.timer_l = QTimer(); self.timer_l.timeout.connect(self.update_interpreter)
        self.ana_timer = QTimer(); self.ana_timer.timeout.connect(self.refresh_analysis)
        self.ana_timer.start(10000)

    def set_dull_monitors(self):
        dull = QPixmap(540, 380); dull.fill(QColor(15, 15, 15))
        if not self.cap_f: self.mon_src.setPixmap(dull)
        if not self.cap_l: self.mon_live.setPixmap(dull)

    def toggle_source_live(self, checked):
        if self.cap_f: self.cap_f.release()
        if checked:
            self.cap_f = cv2.VideoCapture(0); self.timer_f.start(10); self.btn_live_t.setText("🔌 OFF")
        else:
            self.cap_f = None; self.timer_f.stop(); self.btn_live_t.setText("🔥 LIVE FEED"); self.set_dull_monitors()

    def toggle_interpreter(self, checked):
        if checked:
            if os.path.exists('sign_model.h5'):
                self.model = load_model('sign_model.h5')
                self.actions = np.load('actions.npy').tolist()
                self.cap_l = cv2.VideoCapture(0); self.timer_l.start(10); self.btn_ai.setText("🛑 STOP AI")
            else: self.btn_ai.setChecked(False)
        else:
            if self.cap_l: self.cap_l.release()
            self.cap_l = None; self.timer_l.stop(); self.btn_ai.setText("🚀 START AI"); self.set_dull_monitors()

    def update_source(self):
        if self.is_paused or not self.cap_f: return
        ret, frame = self.cap_f.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.scrub.setValue(int(self.cap_f.get(cv2.CAP_PROP_POS_FRAMES)))
            res = self.processor.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp, hands_visible = self.processor.extract_keypoints(res)
            
            if self.is_recording:
                if hands_visible:
                    if not hasattr(self, 'cd_src'): self.cd_src = QDateTime.currentMSecsSinceEpoch()
                    elapsed = (QDateTime.currentMSecsSinceEpoch() - self.cd_src) / 1000
                    if elapsed < 3:
                        cv2.putText(frame, f"READY: {3-int(elapsed)}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 204), 5)
                    else:
                        cv2.putText(frame, "RECORDING...", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        self.rec_buffer.append(kp)
                        self.progress.setValue(int((len(self.rec_buffer)/30)*100))
                        if len(self.rec_buffer) == 30: 
                            self.finish_capture()
                            if hasattr(self, 'cd_src'): delattr(self, 'cd_src')
                else:
                    if hasattr(self, 'cd_src'): delattr(self, 'cd_src')
                    cv2.putText(frame, "WAITING FOR HAND...", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            self.draw(frame, self.mon_src)
        else:
            self.cap_f.release(); self.cap_f = None; self.set_dull_monitors()

    def update_interpreter(self):
        if not self.cap_l: return
        ret, frame = self.cap_l.read()
        if ret:
            frame = cv2.flip(frame, 1)
            res = self.processor.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp, hands_visible = self.processor.extract_keypoints(res)
            
            if hands_visible:
                if not hasattr(self, 'cd_live'): self.cd_live = QDateTime.currentMSecsSinceEpoch()
                elapsed = (QDateTime.currentMSecsSinceEpoch() - self.cd_live) / 1000
                if elapsed < 3:
                    cv2.putText(frame, f"AI SYNC: {3-int(elapsed)}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 204), 4)
                else:
                    cv2.putText(frame, "AI ACTIVE", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.live_seq.append(kp); self.live_seq = self.live_seq[-30:]
                    if len(self.live_seq) == 30 and self.model:
                        p = self.model.predict(np.expand_dims(self.live_seq, 0), verbose=0)[0]
                        if np.max(p) > 0.85:
                            word = self.actions[np.argmax(p)]
                            if not self.sentence or word != self.sentence[-1]:
                                self.sentence.append(word)
                                self.update_text_display()
            else:
                if hasattr(self, 'cd_live'): delattr(self, 'cd_live')
                cv2.putText(frame, "SCANNING...", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            self.draw(frame, self.mon_live)

    def add_space(self):
        self.sentence.append(" ")
        self.update_text_display()

    def backspace_text(self):
        if self.sentence:
            self.sentence.pop()
            self.update_text_display()

    def clear_text(self):
        self.sentence = []
        self.update_text_display()

    def update_text_display(self):
        # Clean double spaces to keep it neat
        text = "".join(self.sentence).replace("  ", " ")
        self.text_out.setText("TEXT: " + text)

    def draw(self, f, lbl):
        h, w, c = f.shape
        qi = QImage(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).data, w, h, w*3, QImage.Format.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qi).scaled(lbl.width(), lbl.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def refresh_analysis(self):
        self.ax.clear()
        path = "SignData/Signs"
        if os.path.exists(path):
            labels = os.listdir(path)
            counts = [len(os.listdir(os.path.join(path, l))) for l in labels]
            if labels:
                self.ax.bar(labels, counts, color='#00FFCC', alpha=0.7)
                self.ax.set_title("NEURAL DATA DENSITY", color='#00FFCC', fontsize=10, fontweight='bold')
        self.ax.tick_params(colors='#00FFCC')
        self.canvas.draw()

    def sync_brain(self):
        self.progress.setValue(10)
        actions, sequences, labels = [], [], []
        path = "SignData/Signs"
        if not os.path.exists(path): return
        for action in os.listdir(path):
            actions.append(action); ap = os.path.join(path, action)
            for seq in os.listdir(ap):
                try:
                    window = [np.load(os.path.join(ap, seq, f"{i}.npy")) for i in range(30)]
                    sequences.append(window); labels.append(actions.index(action))
                except: continue
        if not sequences: return
        self.progress.setValue(30)
        X = np.array(sequences); y = to_categorical(labels).astype(int)
        model = Sequential([
            Input(shape=(30, 285)),
            LSTM(64, return_sequences=True, activation='relu'),
            LSTM(128, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=80, verbose=0)
        model.save('sign_model.h5'); np.save('actions.npy', np.array(actions))
        self.progress.setValue(100)
        QTimer.singleShot(1500, lambda: self.progress.setValue(0))

    def trigger_capture(self):
        n, ok = QInputDialog.getText(self, "Sign", "Name:")
        if ok and n: self.curr_label = n.upper(); self.rec_buffer = []; self.is_recording = True

    def finish_capture(self):
        self.is_recording = False
        p = os.path.join(self.base_path, "Signs", self.curr_label); os.makedirs(p, exist_ok=True)
        idx = len(os.listdir(p)); sp = os.path.join(p, str(idx)); os.makedirs(sp, exist_ok=True)
        for i, k in enumerate(self.rec_buffer): np.save(os.path.join(sp, f"{i}.npy"), k)
        self.progress.setValue(100); QTimer.singleShot(800, lambda: self.progress.setValue(0))

    def load_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Video", "", "*.mp4")
        if p:
            if self.cap_f: self.cap_f.release()
            self.cap_f = cv2.VideoCapture(p); self.scrub.setMaximum(int(self.cap_f.get(cv2.CAP_PROP_FRAME_COUNT))); self.timer_f.start(10)

    def toggle_pause(self): self.is_paused = not self.is_paused

if __name__ == "__main__":
    app = QApplication(sys.argv); win = CyberSignAI(); win.show(); sys.exit(app.exec())