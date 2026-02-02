import sys
import time
import re
import os
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from vision_module import CameraThread
from audio_module import VoiceThread, AudioEngine

# --- STYLING CONSTANTS ---
ACCENT_COLOR = "#00FFCC"
DANGER_COLOR = "#FF4B4B"
BG_DARK = "#0e1117"
PANEL_BG = "#1a1c23"
TEXT_PRIMARY = "#E0E0E0"
TEXT_SECONDARY = "#888888"
HEADER_STYLE = f"color: {TEXT_SECONDARY}; font-weight: bold; font-size: 13px; letter-spacing: 1px;"

class DesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Sign Interpreter Pro")
        self.sentence_buffer = []
        self.last_input_time = time.time()
        self.silence_threshold = 1.8 # Seconds to wait before speaking
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(f"background-color: {BG_DARK}; color: {TEXT_PRIMARY}; font-family: 'Segoe UI', sans-serif;")
        
        self.audio_engine = AudioEngine()
        self.tts_enabled = False 
        self.mode = "SIGN"
        self.cam = None
        self.voice = None
        self.last_predicted_word = "" 
        
        # Initialize the timer before calling UI so connections don't fail
        self.speech_timer = QTimer()
        self.speech_timer.timeout.connect(self.process_buffer)
        self.speech_timer.start(500) 

        self.init_ui()

    # --- CLASS METHODS (Correctly Indented) ---

    def clean_text(self, text):
        # Removes double words: "Hello Hello" -> "Hello"
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
        return text.capitalize() + "."

    def handle_sign_translation(self, text):
        if self.mode == "SIGN" and text != self.last_predicted_word:
            self.add_to_chat("S", text)
            self.sentence_buffer.append(text)
            self.last_input_time = time.time()
            self.last_predicted_word = text

    def handle_voice_input(self, text):
        if self.mode == "SPEECH":
            self.add_to_chat("N", text)
            self.sentence_buffer.extend(text.split())
            self.last_input_time = time.time()

    def process_buffer(self):
        if self.tts_enabled and self.sentence_buffer:
            if time.time() - self.last_input_time > self.silence_threshold:
                full_sentence = " ".join(self.sentence_buffer)
                clean_sentence = self.clean_text(full_sentence)
                self.audio_engine.say(clean_sentence)
                self.sentence_buffer.clear() 

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # --- SIDEBAR ---
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(260)
        sidebar_frame.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 15px; border: 1px solid #2d2f3b;")
        sidebar = QVBoxLayout(sidebar_frame)
        
        sidebar.addStretch(1)
        sidebar.addWidget(QLabel("SYSTEM CONTROL", styleSheet=HEADER_STYLE), alignment=Qt.AlignmentFlag.AlignCenter)
        sidebar.addSpacing(15)

        self.run_btn = QPushButton("RUN SYSTEM")
        self.run_btn.setFixedHeight(55)
        self.run_btn.setStyleSheet(f"background-color: {ACCENT_COLOR}; color: black; font-weight: 800; border-radius: 10px; font-size: 14px;")
        self.run_btn.clicked.connect(self.toggle_system)

        self.mode_btn = QPushButton("MODE: SIGN TO TEXT")
        self.mode_btn.setFixedHeight(45)
        self.mode_btn.setStyleSheet(f"border: 2px solid {ACCENT_COLOR}; color: {ACCENT_COLOR}; font-weight: bold; border-radius: 10px; margin-top: 10px;")
        self.mode_btn.clicked.connect(self.toggle_mode)

        self.speech_toggle = QPushButton("VOICE OUTPUT: OFF")
        self.speech_toggle.setFixedHeight(45)
        self.speech_toggle.setStyleSheet("background-color: #2d2f3b; color: #888; font-weight: bold; border-radius: 10px; margin-top: 10px;")
        self.speech_toggle.clicked.connect(self.toggle_speech_output)

        self.space_btn = QPushButton("INSERT SPACE")
        self.space_btn.setFixedHeight(45)
        self.space_btn.setStyleSheet(f"border: 1px solid {ACCENT_COLOR}; color: {ACCENT_COLOR}; border-radius: 10px; margin-top: 10px;")
        self.space_btn.clicked.connect(lambda: self.chat.insertPlainText(" "))

        self.delete_btn = QPushButton("DELETE LAST WORD")
        self.delete_btn.setFixedHeight(45)
        self.delete_btn.setStyleSheet(f"border: 1px solid {DANGER_COLOR}; color: {DANGER_COLOR}; border-radius: 10px; margin-top: 10px;")
        self.delete_btn.clicked.connect(self.delete_last_entry)

        self.clear_btn = QPushButton("CLEAR CONVERSATION")
        self.clear_btn.setFixedHeight(45)
        self.clear_btn.setStyleSheet(f"border: 1px solid #444; color: {TEXT_SECONDARY}; border-radius: 10px; margin-top: 10px;")
        self.clear_btn.clicked.connect(self.chat_clear)

        self.quit_btn = QPushButton("QUIT APPLICATION")
        self.quit_btn.setFixedHeight(45)
        self.quit_btn.setStyleSheet(f"border: 1px solid #444; color: {DANGER_COLOR}; border-radius: 10px; margin-top: 10px;")
        self.quit_btn.clicked.connect(self.close)

        sidebar.addWidget(self.run_btn); sidebar.addWidget(self.mode_btn); sidebar.addWidget(self.speech_toggle)
        sidebar.addWidget(self.space_btn); sidebar.addWidget(self.delete_btn); sidebar.addWidget(self.clear_btn)
        sidebar.addWidget(self.quit_btn); sidebar.addStretch(1)
        
        dev_label = QLabel("DEVELOPED BY: SAILOS RARAMAI MAPFUMO")
        dev_label.setStyleSheet("color: #4d515e; font-size: 10px; font-weight: bold;")
        sidebar.addWidget(dev_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- VIDEO AREA ---
        video_area = QVBoxLayout()
        title_label = QLabel("ADAPTIVE SIGN INTERPRETER")
        title_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 32px; font-weight: 900;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.feed = QLabel("SYSTEM STANDBY")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("color: #2d2f3b; font-size: 28px; font-weight: 900; background: black; border-radius: 20px; border: 2px solid #1a1c23;")
        
        self.mic_label = QLabel("🎤 LIVE SPEECH RECOGNITION")
        self.mic_label.setFixedSize(260, 35)
        self.mic_label.setStyleSheet(f"background: rgba(255, 75, 75, 0.15); color: {DANGER_COLOR}; border-radius: 17px; font-weight: 800; font-size: 11px;")
        self.mic_label.hide()

        video_area.addWidget(title_label)
        video_area.addWidget(self.feed, stretch=1)
        video_area.addWidget(self.mic_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- RIGHT PANEL ---
        right_frame = QFrame()
        right_frame.setFixedWidth(300)
        right_frame.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 15px; border: 1px solid #2d2f3b;")
        right_panel = QVBoxLayout(right_frame)
        
        right_panel.addWidget(QLabel("CORE STATUS", styleSheet=HEADER_STYLE), alignment=Qt.AlignmentFlag.AlignCenter)
        self.status_box = QLabel("IDLE")
        self.status_box.setFixedHeight(100)
        self.status_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_box.setStyleSheet(f"background: #12141a; border: 1px solid #2d2f3b; font-size: 26px; color: #3d4150; border-radius: 12px; font-weight: 900;")
        
        self.init_bar = QProgressBar()
        self.init_bar.setFixedHeight(8)
        self.init_bar.setTextVisible(False)
        self.init_bar.setStyleSheet(f"QProgressBar {{ background: #12141a; border-radius: 4px; border: none; }} QProgressBar::chunk {{ background: {ACCENT_COLOR}; border-radius: 4px; }}")
        self.init_bar.hide()

        right_panel.addWidget(self.status_box); right_panel.addWidget(self.init_bar)
        right_panel.addSpacing(20)
        right_panel.addWidget(QLabel("CONVERSATION", styleSheet=HEADER_STYLE), alignment=Qt.AlignmentFlag.AlignCenter)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setStyleSheet(f"background: #12141a; border-radius: 12px; padding: 12px; border: 1px solid #2d2f3b; font-size: 15px; color: {TEXT_PRIMARY};")
        right_panel.addWidget(self.chat)

        main_layout.addWidget(sidebar_frame); main_layout.addLayout(video_area, stretch=2); main_layout.addWidget(right_frame)

    def toggle_system(self):
        if (self.cam and self.cam.isRunning()) or (self.voice and self.voice.isRunning()):
            self.stop_all()
        else:
            self.start_booting()

    def start_booting(self):
        self.run_btn.setEnabled(False)
        self.init_bar.show(); self.init_bar.setValue(0)
        self.status_box.setText("BOOTING")
        self.status_box.setStyleSheet(f"border: 1px solid #FFCC00; color: #FFCC00; font-size: 22px; font-weight: 900; border-radius: 12px;")
        self.load_val = 0
        self.boot_timer = QTimer(); self.boot_timer.timeout.connect(self.boot_step); self.boot_timer.start(20)

    def boot_step(self):
        self.load_val += 4
        self.init_bar.setValue(self.load_val)
        if self.load_val >= 100:
            self.boot_timer.stop()
            self.run_system()

    def run_system(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("STOP SYSTEM")
        self.run_btn.setStyleSheet(f"background-color: {DANGER_COLOR}; color: white; font-weight: 800; border-radius: 10px;")
        
        self.cam = CameraThread()
        self.cam.change_pixmap.connect(self.update_video_feed)
        self.cam.status_signal.connect(self.update_progress_and_status)
        if hasattr(self.cam, 'sign_signal'):
            self.cam.sign_signal.connect(self.handle_sign_translation)
        self.cam.start()

        self.voice = VoiceThread()
        self.voice.text_signal.connect(self.handle_voice_input)
        self.voice.vol_signal.connect(self.pulse_label)
        self.voice.start()

    def stop_all(self):
        if self.voice:
            self.voice.stop()
            self.voice = None
        if self.cam: 
            self.cam.stop()
            self.cam = None
            
        self.run_btn.setText("RUN SYSTEM")
        self.run_btn.setEnabled(True)
        self.run_btn.setStyleSheet(f"background-color: {ACCENT_COLOR}; color: black; font-weight: 800; border-radius: 10px; font-size: 14px;")
        
        self.status_box.setText("IDLE")
        self.status_box.setStyleSheet("border: 1px solid #2d2f3b; color: #3d4150; font-size: 26px; font-weight: 900; border-radius: 12px;")
        
        self.feed.clear()
        self.feed.setText("SYSTEM STANDBY")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setStyleSheet("color: #2d2f3b; font-size: 28px; font-weight: 900; background: black; border-radius: 20px; border: 2px solid #1a1c23;")
        
        self.init_bar.hide()
        self.mic_label.hide()
        self.last_predicted_word = ""

    def update_video_feed(self, img):
        if self.cam and self.cam.isRunning():
            self.feed.setPixmap(QPixmap.fromImage(img).scaled(
                self.feed.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))

    def update_progress_and_status(self, val):
        if self.cam and self.cam.isRunning():
            self.init_bar.show()
            self.init_bar.setValue(val)
            status_text = "LISTENING" if self.mode == "SPEECH" else "WATCHING"
            self.status_box.setText(status_text)
            self.status_box.setStyleSheet(f"border: 2px solid {ACCENT_COLOR}; color: {ACCENT_COLOR}; font-size: 26px; font-weight: 900; border-radius: 12px;")

    def toggle_mode(self):
        self.mode = "SPEECH" if self.mode == "SIGN" else "SIGN"
        self.mode_btn.setText(f"MODE: {self.mode} TO TEXT")
        self.last_predicted_word = "" 

    def toggle_speech_output(self):
        self.tts_enabled = not self.tts_enabled
        self.speech_toggle.setText(f"VOICE OUTPUT: {'ON' if self.tts_enabled else 'OFF'}")
        style = f"background: rgba(0, 255, 204, 0.1); border: 1px solid {ACCENT_COLOR}; color: {ACCENT_COLOR};" if self.tts_enabled else "background: #2d2f3b; border: 1px solid #444; color: #888;"
        self.speech_toggle.setStyleSheet(style + "font-weight: bold; border-radius: 10px; margin-top: 10px;")

    def add_to_chat(self, prefix, text):
        color = "#FFCC00" if prefix == "N" else ACCENT_COLOR
        self.chat.append(f'<b style="color: {color};">{prefix}:</b> {text}')

    def chat_clear(self): 
        self.chat.clear()
        self.last_predicted_word = ""

    def delete_last_entry(self):
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(QTextCursor.MoveOperation.PreviousBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        self.chat.setTextCursor(cursor)
    
    def pulse_label(self, vol):
        if vol > 25 and self.mode == "SPEECH": 
            self.mic_label.show()
        else: 
            self.mic_label.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DesktopApp()
    win.showMaximized()
    sys.exit(app.exec())