import os
import json
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3
import threading
from PyQt6.QtCore import QThread, pyqtSignal

class AudioEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock() # Prevents overlapping speech crashes

    def say(self, text):
        def run():
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()
        # Running in a separate thread so the GUI doesn't freeze while speaking
        threading.Thread(target=run, daemon=True).start()

class VoiceThread(QThread):
    text_signal = pyqtSignal(str)
    vol_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.q = queue.Queue()
        
        # Load the Vosk model from your project folder
        model_path = "model-vosk" 
        if not os.path.exists(model_path):
            print("Error: 'model-vosk' folder not found!")
            self.model = None
        else:
            self.model = Model(model_path)
            self.rec = KaldiRecognizer(self.model, 16000)

    def callback(self, indata, frames, time, status):
        """This puts audio data into the queue"""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def run(self):
        if not self.model: return

        # Start recording at 16000Hz (Vosk standard)
        try:
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                   channels=1, callback=self.callback):
                
                while self._run_flag:
                    data = self.q.get()
                    if self.rec.AcceptWaveform(data):
                        result = json.loads(self.rec.Result())
                        text = result.get("text", "")
                        if text:
                            self.text_signal.emit(text)
                            self.vol_signal.emit(100)
                    else:
                        # This handles partial speech/volume visualization
                        self.vol_signal.emit(30)
        except Exception as e:
            print(f"Vosk Audio Error: {e}")

    def stop(self):
        self._run_flag = False
        self.wait()