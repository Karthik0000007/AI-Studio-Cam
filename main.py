import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\ffmpeg-7.1.1-full_build\bin"  # Update if needed

import cv2
from ultralytics import YOLO
import whisper
import sounddevice as sd
import soundfile as sf
import pyttsx3
import time

# === Load YOLO model ===
model = YOLO("yolov8n.pt")

# === Load Whisper model ===
whisper_model = whisper.load_model("base")  # You can try "small" or "medium" for better accuracy

# === Init TTS ===
engine = pyttsx3.init()
engine.setProperty("rate", 170)

# === Webcam Settings ===
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# === Audio Recording Config ===
AUDIO_FILE = "temp.wav"
SAMPLE_RATE = 16000
DURATION = 5  # seconds

print("[INFO] Press 'v' to ask a question | Press 'q' to quit.")

def speak(text):
    engine.say(text)
    engine.runAndWait()

def record_voice():
    print("[🎤] Recording voice...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(AUDIO_FILE, audio, SAMPLE_RATE)
    print("✅ Audio saved.")

def transcribe_audio():
    print("🧠 Transcribing...")
    result = whisper_model.transcribe(AUDIO_FILE)
    return result["text"]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("AI Studio Cam", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('v'):
        speak("Recording now.")
        record_voice()
        try:
            query = transcribe_audio()
            print("[User]:", query)

            # Extract detected object names
            detected_classes = [model.model.names[int(cls)] for cls in results[0].boxes.cls]
            unique_objs = list(set(detected_classes))
            object_summary = ", ".join(unique_objs) if unique_objs else "nothing I can recognize"

            # Basic QA logic
            if "what" in query.lower():
                reply = f"I can see: {object_summary}"
            else:
                reply = "Try asking what I can see."

            print("[AI]:", reply)
            speak(reply)

        except Exception as e:
            print(f"[ERROR]: {e}")
            speak("Something went wrong while transcribing.")

cap.release()
cv2.destroyAllWindows()

# Optional: Delete the audio file after use
if os.path.exists(AUDIO_FILE):
    os.remove(AUDIO_FILE)
