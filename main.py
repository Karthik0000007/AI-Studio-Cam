# import cv2
# import time
# import torch
# import whisper
# import threading
# import numpy as np
# from ultralytics import YOLO
# from clip_memory import save_snapshot_and_embedding, metadata
# from tts import tts_speak
# from db_handler import query_recent
# import re
# import os
# os.environ["PATH"] += os.pathsep + r"C:\Program Files\ffmpeg-7.1.1-full_build\bin"

# # === Setup Camera ===
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# # === Load YOLOv8 Model ===
# detector = YOLO("yolov8n.pt")

# # === Load Whisper Model ===
# whisper_model = whisper.load_model("base")

# # === Voice Command Handler ===
# import sounddevice as sd
# import soundfile as sf

# SAMPLE_RATE = 16000
# DURATION = 5
# AUDIO_FILE = "temp.wav"

# def listen_and_transcribe():
#     print("🎤 Listening...")
#     recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
#     sd.wait()
#     sf.write(AUDIO_FILE, recording, SAMPLE_RATE)

#     print("🧠 Transcribing...")
#     result = whisper_model.transcribe(AUDIO_FILE)
#     query = result["text"].lower().strip()
#     print("🗣️ You said:", query)
#     return query

# # === Extract Time from Query ===
# def extract_time_from_query(query):
#     match = re.search(r'(\d+)\s*(second|seconds|minute|minutes)', query)
#     if match:
#         num = int(match.group(1))
#         unit = match.group(2)
#         if "minute" in unit:
#             return num * 60
#         return num
#     return None

# # === Frame Loop ===
# last_detection_time = time.time()
# voice_thread_active = False

# def handle_voice_query():
#     global voice_thread_active
#     voice_thread_active = True

#     query = listen_and_transcribe()
#     seconds = extract_time_from_query(query)

#     if seconds:
#         print(f"⏳ Querying memory {seconds} seconds back...")
#         memories = query_recent(seconds)
#         if memories:
#             objects = memories[-1].get("objects", [])
#             if objects:
#                 reply = f"Around {seconds} seconds ago, I saw: {', '.join(objects)}."
#             else:
#                 reply = f"I didn't see anything distinct {seconds} seconds ago."
#         else:
#             reply = f"Sorry, I couldn't recall anything from {seconds} seconds ago."
#     else:
#         reply = "Sorry, I couldn't understand that."

#     print("💬 Reply:", reply)
#     tts_speak(reply)
#     voice_thread_active = False

# print("🟢 AI Studio Cam is running... Press 'v' to ask a question.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = detector(frame)[0]
#     boxes = results.boxes
#     names = results.names

#     unique_objects = set()
#     for box in boxes:
#         cls_id = int(box.cls[0])
#         name = names[cls_id]
#         unique_objects.add(name)

#         cords = box.xyxy[0].cpu().numpy().astype(int)
#         cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
#         cv2.putText(frame, name, (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # Save snapshot if any object is seen
#     if unique_objects and (time.time() - last_detection_time) > 5:
#         save_snapshot_and_embedding(frame, list(unique_objects))
#         last_detection_time = time.time()

#     # Show camera feed
#     cv2.imshow("AI Studio Cam", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('v') and not voice_thread_active:
#         threading.Thread(target=handle_voice_query).start()

# cap.release()
# cv2.destroyAllWindows()

import os
import cv2
import time
import pyttsx3
import whisper
import sounddevice as sd
import soundfile as sf
from ultralytics import YOLO

# Set PATH for ffmpeg (Whisper dependency)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\ffmpeg-7.1.1-full_build\bin"

# === Local imports ===
from clip_memory import (
    save_snapshot_and_embedding,
    save_memory,
    search_similar_scene,
    find_last_seen_object,
    get_snapshot_near_seconds_ago
)

# === Config ===
AUDIO_FILE = "temp.wav"
SAMPLE_RATE = 16000
DURATION = 5  # seconds
SNAPSHOT_INTERVAL = 15  # seconds

# === Initialize Models ===
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
yolo_model = YOLO("yolov8n.pt")
whisper_model = whisper.load_model("base")
engine = pyttsx3.init()
engine.setProperty("rate", 170)

# === Initialize Camera ===
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

last_snapshot_time = time.time()

def speak(text):
    print("[AI]:", text)
    engine.say(text)
    engine.runAndWait()

def record_voice():
    print("[🎤] Recording voice...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(AUDIO_FILE, audio, SAMPLE_RATE)

def transcribe_audio():
    print("🧠 Transcribing...")
    result = whisper_model.transcribe(AUDIO_FILE)
    return result["text"]

print("[INFO] Press 'v' to speak | Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = yolo_model(frame)
    annotated_frame = results[0].plot()

    # === Get Detected Object Names ===
    detected_classes = [yolo_model.model.names[int(cls)] for cls in results[0].boxes.cls]
    unique_objects = list(set(detected_classes))

    # === Show Frame with Boxes ===
    cv2.imshow("AI Studio Cam", annotated_frame)

    # === Take Snapshot Every X Seconds ===
    if time.time() - last_snapshot_time > SNAPSHOT_INTERVAL:
        save_snapshot_and_embedding(frame, unique_objects)
        last_snapshot_time = time.time()

    # === Key Presses ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('v'):
        speak("I'm listening now.")
        try:
            record_voice()
            query = transcribe_audio()
            print("[User]:", query)

            query_lower = query.lower()

            # === Object Recall ===
            if "last" in query_lower and "see" in query_lower:
                for obj in ["person", "phone", "laptop", "chair", "bottle", "stapler"]:  # add more if needed
                    if obj in query_lower:
                        match = find_last_seen_object(obj)
                        if match:
                            reply = f"I last saw a {obj} at {match['timestamp']} with: {', '.join(match['objects'])}."
                        else:
                            reply = f"I haven't seen a {obj} yet."
                        break
                else:
                    # fallback to CLIP semantic search
                    match = search_similar_scene(query)
                    reply = f"I found something similar at {match['timestamp']} seeing: {', '.join(match['objects'])}." if match else "I couldn’t find a moment like that."

            # === Time-based Recall ===
            elif "seconds ago" in query_lower:
                try:
                    seconds = int([s for s in query_lower.split() if s.isdigit()][0])
                    match = get_snapshot_near_seconds_ago(seconds)
                    reply = f"Around {seconds} seconds ago, I saw: {', '.join(match['objects'])}." if match else "I couldn't find anything from that time."
                except:
                    reply = "Sorry, I couldn't understand the time reference."

            # === General Visual QA ===
            else:
                if unique_objects:
                    reply = f"I can currently see: {', '.join(unique_objects)}."
                else:
                    reply = "I'm not detecting anything clearly right now."

            speak(reply)

        except Exception as e:
            print(f"[ERROR]: {e}")
            speak("Sorry, I had trouble understanding that.")

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
save_memory()

if os.path.exists(AUDIO_FILE):
    os.remove(AUDIO_FILE)
