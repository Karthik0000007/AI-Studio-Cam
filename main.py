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
