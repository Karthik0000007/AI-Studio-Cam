import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\ffmpeg-7.1.1-full_build\bin"

import whisper
import sounddevice as sd
import soundfile as sf

# Config
SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "test_audio.wav"

# Step 1: Record
print("🎤 Speak now...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
sf.write(AUDIO_FILE, recording, SAMPLE_RATE)
print("✅ Recording saved.")

# Step 2: Transcribe with Whisper
model = whisper.load_model("base")
result = model.transcribe(AUDIO_FILE)
print("📜 You said:", result["text"])
