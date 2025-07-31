from tts import tts_speak
import whisper
import sounddevice as sd
import soundfile as sf
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\ffmpeg-7.1.1-full_build\bin"

SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "test.wav"

print("🎤 Speak now...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
sf.write(AUDIO_FILE, recording, SAMPLE_RATE)

print("🧠 Transcribing...")
model = whisper.load_model("base")
result = model.transcribe(AUDIO_FILE)
query = result["text"].lower()
print("🗣️ You said:", query)

tts_speak(f"You said: {query}")
