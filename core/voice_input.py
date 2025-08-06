import sounddevice as sd
import soundfile as sf
import whisper
import os

SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "voice_input.wav"

# Initialize model once
_model = None

def get_whisper_model():
    """Lazy load Whisper model"""
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model

def capture_and_transcribe():
    """Capture audio and return transcribed text"""
    try:
        print("🎤 Speak now...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        sf.write(AUDIO_FILE, recording, SAMPLE_RATE)

        print("🧠 Transcribing...")
        model = get_whisper_model()
        result = model.transcribe(AUDIO_FILE)
        text = result["text"].lower().strip()
        print("🗣️ You said:", text)
        
        # Clean up audio file
        if os.path.exists(AUDIO_FILE):
            os.remove(AUDIO_FILE)
            
        return text
    except Exception as e:
        print(f"[ERROR] Voice capture failed: {e}")
        return ""
