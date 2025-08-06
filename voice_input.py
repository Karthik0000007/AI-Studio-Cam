import sounddevice as sd
import soundfile as sf
import whisper

SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "voice_input.wav"

model = whisper.load_model("base")

def capture_and_transcribe():
    print("🎤 Speak now...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(AUDIO_FILE, recording, SAMPLE_RATE)

    print("🧠 Transcribing...")
    result = model.transcribe(AUDIO_FILE)
    text = result["text"].lower().strip()
    print("🗣️ You said:", text)
    return text
