import pyttsx3

engine = pyttsx3.init()

def tts_speak(text):
    print("🔊 Speaking:", text)
    engine.say(text)
    engine.runAndWait()
