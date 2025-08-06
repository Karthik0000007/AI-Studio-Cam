"""
Test Microphone and TTS Functionality

Simple test script to verify voice input and TTS are working.
"""

import os
import logging
from tts import tts_speak
from voice_input import capture_and_transcribe

# Set PATH for ffmpeg (Whisper dependency)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\ffmpeg-7.1.1-full_build\bin"

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_microphone():
    """Test microphone and TTS functionality"""
    try:
        print("=== Testing Microphone and TTS ===")
        
        # Test voice input
        print("Testing voice input...")
        query = capture_and_transcribe()
        
        if query:
            print(f"✅ Voice input successful: '{query}'")
            
            # Test TTS
            print("Testing TTS...")
            tts_speak(f"You said: {query}")
            print("✅ TTS test completed")
        else:
            print("❌ Voice input failed or returned empty")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_microphone()