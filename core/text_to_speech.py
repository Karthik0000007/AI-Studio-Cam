"""
Text-to-Speech Module

Handles text-to-speech functionality with error handling.
"""

import pyttsx3
import logging

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self, rate=170):
        """Initialize TTS engine"""
        self.rate = rate
        self.engine = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the TTS engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def speak(self, text):
        """Convert text to speech"""
        try:
            print(f"🔊 Speaking: {text}")
            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                logger.warning("TTS engine not available, text only output")
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            print(f"🔊 (TTS Failed): {text}")
    
    def cleanup(self):
        """Clean up TTS resources"""
        try:
            if self.engine:
                self.engine.stop()
                logger.info("TTS engine cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up TTS engine: {e}")

# Global instance for backward compatibility
_tts_engine = TTSEngine()

def tts_speak(text):
    """Legacy function for backward compatibility"""
    _tts_engine.speak(text)