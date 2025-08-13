"""
Voice Processor Module

Handles voice input, transcription, text-to-speech, and query processing functionality.
"""

import os
import pyttsx3
import logging
import sys
import sounddevice as sd
import whisper
import warnings
import re

# Suppress FP16 warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
DURATION = 5

class VoiceProcessor:
    def __init__(self, speech_rate=170):
        """Initialize TTS engine and Whisper model"""
        self.speech_rate = speech_rate
        self.engine = None
        self.whisper_model = None
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.speech_rate)
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def _get_whisper_model(self):
        """Lazy load Whisper model - using tiny model for speed"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("tiny")
        return self.whisper_model
    
    def speak(self, text):
        """Convert text to speech"""
        try:
            print(f"[AI]: {text}")
            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                logger.warning("TTS engine not available, text only output")
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            print(f"[AI] (TTS Failed): {text}")
    
    def listen(self):
        """Capture and transcribe voice input"""
        try:
            self.speak("I'm listening now.")
            transcribed_text = self._capture_and_transcribe()
            logger.info(f"Voice input transcribed: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            logger.error(f"Voice input failed: {e}")
            return ""
    
    def _capture_and_transcribe(self):
        """Capture audio and return transcribed text"""
        try:
            print("🎤 Speak now...")
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            print("🧠 Transcribing...")
            model = self._get_whisper_model()
            
            audio_data = recording.flatten().astype('float32')
            result = model.transcribe(audio_data, fp16=False)
            
            text = result["text"].lower().strip()
            print("🗣️ You said:", text)
            
            return text
        except Exception as e:
            print(f"[ERROR] Voice capture failed: {e}")
            return ""
    
    def process_query(self, query, memory_manager, current_objects):
        """Process voice query and generate response"""
        try:
            query_lower = query.lower()
            
            if "last" in query_lower and "see" in query_lower:
                return self._handle_object_recall(query_lower, memory_manager)
            elif "seconds ago" in query_lower:
                return self._handle_time_recall(query_lower, memory_manager)
            else:
                return self._handle_current_view(current_objects)
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return "Sorry, I had trouble processing that request."
    
    def _handle_object_recall(self, query_lower, memory_manager):
        """Handle object recall queries"""
        common_objects = ["person", "phone", "laptop", "chair", "bottle", "stapler", "book", "cup"]
        
        for obj in common_objects:
            if obj in query_lower:
                match = memory_manager.find_last_seen_object(obj)
                if match:
                    return f"I last saw a {obj} at {match['timestamp']} with: {', '.join(match['objects'])}."
                else:
                    return f"I haven't seen a {obj} yet."
        
        # Fallback to semantic search
        match = memory_manager.search_similar_scene(query_lower)
        if match:
            return f"I found something similar at {match['timestamp']} seeing: {', '.join(match['objects'])}."
        else:
            return "I couldn't find a moment like that."
    
    def _handle_time_recall(self, query_lower, memory_manager):
        """Handle time-based recall queries"""
        try:
            time_match = re.search(r'(\d+)\s*seconds?\s*ago', query_lower)
            if time_match:
                seconds = int(time_match.group(1))
                match = memory_manager.get_snapshot_near_seconds_ago(seconds)
                if match:
                    return f"{seconds} seconds ago, I saw: {', '.join(match['objects'])}."
                else:
                    return f"I don't have a snapshot from {seconds} seconds ago."
            else:
                return "Please specify how many seconds ago, like '30 seconds ago'."
        except Exception as e:
            logger.error(f"Time recall processing failed: {e}")
            return "Sorry, I had trouble with that time-based request."
    
    def _handle_current_view(self, current_objects):
        """Handle current view queries"""
        if current_objects:
            return f"I currently see: {', '.join(current_objects)}."
        else:
            return "I don't see any objects in the current view."
    
    def cleanup(self):
        """Clean up TTS resources"""
        try:
            if self.engine:
                self.engine.stop()
                logger.info("TTS engine cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up TTS engine: {e}")