"""
Voice Processor Module

Handles voice input, transcription, and text-to-speech functionality.
"""

import os
import pyttsx3
import logging
import sys
sys.path.append(os.path.dirname(__file__))

from voice_input import capture_and_transcribe

logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self, speech_rate=170):
        """Initialize TTS engine"""
        self.speech_rate = speech_rate
        self.engine = None
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
            transcribed_text = capture_and_transcribe()
            logger.info(f"Voice input transcribed: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            logger.error(f"Voice input failed: {e}")
            return ""
    
    def process_query(self, query, memory_manager, current_objects):
        """Process voice query and generate response"""
        try:
            query_lower = query.lower()
            
            # Object recall queries
            if "last" in query_lower and "see" in query_lower:
                return self._handle_object_recall(query_lower, memory_manager)
            
            # Time-based recall queries
            elif "seconds ago" in query_lower:
                return self._handle_time_recall(query_lower, memory_manager)
            
            # Current view queries
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
            # Extract seconds from query
            words = query_lower.split()
            seconds = None
            for word in words:
                if word.isdigit():
                    seconds = int(word)
                    break
            
            if seconds is None:
                return "Sorry, I couldn't understand the time reference."
            
            match = memory_manager.get_snapshot_near_seconds_ago(seconds)
            if match:
                return f"Around {seconds} seconds ago, I saw: {', '.join(match['objects'])}."
            else:
                return "I couldn't find anything from that time."
                
        except Exception as e:
            logger.error(f"Time recall processing failed: {e}")
            return "Sorry, I couldn't understand the time reference."
    
    def _handle_current_view(self, current_objects):
        """Handle current view queries"""
        if current_objects:
            return f"I can currently see: {', '.join(current_objects)}."
        else:
            return "I'm not detecting anything clearly right now."
    
    def cleanup(self):
        """Clean up voice processor resources"""
        try:
            if self.engine:
                self.engine.stop()
                logger.info("Voice processor cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up voice processor: {e}")