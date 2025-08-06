"""
Memory Manager Module

Manages visual memory, embeddings, and database operations.
"""

import logging
import os
import sys
sys.path.append(os.path.dirname(__file__))

from clip_memory import (
    save_snapshot_and_embedding,
    save_memory,
    search_similar_scene,
    find_last_seen_object,
    get_snapshot_near_seconds_ago
)

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, snapshot_interval=15):
        """Initialize memory manager"""
        self.snapshot_interval = snapshot_interval
        self.last_snapshot_time = 0
        
    def should_take_snapshot(self, current_time):
        """Check if it's time to take a snapshot"""
        return (current_time - self.last_snapshot_time) > self.snapshot_interval
    
    def save_snapshot(self, frame, detected_objects, current_time):
        """Save snapshot with embedding"""
        try:
            save_snapshot_and_embedding(frame, detected_objects)
            self.last_snapshot_time = current_time
            logger.info(f"Snapshot saved with objects: {detected_objects}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            raise
    
    def search_similar_scene(self, query):
        """Search for similar scenes using CLIP"""
        try:
            return search_similar_scene(query)
        except Exception as e:
            logger.error(f"Scene search failed: {e}")
            return None
    
    def find_last_seen_object(self, object_name):
        """Find when an object was last seen"""
        try:
            return find_last_seen_object(object_name)
        except Exception as e:
            logger.error(f"Object search failed: {e}")
            return None
    
    def get_snapshot_near_seconds_ago(self, seconds):
        """Get snapshot from X seconds ago"""
        try:
            return get_snapshot_near_seconds_ago(seconds)
        except Exception as e:
            logger.error(f"Time-based search failed: {e}")
            return None
    
    def save_all_memory(self):
        """Save all memory data"""
        try:
            save_memory()
            logger.info("Memory saved successfully")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise