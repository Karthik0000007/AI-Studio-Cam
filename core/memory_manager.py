"""
Memory Manager Module - Further simplified version
Manages visual memory, embeddings, and database operations.
"""

import logging
import os
import cv2
import json
import torch
import faiss
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import open_clip

logger = logging.getLogger(__name__)

class MemoryManager:
    """Unified memory manager for visual memory, embeddings, and database operations"""
    
    def __init__(self, snapshot_interval=15):
        self.snapshot_interval = snapshot_interval
        self.last_snapshot_time = 0
        
        # CLIP model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.DIM = 512  # CLIP embedding size
        
        # Paths
        self.embedding_path = "memory/embeddings.faiss"
        self.metadata_path = "memory/metadata.json"
        self.snapshots_dir = "memory/snapshots"
        
        # Initialize components
        self._init_clip()
        self._init_storage()
        self._init_mongodb()
        self._load_existing_data()
    
    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            logger.info("CLIP model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise
    
    def _init_storage(self):
        """Initialize storage directories"""
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.faiss_index = faiss.IndexFlatL2(self.DIM)
        self.metadata = []
    
    def _init_mongodb(self):
        """Initialize MongoDB connection if available"""
        self.mongodb_enabled = False
        try:
            from pymongo.mongo_client import MongoClient
            from pymongo.server_api import ServerApi
            
            uri = "mongodb+srv://sairamkarthikmalladi:<b413Sf103OeUIouI>@cluster0.zntg1pw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            self.client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            
            self.db = self.client["Ai_Studio_Cam"]
            self.collection = self.db["snapshots"]
            self.mongodb_enabled = True
            logger.info("✅ Connected to MongoDB Atlas")
        except Exception:
            logger.info("📁 Using local storage only")
    
    def _load_existing_data(self):
        """Load existing metadata and FAISS index"""
        self.metadata = self._load_all_metadata()
        if not self.metadata and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except json.JSONDecodeError:
                self.metadata = []
        
        if os.path.exists(self.embedding_path) and len(self.metadata) > 0:
            try:
                self.faiss_index = faiss.read_index(self.embedding_path)
            except Exception:
                self.faiss_index = faiss.IndexFlatL2(self.DIM)
    
    def should_take_snapshot(self, current_time):
        """Check if it's time to take a snapshot"""
        return (current_time - self.last_snapshot_time) > self.snapshot_interval
    
    def save_snapshot(self, frame, detected_objects, current_time):
        """Save snapshot with embedding"""
        try:
            self._save_snapshot_and_embedding(frame, detected_objects)
            self.last_snapshot_time = current_time
            logger.info(f"Snapshot saved with objects: {detected_objects}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            raise
    
    def _save_snapshot_and_embedding(self, frame, detected_objects):
        """Save snapshot image and generate CLIP embedding"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snap_filename = f"snap_{timestamp}.jpg"
        snap_path = os.path.join(self.snapshots_dir, snap_filename)
        
        if not cv2.imwrite(snap_path, frame):
            raise RuntimeError(f"Failed to save image to {snap_path}")

        # Generate CLIP embedding
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.clip_model.encode_image(img_tensor).cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        self.faiss_index.add(embedding.astype("float32").reshape(1, -1))

        # Create metadata entry
        entry = {
            "timestamp": timestamp,
            "datetime": datetime.utcnow().isoformat(),
            "snapshot": f"snapshots/{snap_filename}",
            "objects": detected_objects
        }
        
        self.metadata.append(entry)
        self._save_snapshot_to_db(entry.copy())
        self._save_local_data()
    
    def _save_local_data(self):
        """Save metadata and FAISS index to local files"""
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            faiss.write_index(self.faiss_index, self.embedding_path)
        except Exception as e:
            logger.error(f"Failed to save local data: {e}")
    
    def _save_snapshot_to_db(self, metadata_entry):
        """Save snapshot to MongoDB if available"""
        if self.mongodb_enabled:
            try:
                result = self.collection.insert_one(metadata_entry)
                logger.info(f"[DB] Snapshot saved with ID: {result.inserted_id}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to save to database: {e}")
    
    def _load_all_metadata(self):
        """Load metadata from MongoDB if available"""
        if self.mongodb_enabled:
            try:
                return list(self.collection.find({}, {"_id": 0}))
            except Exception as e:
                logger.error(f"[ERROR] Failed to load metadata from database: {e}")
        return []
    
    def _query_mongodb(self, query, fallback_func):
        """Generic MongoDB query with fallback to local search"""
        if self.mongodb_enabled:
            try:
                return list(self.collection.find(query, {"_id": 0}))
            except Exception as e:
                logger.error(f"[ERROR] MongoDB query failed: {e}")
        return fallback_func()
    
    def query_by_object(self, object_name):
        """Query by object from MongoDB if available, otherwise search local metadata"""
        def local_search():
            return [entry for entry in self.metadata 
                    if object_name.lower() in [obj.lower() for obj in entry['objects']]]
        
        return self._query_mongodb({"objects": object_name}, local_search)
    
    def query_recent(self, seconds=30):
        """Query recent snapshots from MongoDB if available, otherwise search local metadata"""
        def local_search():
            target_time = datetime.utcnow() - timedelta(seconds=seconds)
            results = []
            for entry in self.metadata:
                try:
                    entry_time = datetime.fromisoformat(entry['datetime'].replace('Z', '+00:00'))
                    if entry_time >= target_time:
                        results.append(entry)
                except Exception:
                    continue
            return results
        
        if self.mongodb_enabled:
            try:
                now = datetime.utcnow()
                threshold = now - timedelta(seconds=seconds)
                return list(self.collection.find({"datetime": {"$gte": threshold}}, {"_id": 0}))
            except Exception as e:
                logger.error(f"[ERROR] Failed to query recent snapshots: {e}")
        
        return local_search()
    
    def search_similar_scene(self, query):
        """Search for similar scenes using CLIP"""
        if not self.metadata or self.faiss_index.ntotal == 0:
            return None
        
        try:
            text_tokens = self.clip_tokenizer([query]).to(self.device)
            with torch.no_grad():
                text_embedding = self.clip_model.encode_text(text_tokens).cpu().numpy()
                text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
            
            D, I = self.faiss_index.search(text_embedding.astype("float32"), min(5, len(self.metadata)))
            
            if I[0].size > 0 and D[0][0] < 1.0:
                best_match_idx = I[0][0]
                if best_match_idx < len(self.metadata):
                    return self.metadata[best_match_idx]
            return None
            
        except Exception as e:
            logger.error(f"Scene search failed: {e}")
            return None
    
    def find_last_seen_object(self, object_name):
        """Find when an object was last seen"""
        if not self.metadata:
            return None
        
        for entry in reversed(self.metadata[-100:]):
            if object_name.lower() in [obj.lower() for obj in entry['objects']]:
                return entry
        return None
    
    def get_snapshot_near_seconds_ago(self, seconds):
        """Get snapshot from X seconds ago"""
        if not self.metadata:
            return None
        
        target_time = datetime.utcnow() - timedelta(seconds=seconds)
        closest_entry = None
        min_time_diff = float('inf')
        
        for entry in self.metadata:
            try:
                entry_time = datetime.fromisoformat(entry['datetime'].replace('Z', '+00:00'))
                time_diff = abs((entry_time - target_time).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_entry = entry
            except Exception:
                continue
        
        return closest_entry if closest_entry and min_time_diff <= 30 else None
    
    def save_all_memory(self):
        """Save all memory data"""
        try:
            self._save_local_data()
            logger.info("Memory saved successfully")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise
    
    def get_memory_stats(self):
        """Get memory statistics"""
        return {
            "total_snapshots": len(self.metadata),
            "faiss_embeddings": self.faiss_index.ntotal,
            "last_snapshot": self.metadata[-1]['timestamp'] if self.metadata else None,
            "snapshot_interval": self.snapshot_interval,
            "mongodb_enabled": self.mongodb_enabled
        }
    
    def clear_memory(self):
        """Clear all memory data (use with caution)"""
        try:
            self.metadata.clear()
            self.faiss_index = faiss.IndexFlatL2(self.DIM)
            
            # Remove files
            for path in [self.embedding_path, self.metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            # Clear snapshots directory
            for file in os.listdir(self.snapshots_dir):
                if file.endswith('.jpg'):
                    os.remove(os.path.join(self.snapshots_dir, file))
            
            # Clear MongoDB if available
            if self.mongodb_enabled:
                try:
                    self.collection.delete_many({})
                    logger.info("MongoDB collection cleared")
                except Exception as e:
                    logger.warning(f"Failed to clear MongoDB: {e}")
            
            logger.info("Memory cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            raise