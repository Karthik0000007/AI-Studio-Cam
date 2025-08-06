import os
import cv2
import json
import time
import torch
import faiss
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import open_clip
import sys
sys.path.append(os.path.dirname(__file__))

from db_handler import save_snapshot_to_db, load_all_metadata
import certifi

# === Setup ===
os.makedirs("memory/snapshots", exist_ok=True)

# === Load CLIP model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# === FAISS Index Setup ===
DIM = 512  # CLIP embedding size
faiss_index = faiss.IndexFlatL2(DIM)
embedding_path = "memory/embeddings.faiss"
metadata_path = "memory/metadata.json"

# === Load Metadata from DB or JSON ===
metadata = []
try:
    metadata = load_all_metadata()
    print(f"[✅] Loaded {len(metadata)} entries from database")
except Exception as e:
    print(f"[⚠️] Failed to load metadata from DB: {e}")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"[✅] Loaded {len(metadata)} entries from local JSON")
        except json.JSONDecodeError:
            print("[⚠️] metadata.json is empty or corrupted. Starting fresh.")
            metadata = []

# === Load existing FAISS index if available ===
if os.path.exists(embedding_path) and len(metadata) > 0:
    try:
        faiss_index = faiss.read_index(embedding_path)
        print(f"[✅] Loaded FAISS index with {faiss_index.ntotal} embeddings")
    except Exception as e:
        print(f"[⚠️] Failed to load FAISS index: {e}. Creating new index.")
        faiss_index = faiss.IndexFlatL2(DIM)

# === Save Snapshot + Embedding ===
def save_snapshot_and_embedding(frame, detected_objects):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snap_filename = f"snap_{timestamp}.jpg"
        snap_path = os.path.join("memory/snapshots", snap_filename)
        
        # Save image
        success = cv2.imwrite(snap_path, frame)
        if not success:
            raise RuntimeError(f"Failed to save image to {snap_path}")

        # Convert to RGB + preprocess
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = clip_model.encode_image(img_tensor).cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        # Ensure it's 2D and correct format
        embedding_vector = embedding.astype("float32").reshape(1, -1)
        
        # Verify embedding dimensions
        if embedding_vector.shape[1] != DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {DIM}, got {embedding_vector.shape[1]}")
        
        faiss_index.add(embedding_vector)

        entry = {
            "timestamp": timestamp,
            "datetime": datetime.utcnow().isoformat(),  # JSON serializable
            "snapshot": f"snapshots/{snap_filename}",
            "objects": detected_objects
        }

        metadata.append(entry)
        
        # Try to save to database (will skip silently if MongoDB not available)
        try:
            save_snapshot_to_db(entry.copy())  # Use copy to avoid modifying original
        except Exception as e:
            print(f"[WARNING] Database save failed: {e}")
        
        # Always save locally
        save_memory()

        print(f"[🧠] Snapshot saved: {snap_filename}, Objects: {detected_objects}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save snapshot: {e}")
        raise

# === Search by Query Text ===
def search_similar_scene(query_text, top_k=1):
    try:
        global faiss_index
        if faiss_index.ntotal == 0:
            if os.path.exists(embedding_path):
                print("[ℹ️] Loading FAISS index from file...")
                faiss_index = faiss.read_index(embedding_path)
            else:
                print("[⚠️] FAISS index is empty. No results available.")
                return None

        with torch.no_grad():
            tokens = tokenizer([query_text]).to(device)
            text_features = clip_model.encode_text(tokens).cpu().numpy()
            text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        D, I = faiss_index.search(text_features, top_k)
        
        # Check if we got valid results
        if len(I[0]) == 0 or I[0][0] == -1:
            print("[⚠️] No similar scenes found")
            return None
            
        best_index = I[0][0]
        similarity_score = float(D[0][0])

        if best_index < len(metadata):
            result = metadata[best_index].copy()
            result['similarity_score'] = similarity_score
            print(f"[🔍] Found similar scene (score: {similarity_score:.3f}): {result.get('timestamp', 'unknown')}")
            return result
        else:
            print(f"[⚠️] Index mismatch: {best_index} >= {len(metadata)}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Scene search failed: {e}")
        return None

# === Save FAISS + Metadata ===
def save_memory():
    try:
        # Only save FAISS index if it has data
        if faiss_index.ntotal > 0:
            faiss.write_index(faiss_index, embedding_path)
        else:
            print("[⚠️] FAISS index is empty, skipping save")
        
        # Create a clean copy of metadata for JSON serialization
        clean_metadata = []
        for entry in metadata:
            clean_entry = {}
            for key, value in entry.items():
                # Skip any MongoDB ObjectId fields or other non-serializable objects
                if key != '_id' and not hasattr(value, '__dict__'):
                    # Handle datetime objects
                    if isinstance(value, datetime):
                        clean_entry[key] = value.isoformat()
                    else:
                        clean_entry[key] = value
            clean_metadata.append(clean_entry)
        
        with open(metadata_path, "w") as f:
            json.dump(clean_metadata, f, indent=2, default=str)
        print(f"[ℹ️] Memory saved successfully ({len(clean_metadata)} entries, {faiss_index.ntotal} embeddings)")
    except Exception as e:
        print(f"[ERROR] Failed to save memory: {e}")
        # Don't raise - just log the error so app continues running

# === Get Snapshot X Seconds Ago ===
def get_snapshot_near_seconds_ago(seconds):
    try:
        target_time = datetime.now() - timedelta(seconds=seconds)
        closest = None
        closest_diff = float("inf")

        for entry in metadata:
            try:
                entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d_%H-%M-%S")
                diff = abs((entry_time - target_time).total_seconds())
                if diff < closest_diff:
                    closest_diff = diff
                    closest = entry
            except Exception as e:
                print(f"[⚠️] Error parsing timestamp for entry: {e}")
                continue

        return closest
        
    except Exception as e:
        print(f"[ERROR] Time-based search failed: {e}")
        return None

# === Find Last Seen Object ===
def find_last_seen_object(object_name):
    try:
        for entry in reversed(metadata):
            objects = entry.get("objects", [])
            # Handle both string and list objects
            if isinstance(objects, str):
                objects = [objects]
            
            for obj in objects:
                if isinstance(obj, str) and object_name.lower() in obj.lower():
                    print(f"[🔍] Found '{object_name}' in snapshot from {entry.get('timestamp', 'unknown')}")
                    return entry
        
        print(f"[⚠️] Object '{object_name}' not found in any snapshots")
        return None
    except Exception as e:
        print(f"[ERROR] Object search failed: {e}")
        return None

# === Initialize/Reset Memory System ===
def initialize_memory():
    """Initialize or reset the memory system"""
    global metadata, faiss_index
    try:
        metadata = []
        faiss_index = faiss.IndexFlatL2(DIM)
        
        # Create directories
        os.makedirs("memory/snapshots", exist_ok=True)
        
        print("[✅] Memory system initialized")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize memory: {e}")
        return False

# === Get Memory Stats ===
def get_memory_stats():
    """Get current memory system statistics"""
    try:
        stats = {
            "total_snapshots": len(metadata),
            "total_embeddings": faiss_index.ntotal,
            "memory_files": {
                "embeddings_exist": os.path.exists(embedding_path),
                "metadata_exist": os.path.exists(metadata_path)
            }
        }
        return stats
    except Exception as e:
        print(f"[ERROR] Failed to get memory stats: {e}")
        return None
