# # clip_memory.py
# import os
# import cv2
# import json
# import time
# import torch
# import faiss
# import numpy as np
# from PIL import Image
# from datetime import datetime
# import open_clip
# from datetime import datetime, timedelta
# from db_handler import save_snapshot_to_db
# from db_handler import load_all_metadata
# from datetime import datetime

# metadata = load_all_metadata()

# # Create required folders
# os.makedirs("memory/snapshots", exist_ok=True)

# # === Load CLIP model ===
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
# clip_model = clip_model.to(device)
# tokenizer = open_clip.get_tokenizer("ViT-B-32")

# # === FAISS Index Setup ===
# DIM = 512  # CLIP embedding size
# faiss_index = faiss.IndexFlatL2(DIM)
# embedding_path = "memory/embeddings.faiss"
# metadata_path = "memory/metadata.json"

# # === In-memory metadata (loaded from disk or created fresh) ===
# metadata = []
# if os.path.exists(metadata_path):
#     try:
#         with open(metadata_path, "r") as f:
#             metadata = json.load(f)
#     except json.JSONDecodeError:
#         print("[⚠️] metadata.json is empty or corrupted. Starting fresh.")

# def save_snapshot_and_embedding(frame, detected_objects):
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     snap_name = f"snap_{timestamp}.jpg"
#     snap_path = os.path.join("memory/snapshots", snap_name)
#     cv2.imwrite(snap_path, frame)

#     # Convert to RGB + preprocess
#     img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

#     with torch.no_grad():
#         embedding = clip_model.encode_image(img_tensor).cpu().numpy()
#         embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # Normalize
#         faiss_index.add(embedding)

#     image_filename = f"snapshots/snap_{timestamp}.jpg"
#     # Save metadata
#     entry = {
#     "timestamp": timestamp,
#     "datetime": datetime.utcnow(),  # For time-based queries
#     "snapshot": image_filename,
#     "objects": detected_objects
#     }

#     metadata.append(entry)
#     embedding_vector = np.array(embedding).astype("float32")

#     # Ensure it's 2D: shape (1, vector_dim)
#     if embedding_vector.ndim == 1:
#         embedding_vector = embedding_vector.reshape(1, -1)

#     faiss_index.add(embedding_vector)
#     save_snapshot_to_db(entry)

#     print(f"[🧠] Snapshot saved: {snap_name}, Objects: {detected_objects}")

# def search_similar_scene(query_text, top_k=1):
#     # Load index if not already done
#     if faiss_index.ntotal == 0 and os.path.exists(embedding_path):
#         faiss_index = faiss.read_index(embedding_path)

#     # Encode query text
#     with torch.no_grad():
#         tokens = tokenizer([query_text]).to(device)
#         text_features = clip_model.encode_text(tokens).cpu().numpy()
#         text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

#     # Search
#     D, I = faiss_index.search(text_features, top_k)
#     best_index = I[0][0]
#     if best_index < len(metadata):
#         return metadata[best_index]
#     else:
#         return None


# def save_memory():
#     faiss.write_index(faiss_index, embedding_path)
#     with open(metadata_path, "w") as f:
#         json.dump(metadata, f, indent=2)

# def get_snapshot_near_seconds_ago(seconds):
#     target_time = datetime.now() - timedelta(seconds=seconds)
#     closest = None
#     closest_diff = float("inf")

#     for entry in metadata:
#         entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d_%H-%M-%S")
#         diff = abs((entry_time - target_time).total_seconds())
#         if diff < closest_diff:
#             closest_diff = diff
#             closest = entry

#     return closest

# def find_last_seen_object(object_name):
#     for entry in reversed(metadata):
#         if object_name.lower() in [obj.lower() for obj in entry["objects"]]:
#             return entry
#     return None
# clip_memory.py
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
except Exception as e:
    print(f"[⚠️] Failed to load metadata from DB: {e}")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            print("[⚠️] metadata.json is empty or corrupted. Starting fresh.")

# === Save Snapshot + Embedding ===
def save_snapshot_and_embedding(frame, detected_objects):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snap_filename = f"snap_{timestamp}.jpg"
    snap_path = os.path.join("memory/snapshots", snap_filename)
    cv2.imwrite(snap_path, frame)

    # Convert to RGB + preprocess
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    # Ensure it's 2D
    embedding_vector = embedding.astype("float32").reshape(1, -1)
    faiss_index.add(embedding_vector)

    entry = {
        "timestamp": timestamp,
        "datetime": datetime.utcnow().isoformat(),  # JSON serializable
        "snapshot": f"snapshots/{snap_filename}",
        "objects": detected_objects
    }

    metadata.append(entry)
    save_snapshot_to_db(entry)
    save_memory()  # Save index and metadata after every snapshot

    print(f"[🧠] Snapshot saved: {snap_filename}, Objects: {detected_objects}")

# === Search by Query Text ===
def search_similar_scene(query_text, top_k=1):
    if faiss_index.ntotal == 0:
        if os.path.exists(embedding_path):
            print("[ℹ️] Loading FAISS index from file...")
            faiss_index.read_index(embedding_path)
        else:
            print("[⚠️] FAISS index is empty. No results available.")
            return None

    with torch.no_grad():
        tokens = tokenizer([query_text]).to(device)
        text_features = clip_model.encode_text(tokens).cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

    D, I = faiss_index.search(text_features, top_k)
    best_index = I[0][0]

    if best_index < len(metadata):
        return metadata[best_index]
    else:
        return None

# === Save FAISS + Metadata ===
def save_memory():
    faiss.write_index(faiss_index, embedding_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

# === Get Snapshot X Seconds Ago ===
def get_snapshot_near_seconds_ago(seconds):
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
            print(f"[⚠️] Error parsing timestamp: {e}")

    return closest

# === Find Last Seen Object ===
def find_last_seen_object(object_name):
    for entry in reversed(metadata):
        if object_name.lower() in [obj.lower() for obj in entry.get("objects", [])]:
            return entry
    return None
