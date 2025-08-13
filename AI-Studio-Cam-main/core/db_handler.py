from datetime import datetime, timedelta
import os

# === MongoDB Setup (Optional) ===
MONGODB_ENABLED = False
client = None
db = None
collection = None

try:
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
    
    # MongoDB Atlas URI
    uri = "mongodb+srv://sairamkarthikmalladi:<b413Sf103OeUIouI>@cluster0.zntg1pw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    # Try to connect with a short timeout
    client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    
    # If we get here, connection succeeded
    DB_NAME = "Ai_Studio_Cam"
    COLLECTION_NAME = "snapshots"
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    MONGODB_ENABLED = True
    print("✅ Connected to MongoDB Atlas")
    
except Exception as e:
    print(f"⚠️  MongoDB not available: {e}")
    print("📁 Using local storage only")

# === DB Functions (MongoDB Optional) ===
def save_snapshot_to_db(metadata_entry):
    """Save snapshot to MongoDB if available, otherwise skip silently"""
    if not MONGODB_ENABLED:
        return  # Skip silently if MongoDB not available
    
    try:
        result = collection.insert_one(metadata_entry)
        print(f"[DB] Snapshot saved with ID: {result.inserted_id}")
    except Exception as e:
        print(f"[ERROR] Failed to save to database: {e}")
        # Don't raise - just log the error

def load_all_metadata():
    """Load metadata from MongoDB if available, otherwise return empty list"""
    if not MONGODB_ENABLED:
        return []
    
    try:
        return list(collection.find({}, {"_id": 0}))
    except Exception as e:
        print(f"[ERROR] Failed to load metadata from database: {e}")
        return []

def query_by_object(object_name):
    """Query by object from MongoDB if available, otherwise return empty list"""
    if not MONGODB_ENABLED:
        return []
    
    try:
        return list(collection.find({"objects": object_name}, {"_id": 0}))
    except Exception as e:
        print(f"[ERROR] Failed to query by object '{object_name}': {e}")
        return []

def query_recent(seconds=30):
    """Query recent snapshots from MongoDB if available, otherwise return empty list"""
    if not MONGODB_ENABLED:
        return []
    
    try:
        now = datetime.utcnow()
        threshold = now - timedelta(seconds=seconds)
        return list(collection.find({"datetime": {"$gte": threshold}}, {"_id": 0}))
    except Exception as e:
        print(f"[ERROR] Failed to query recent snapshots: {e}")
        return []