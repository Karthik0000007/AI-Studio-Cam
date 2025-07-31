from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import os

# === MongoDB Atlas URI ===
uri = "mongodb+srv://sairamkarthikmalladi:<b413Sf103OeUIouI>@cluster0.zntg1pw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# === Connect to MongoDB ===
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print("❌ Connection error:", e)

# === DB & Collection Setup ===
DB_NAME = "Ai_Studio_Cam"
COLLECTION_NAME = "snapshots"

db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# === DB Functions ===
def save_snapshot_to_db(metadata_entry):
    collection.insert_one(metadata_entry)

def load_all_metadata():
    return list(collection.find({}, {"_id": 0}))

def query_by_object(object_name):
    return list(collection.find({"objects": object_name}, {"_id": 0}))

def query_recent(seconds=30):
    now = datetime.utcnow()
    threshold = now - timedelta(seconds=seconds)
    return list(collection.find({"datetime": {"$gte": threshold}}, {"_id": 0}))
