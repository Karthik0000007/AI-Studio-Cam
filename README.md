# 🧠 AI Studio Cam

A real-time AI-powered memory system that sees, understands, and remembers your environment using a camera, voice interface, and large language models. Inspired by J.A.R.V.I.S. and SAO: Ordinal Scale.

---

## 📦 Features

- 🎯 YOLOv8-based object detection pipeline using Ultralytics
- 📸 Automatic snapshot saving logic via OpenCV
- 🧠 Visual memory embedding using CLIP (OpenAI ViT-B/32)
- 📝 Object and context logging into `metadata.json`
- 🧠 FAISS vector index for storing and querying object memories
- 🔁 Combined pipeline: detect → snapshot → embed → log → index
- 🔍 Visual memory querying like “What did you see X seconds ago?”
- 🎤 Microphone integration using Whisper (OpenAI or local)
- 🗣️ Voice output using edge-tts (or pyttsx3)
- 🌐 Flask/FastAPI interface for natural interaction
- 🧠 Store and search richer CLIP-based semantic embeddings
- ⌛ Long-term memory recall support with FAISS + timestamps
- 🤖 Demo: “What did you see 5 mins ago?” with a verbal response

---

## 🚧 In Progress

- Real-time conversation + memory updates
- Contextual filtering and conversation history
- Interface UX improvement and memory visualization
- Offline Whisper + GPT model support (for privacy-first mode)

---

## 🛠️ Tech Stack

- **Computer Vision**: YOLOv8, OpenCV
- **Embeddings**: OpenAI CLIP (ViT-B/32)
- **Vector Search**: FAISS
- **Speech Recognition**: OpenAI Whisper / faster-whisper
- **TTS**: edge-tts (Microsoft) / pyttsx3
- **Language Model**: OpenAI GPT-4 API
- **Web Framework**: Flask / FastAPI (planned)

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/ai-studio-cam.git
cd ai-studio-cam
pip install -r requirements.txt
python main.py
```
```bash
ai-studio-cam/
├── main.py                   # Main pipeline entry
├── db_handler.py             # Handles connecting to a MongoDB Atlas database
├── clip_memory.py            # manages saving, searching, and retrieving image snapshots
├── memory.py                 # Metadata logging
├── testing.py                # Tests the connection to a MongoDB Atlas database
├── voice_input.py            # Records audio from the microphone, saves it
├── tts.py                    # uses the pyttsx3 library to convert text to speech
└── requirements.txt
```