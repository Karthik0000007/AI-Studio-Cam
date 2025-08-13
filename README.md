# AI Studio Cam 🎥🧠

A real-time AI-powered memory system that sees, understands, and remembers your environment using computer vision, voice interface, and large language models.

## ✨ Features

- **🎯 Dual Model Support**: Switch between YOLO (general objects) and Custom CNN (trained on your data)
- **🧠 Visual Memory**: CLIP-based embeddings with FAISS search for visual memory
- **🎤 Voice Interface**: Speech recognition and text-to-speech for natural interaction
- **📊 Real-time Detection**: Live camera feed with object detection and classification
- **💾 Persistent Memory**: Local storage with optional MongoDB cloud backup
- **🔧 Easy Configuration**: Interactive setup and runtime model switching
- **📈 Training Pipeline**: Complete CNN training with hyperparameter tuning

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-studio-cam.git
cd ai-studio-cam

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Basic Usage
```bash
# Start with default YOLO model
python main.py

# Train a custom CNN model
python main.py --train

# Configure models interactively
python main.py --config

# Start with CNN model
python main.py --model cnn
```

## �� Runtime Controls

### Keyboard Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| `v` | Voice input | Activate voice recognition for hands-free interaction |
| `q` | Quit application | Safely exit the application and save memory |
| `s` | Switch models | Toggle between YOLO ↔ CNN models during runtime |
| `d` | Detailed predictions | Show detailed CNN predictions with confidence scores (CNN mode only) |
| `c` | Change confidence | Adjust confidence threshold for CNN detections (CNN mode only) |
| `m` | Memory stats | Display current memory statistics and usage |
| `r` | Recent snapshots | Show recent snapshots from the last few minutes |
| `f` | Find object | Search for when a specific object was last seen |
| `t` | Take snapshot | Manually capture a snapshot (bypasses interval) |
| `h` | Help | Display this help menu and available commands |

### Voice Commands

| Command | Action | Example |
|---------|--------|---------|
| **"What do you see?"** | Current view analysis | Get description of current camera view |
| **"When did you last see [object]?"** | Object recall | "When did you last see a phone?" |
| **"What did you see [time] ago?"** | Time-based recall | "What did you see 30 seconds ago?" |
| **"Show me something similar to [description]"** | Semantic search | "Show me something similar to a red cup" |
| **"Take a snapshot"** | Manual capture | Capture current frame immediately |
| **"Switch to [model]"** | Model switching | "Switch to CNN" or "Switch to YOLO" |
| **"Memory status"** | System info | Get current memory and system status |
| **"Clear memory"** | Reset memory | Clear all stored snapshots and embeddings |
| **"Search for [object]"** | Object search | "Search for all images with a laptop" |

### Interactive Features

#### Model Switching (`s`)
- **YOLO Mode**: General object detection with 80+ COCO classes
- **CNN Mode**: Specialized classification with custom-trained classes
- **Real-time**: Switch without restarting the application
- **Preserves**: Current camera settings and memory state

#### Confidence Adjustment (`c`)
- **Range**: 0.1 to 0.9 (10% to 90%)
- **Default**: 0.5 (50%)
- **Effect**: Higher = fewer but more confident detections
- **Real-time**: Changes apply immediately to next frame

#### Detailed Predictions (`d`)
- **CNN Mode Only**: Shows top-3 predictions with confidence scores
- **Visual Overlay**: Displays predictions on camera feed
- **Real-time**: Updates with each frame
- **Format**: "Class: Confidence%" (e.g., "laptop: 87.3%")

#### Memory Management
- **Automatic Snapshots**: Every 15 seconds (configurable)
- **Manual Snapshots**: Press `t` for immediate capture
- **Memory Stats**: Press `m` for usage information
- **Object Search**: Press `f` to find specific objects
- **Recent View**: Press `r` for recent snapshots

### Camera Controls

| Action | Description | Default |
|--------|-------------|---------|
| **Resolution** | Camera resolution | 1280x720 |
| **FPS** | Frame rate | Camera default |
| **Index** | Camera device | 0 (first camera) |
| **Auto-focus** | Focus adjustment | Camera default |

### Memory System Controls

| Feature | Description | Default |
|---------|-------------|---------|
| **Snapshot Interval** | Time between auto-snapshots | 15 seconds |
| **Storage Location** | Local memory folder | `memory/` |
| **Cloud Backup** | MongoDB Atlas integration | Optional |
| **Embedding Model** | CLIP ViT-B-32 | OpenAI pretrained |
| **Search Engine** | FAISS vector search | L2 distance |

### Performance Controls

| Setting | Description | Impact |
|---------|-------------|--------|
| **Confidence Threshold** | Detection sensitivity | Speed vs. Accuracy |
| **Camera Resolution** | Frame size | Processing speed |
| **Model Type** | YOLO vs. CNN | Detection vs. Classification |
| **Memory Cleanup** | Periodic cleanup | Storage vs. Performance |

### Emergency Controls

| Action | Description | Use Case |
|---------|-------------|----------|
| **Force Quit** | `Ctrl+C` in terminal | Application unresponsive |
| **Memory Reset** | Voice command "clear memory" | Start fresh |
| **Model Reload** | Restart application | Model corruption |
| **Camera Reset** | Restart application | Camera issues |

## 📁 Project Structure

```
AI Studio Cam/
├── core/           # Core modules (camera, memory, voice)
├── config/         # Configuration management
├── training/       # CNN training pipeline
├── data/          # Data handling utilities
├── models/        # Trained models
├── memory/        # Visual memory storage
├── tests/         # Test files
└── main.py        # Main application entry point
```

## 🤖 Model Types

### YOLO Model
- **Purpose**: General object detection
- **Classes**: 80+ COCO classes (person, car, phone, etc.)
- **Speed**: ~45-70ms per frame
- **Use Case**: General-purpose object detection

### Custom CNN Model
- **Purpose**: Specialized classification
- **Classes**: 10 custom classes (book, bottle, chair, cup, keyboard, laptop, mouse, pen, phone, remote)
- **Training**: Automated with hyperparameter tuning
- **Use Case**: High accuracy on specific objects

## 🎯 Training Your CNN

```bash
# Start training with hyperparameter optimization
python main.py --train
```

### Training Features
- **Automated HP Tuning**: Uses Optuna for optimization
- **Data Augmentation**: Rotation, flip, color jitter, etc.
- **Multiple Architectures**: Custom CNN and ResNet-based
- **Comprehensive Evaluation**: Accuracy, confusion matrix, training plots
- **Model Persistence**: Saves best models with metadata

## 🔧 Configuration

### Interactive Configuration
```bash
python main.py --config
```

### Manual Configuration
Edit `config/model_config.json`:
```json
{
  "active_model": "yolo",
  "yolo_config": {
    "model_path": "models/yolov8n.pt",
    "confidence": 0.25
  },
  "cnn_config": {
    "model_path": null,
    "confidence": 0.5,
    "auto_load_latest": true
  }
}
```

## 🎤 Voice Commands

- **"What do you see?"** - Current view
- **"When did you last see a [object]?"** - Object recall
- **"What did you see 30 seconds ago?"** - Time-based recall
- **"Show me something similar to [description]"** - Semantic search

## 📊 Memory System

- **Visual Embeddings**: CLIP-based image embeddings
- **Vector Search**: FAISS for fast similarity search
- **Persistent Storage**: Local JSON + FAISS index
- **Cloud Backup**: Optional MongoDB Atlas integration
- **Automatic Snapshots**: Every 15 seconds (configurable)

## 🛠️ Development

### Project Installation
```bash
# Development installation
pip install -e .

# With development dependencies
pip install -e .[dev]

# With GPU support
pip install -e .[gpu]
```

## 📋 Requirements

- **Python**: 3.8+
- **GPU**: Optional (CUDA support)
- **Camera**: Webcam or USB camera
- **Microphone**: For voice input
- **Storage**: ~2GB for models and memory

## 🔍 Troubleshooting

### Common Issues
1. **Camera not found**: Check camera index in config
2. **CUDA errors**: Install appropriate PyTorch version
3. **Audio issues**: Install PyAudio dependencies
4. **MongoDB errors**: Disable in config or fix connection

### Performance Tips
- Use GPU for faster inference
- Adjust confidence thresholds
- Reduce camera resolution if needed
- Clear memory folder periodically

## 📈 Performance

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLO | 45-70ms | General | Broad object detection |
| Custom CNN | 30-50ms | High | Specific object classification |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLO**: Ultralytics YOLOv8
- **CLIP**: OpenAI CLIP model
- **FAISS**: Facebook AI Similarity Search
- **Whisper**: OpenAI Whisper for speech recognition