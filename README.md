# AI Studio Cam 🎥🧠

A real-time AI-powered memory system that sees, understands, and remembers your environment using camera, voice interface, and large language models.

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

## 🎮 Runtime Controls

| Key | Action |
|-----|--------|
| `v` | Voice input |
| `q` | Quit application |
| `s` | Switch between YOLO ↔ CNN models |
| `d` | Detailed CNN predictions (CNN mode only) |
| `c` | Change confidence threshold (CNN mode only) |

## 📁 Project Structure

```
AI Studio Cam/
├── src/
│   ├── core/           # Core modules (camera, memory, voice)
│   ├── config/         # Configuration management
│   ├── training/       # CNN training pipeline
│   ├── data/          # Data handling utilities
│   └── utils/         # General utilities
├── dataset/           # Training dataset (10 classes)
├── models/           # Trained models
├── results/          # Training results
├── memory/           # Visual memory storage
└── main.py          # Main application entry point
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

# Or run directly
python -m src.training.train_cnn_model
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

### Adding New Models
1. Create model class in `src/core/`
2. Update camera handler factory
3. Add configuration options
4. Update main application

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