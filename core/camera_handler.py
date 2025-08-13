"""
Camera Handler Module - Simplified version
Manages camera operations, object detection, and frame processing.
Supports both YOLO and custom CNN models with unified interface.
"""

import cv2
import torch
import os
import glob
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

# Set environment variable to allow unsafe loading for ultralytics models
os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = 'False'

logger = logging.getLogger(__name__)

# Import CustomCNN from training module
try:
    from training.train_cnn_model import CustomCNN
except ImportError as e:
    logger.error(f"CNN model import error: {e}")

class CNNObjectDetector:
    """CNN-based object detector for custom classification"""
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        self.transform = None
        self._load_model()
    
    def _find_latest_model(self):
        core_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(core_dir)
        models_dir = os.path.join(project_root, 'models')
        
        model_files = glob.glob(os.path.join(models_dir, 'custom_cnn_*.pth'))
        if not model_files:
            raise FileNotFoundError("No trained models found. Please train a model first.")
        return max(model_files, key=os.path.getctime)
    
    def _load_model(self):
        model_path = self._find_latest_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        num_classes = len(self.class_names)
        
        hyperparams = checkpoint.get('hyperparameters', {})
        dropout_rate = hyperparams.get('dropout_rate', 0.2)
        num_filters = hyperparams.get('num_filters', 16)
        
        self.model = CustomCNN(num_classes, dropout_rate, num_filters).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"CNN Model loaded: {model_path}")
        logger.info(f"Classes: {self.class_names}")
    
    def predict(self, image):
        if image is None or image.size == 0:
            return None, 0.0
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_class = predicted_class.item()
                
                if confidence >= self.confidence_threshold:
                    class_name = self.class_names[predicted_class]
                    return class_name, confidence
                else:
                    return None, confidence
                    
        except Exception as e:
            logger.error(f"CNN prediction error: {e}")
            return None, 0.0

class CameraHandler:
    """Unified camera handler supporting both YOLO and CNN models"""

    def __init__(self, camera_index=0, width=1280, height=720, model_type="yolo", model_path="models/yolov8n.pt", confidence_threshold=0.5):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.model_type = model_type
        self.cap = None
        self.yolo_model = None
        self.cnn_model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
    def initialize(self):
        """Initialize camera and selected model"""
        cv2.setLogLevel(0)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Initialize selected model
        if self.model_type == "yolo":
            self._init_yolo()
        elif self.model_type == "cnn":
            self._init_cnn()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Camera and {self.model_type.upper()} model initialized successfully")
    
    def _init_yolo(self):
        """Initialize YOLO model"""
        try:
            from ultralytics import YOLO
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
            
            self.yolo_model = YOLO(self.model_path)
            torch.load = original_load
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise
    
    def _init_cnn(self):
        """Initialize CNN model"""
        try:
            self.cnn_model = CNNObjectDetector(confidence_threshold=self.confidence_threshold)
        except Exception as e:
            logger.error(f"Failed to initialize CNN model: {e}")
            raise
    
    def capture_frame(self):
        """Capture a single frame from camera"""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized or not opened")
            
        success, frame = self.cap.read()
        if not success:
            raise RuntimeError("Failed to capture frame")
            
        return frame
    
    def detect_objects(self, frame):
        """Detect objects in frame and return results"""
        if self.model_type == "yolo":
            return self._detect_yolo(frame)
        elif self.model_type == "cnn":
            return self._detect_cnn(frame)
        else:
            raise RuntimeError(f"Unknown model type: {self.model_type}")
    
    def _detect_yolo(self, frame):
        """Detect objects using YOLO model"""
        if not self.yolo_model:
            raise RuntimeError("YOLO model not initialized")
            
        try:
            results = self.yolo_model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            detected_classes = [
                self.yolo_model.model.names[int(cls)] 
                for cls in results[0].boxes.cls
            ]
            unique_objects = list(set(detected_classes))
            
            return annotated_frame, unique_objects
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return frame, []
    
    def _detect_cnn(self, frame):
        """Detect objects using CNN model"""
        if not self.cnn_model:
            raise RuntimeError("CNN model not initialized")
            
        try:
            class_name, confidence = self.cnn_model.predict(frame)
            
            annotated_frame = frame.copy()
            if class_name:
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return annotated_frame, [class_name]
            else:
                cv2.putText(annotated_frame, "No confident detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return annotated_frame, []
                
        except Exception as e:
            logger.error(f"CNN detection failed: {e}")
            return frame, []
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            logger.info("Camera released")
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold (CNN mode only)"""
        if self.model_type == "cnn" and self.cnn_model:
            self.cnn_model.confidence_threshold = threshold
            logger.info(f"CNN confidence threshold set to {threshold}")
        else:
            logger.warning("Confidence threshold can only be set in CNN mode")
    
    def get_model_info(self):
        """Get information about the current model"""
        if self.model_type == "yolo":
            return {
                "type": "YOLO",
                "model_path": self.model_path,
                "classes": "80+ COCO classes" if self.yolo_model else "Not loaded"
            }
        elif self.model_type == "cnn":
            return {
                "type": "CNN",
                "classes": self.cnn_model.class_names if self.cnn_model else "Not loaded",
                "confidence_threshold": self.confidence_threshold
            }
        else:
            return {"type": "Unknown", "status": "Not initialized"}
