"""
Camera Handler Module

Manages camera operations, object detection, and frame processing.
"""

import cv2
import time
from ultralytics import YOLO
import logging
import torch
import os

# Set environment variable to allow unsafe loading for ultralytics models
os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = 'False'

logger = logging.getLogger(__name__)

class CameraHandler:
    def __init__(self, camera_index=0, width=1280, height=720, model_path="models/yolov8n.pt"):
        """Initialize camera and YOLO model"""
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.yolo_model = None
        self.model_path = model_path
        
    def initialize(self):
        """Initialize camera and model"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Initialize YOLO model with weights_only=False for compatibility
            import torch
            # Temporarily set weights_only to False for YOLO model loading
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
            
            self.yolo_model = YOLO(self.model_path)
            
            # Restore original torch.load
            torch.load = original_load
            logger.info("Camera and YOLO model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera handler: {e}")
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
        if not self.yolo_model:
            raise RuntimeError("YOLO model not initialized")
            
        try:
            results = self.yolo_model(frame)
            annotated_frame = results[0].plot()
            
            # Extract detected object names
            detected_classes = [
                self.yolo_model.model.names[int(cls)] 
                for cls in results[0].boxes.cls
            ]
            unique_objects = list(set(detected_classes))
            
            return annotated_frame, unique_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return frame, []
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap and self.cap.isOpened()
    
    def release(self):
        """Release camera resources"""
        try:
            if self.cap:
                self.cap.release()
                logger.info("Camera released successfully")
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")
        finally:
            self.cap = None
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()