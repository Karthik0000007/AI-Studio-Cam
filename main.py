"""
AI Studio Cam - Main Application
"""

import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.camera_handler import CameraHandler
from core.camera_handler_cnn import CameraHandlerCNN
from config.model_config import ModelConfig

class AIStudioCam:
    def __init__(self, use_cnn=False):
        self.use_cnn = use_cnn
        self.camera_handler = None
        
    def initialize(self):
        if self.use_cnn:
            self.camera_handler = CameraHandlerCNN()
            print("Using CNN model")
        else:
            self.camera_handler = CameraHandler()
            print("Using YOLO model")
        
        self.camera_handler.initialize()
    
    def run(self):
        print("AI Studio Cam - Press 'q' to quit")
        
        while self.camera_handler.is_opened():
            frame = self.camera_handler.capture_frame()
            annotated_frame, detected_objects = self.camera_handler.detect_objects(frame)
            
            cv2.imshow("AI Studio Cam", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        if self.camera_handler:
            self.camera_handler.release()
        cv2.destroyAllWindows()

def main():
    """Main application function"""
    
    # Load configuration
    config = ModelConfig()
    
    # Interactive model selection
    print("AI Studio Cam - Model Selection")
    print("=" * 30)
    print("1. YOLO (default)")
    print("2. CNN")
    
    choice = input("Enter choice (1 or 2): ").strip()
    model_type = 'cnn' if choice == '2' else 'yolo'
    
    print(f"Selected: {model_type.upper()}")
    print()
    
    # Check if CNN model is available
    if model_type == 'cnn':
        latest_model = config.get_latest_cnn_model()
        if not latest_model:
            print("No CNN models found!")
            print("Would you like to train a CNN model now?")
            train_choice = input("Train CNN model? (y/n): ").strip().lower()
            
            if train_choice in ['y', 'yes']:
                print("Starting CNN training...")
                # Import and run the training script
                import subprocess
                import sys
                import os
                
                training_script = os.path.join(os.path.dirname(__file__), 'training', 'train_cnn_model.py')
                if os.path.exists(training_script):
                    try:
                        subprocess.run([sys.executable, training_script], check=True)
                        print("Training completed! Please run the application again.")
                    except subprocess.CalledProcessError:
                        print("Training failed. Please check the training script.")
                    return 0
                else:
                    print("Training script not found!")
                    return 1
            else:
                print("Please train a CNN model first or use YOLO instead.")
                return 1
    
    # Start the application
    print(f"Starting AI Studio Cam with {model_type.upper()} model...")
    use_cnn = model_type == 'cnn'
    
    app = AIStudioCam(use_cnn=use_cnn)
    app.initialize()
    app.run()
    
    return 0

if __name__ == "__main__":
    exit(main())