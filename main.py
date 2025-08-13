"""
AI Studio Cam - Main Application

Consolidated with model configuration functionality.
"""

import os
import cv2
import sys
import time
import threading
import json
import glob

# Fix Qt MIME database warning on Windows
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.camera_handler import CameraHandler
from core.voice_processor import VoiceProcessor
from core.memory_manager import MemoryManager

class ModelConfig:
    """Embedded configuration manager for AI Studio Cam models"""
    
    # Default configuration embedded as constant
    DEFAULT_CONFIG = {
        'active_model': 'yolo',
        'yolo_config': {
            'model_path': 'models/yolov8n.pt',
            'confidence': 0.25
        },
        'cnn_config': {
            'confidence': 0.5,
            'auto_load_latest': True
        },
        'camera_config': {
            'camera_index': 0,
            'width': 1280,
            'height': 720
        }
    }
    
    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        # Update model path with absolute path
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.config['yolo_config']['model_path'] = os.path.join(project_root, 'models', 'yolov8n.pt')
    
    def get_active_model(self):
        """Get the currently active model type"""
        return self.config['active_model']
    
    def set_active_model(self, model_type):
        """Set the active model type"""
        if model_type in ['yolo', 'cnn']:
            self.config['active_model'] = model_type
        else:
            raise ValueError("Model type must be 'yolo' or 'cnn'")
    
    def get_latest_cnn_model(self):
        """Get the path to the latest trained CNN model"""
        project_root = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(project_root, 'models')
        
        model_files = glob.glob(os.path.join(models_dir, 'custom_cnn_*.pth'))
        if not model_files:
            return None
        
        return max(model_files, key=os.path.getctime)
    
    def get_yolo_config(self):
        """Get YOLO configuration"""
        return self.config['yolo_config']
    
    def get_cnn_config(self):
        """Get CNN configuration"""
        return self.config['cnn_config']
    
    def get_camera_config(self):
        """Get camera configuration"""
        return self.config['camera_config']


class AIStudioCam:

    def __init__(self, use_cnn=False):
        self.use_cnn = use_cnn
        self.camera_handler = None
        self.voice_processor = VoiceProcessor()
        self.memory_manager = MemoryManager()
        self.listening = False
        self.current_objects = []

    def initialize(self):
        """Initialize camera and voice systems"""
        # Get configuration
        config = ModelConfig()
        
        if self.use_cnn:
            # Check if CNN model is available
            latest_model = config.get_latest_cnn_model()
            if not latest_model:
                raise RuntimeError("No CNN models found. Please train a model first.")
            
            cnn_config = config.get_cnn_config()
            self.camera_handler = CameraHandler(
                model_type="cnn",
                confidence_threshold=cnn_config['confidence']
            )
            print("Using CNN model")
        else:
            yolo_config = config.get_yolo_config()
            self.camera_handler = CameraHandler(
                model_type="yolo",
                model_path=yolo_config['model_path']
            )
            print("Using YOLO model")

        self.camera_handler.initialize()
        self.voice_processor.speak("AI Studio Cam initialized. Press 'v' to talk to me, or 'q' to quit.")

    def handle_voice_command(self):
        """Handle voice input in a separate thread"""
        try:
            query = self.voice_processor.listen()
            if query:
                response = self.voice_processor.process_query(
                    query, self.memory_manager, self.current_objects
                )
                self.voice_processor.speak(response)
        except Exception as e:
            print(f"Voice command error: {e}")
        finally:
            self.listening = False

    def run(self):
        """Main application loop with speech integration"""
        print("\n=== AI Studio Cam with Voice Integration ===")
        print("🎮 Runtime Controls:")
        print("  'v' - Voice command (hands-free interaction)")
        print("  's' - Switch between YOLO ↔ CNN models")
        print("  'd' - Detailed CNN predictions (CNN mode only)")
        print("  'c' - Change confidence threshold (CNN mode only)")
        print("  'm' - Memory statistics and usage")
        print("  'r' - Recent snapshots from last few minutes")
        print("  'f' - Find when object was last seen")
        print("  't' - Take manual snapshot (bypasses interval)")
        print("  'h' - Show this help menu")
        print("  'q' - Quit application safely")
        print("\n🎤 Voice Commands:")
        print("  'What do you see?' - Current view analysis")
        print("  'When did you last see [object]?' - Object recall")
        print("  'What did you see [time] ago?' - Time-based recall")
        print("  'Show me something similar to [description]' - Semantic search")
        print("  'Take a snapshot' - Manual capture")
        print("  'Switch to [model]' - Model switching")
        print("  'Memory status' - System information")
        print("  'Clear memory' - Reset memory")
        print("  'Search for [object]' - Object search")
        print("=" * 60)

        while self.camera_handler.is_opened():
            current_time = time.time()
            frame = self.camera_handler.capture_frame()
            annotated_frame, detected_objects = self.camera_handler.detect_objects(frame)

            # Update current objects
            self.current_objects = detected_objects

            # Auto-save snapshots when memory manager determines it's time
            if self.memory_manager.should_take_snapshot(current_time):
                try:
                    self.memory_manager.save_snapshot(frame, detected_objects, current_time)
                except Exception as e:
                    print(f"Snapshot save error: {e}")

            # Add status overlay
            status_text = "Listening..." if self.listening else "Press 'v' to speak"
            cv2.putText(annotated_frame, status_text, (10, annotated_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("AI Studio Cam", annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v') and not self.listening:
                self.listening = True
                # Start voice processing in separate thread
                voice_thread = threading.Thread(target=self.handle_voice_command)
                voice_thread.daemon = True
                voice_thread.start()
            elif key == ord('s'):
                # Switch between YOLO and CNN models
                try:
                    if self.use_cnn:
                        print("Switching to YOLO model...")
                        self.use_cnn = False
                        self.camera_handler.release()
                        config = ModelConfig()
                        yolo_config = config.get_yolo_config()
                        self.camera_handler = CameraHandler(
                            model_type="yolo",
                            model_path=yolo_config['model_path']
                        )
                        self.camera_handler.initialize()
                        print("Switched to YOLO model")
                    else:
                        print("Switching to CNN model...")
                        config = ModelConfig()
                        latest_model = config.get_latest_cnn_model()
                        if not latest_model:
                            print("No CNN models found. Please train a model first.")
                            continue
                        self.use_cnn = True
                        self.camera_handler.release()
                        cnn_config = config.get_cnn_config()
                        self.camera_handler = CameraHandler(
                            model_type="cnn",
                            confidence_threshold=cnn_config['confidence']
                        )
                        self.camera_handler.initialize()
                        print("Switched to CNN model")
                except Exception as e:
                    print(f"Model switch error: {e}")
            elif key == ord('d') and self.use_cnn:
                # Detailed CNN predictions (CNN mode only)
                try:
                    if detected_objects:
                        print("\n=== Detailed CNN Predictions ===")
                        for obj in detected_objects:
                            print(f"Object: {obj.get('class', 'Unknown')}")
                            print(f"Confidence: {obj.get('confidence', 0):.3f}")
                            print(f"Location: {obj.get('bbox', 'Unknown')}")
                            print("---")
                    else:
                        print("No objects detected in current frame")
                except Exception as e:
                    print(f"Detailed predictions error: {e}")
            elif key == ord('c') and self.use_cnn:
                # Change confidence threshold (CNN mode only)
                try:
                    print(f"Current confidence threshold: {self.camera_handler.confidence_threshold}")
                    new_threshold = input("Enter new confidence threshold (0.1-1.0): ")
                    try:
                        new_threshold = float(new_threshold)
                        if 0.1 <= new_threshold <= 1.0:
                            self.camera_handler.confidence_threshold = new_threshold
                            print(f"Confidence threshold updated to {new_threshold}")
                        else:
                            print("Threshold must be between 0.1 and 1.0")
                    except ValueError:
                        print("Invalid threshold value")
                except Exception as e:
                    print(f"Confidence change error: {e}")
            elif key == ord('m'):
                # Memory statistics and usage
                try:
                    stats = self.memory_manager.get_memory_stats()
                    print("\n=== Memory Statistics ===")
                    print(f"Total snapshots: {stats.get('total_snapshots', 0)}")
                    print(f"Memory usage: {stats.get('memory_usage', 'Unknown')}")
                    print(f"Last snapshot: {stats.get('last_snapshot', 'Never')}")
                    print("=" * 30)
                except Exception as e:
                    print(f"Memory stats error: {e}")
            elif key == ord('r'):
                # Recent snapshots from last few minutes
                try:
                    recent_snapshots = self.memory_manager.get_recent_snapshots(minutes=5)
                    if recent_snapshots:
                        print(f"\n=== Recent Snapshots (Last 5 minutes) ===")
                        for snapshot in recent_snapshots:
                            print(f"Time: {snapshot.get('timestamp', 'Unknown')}")
                            print(f"Objects: {snapshot.get('objects', [])}")
                            print("---")
                    else:
                        print("No recent snapshots found")
                except Exception as e:
                    print(f"Recent snapshots error: {e}")
            elif key == ord('f'):
                # Find when object was last seen
                try:
                    search_object = input("Enter object to search for: ").strip()
                    if search_object:
                        last_seen = self.memory_manager.find_object_last_seen(search_object)
                        if last_seen:
                            print(f"'{search_object}' was last seen at: {last_seen}")
                        else:
                            print(f"'{search_object}' not found in memory")
                except Exception as e:
                    print(f"Object search error: {e}")
            elif key == ord('t'):
                # Take manual snapshot (bypasses interval)
                try:
                    self.memory_manager.save_snapshot(frame, detected_objects, current_time)
                    print("Manual snapshot saved!")
                except Exception as e:
                    print(f"Manual snapshot error: {e}")
            elif key == ord('h'):
                # Show help menu
                print("\n=== AI Studio Cam with Voice Integration ===")
                print("🎮 Runtime Controls:")
                print("  'v' - Voice command (hands-free interaction)")
                print("  's' - Switch between YOLO ↔ CNN models")
                print("  'd' - Detailed CNN predictions (CNN mode only)")
                print("  'c' - Change confidence threshold (CNN mode only)")
                print("  'm' - Memory statistics and usage")
                print("  'r' - Recent snapshots from last few minutes")
                print("  'f' - Find when object was last seen")
                print("  't' - Take manual snapshot (bypasses interval)")
                print("  'h' - Show this help menu")
                print("  'q' - Quit application safely")
                print("\n🎤 Voice Commands:")
                print("  'What do you see?' - Current view analysis")
                print("  'When did you last see [object]?' - Object recall")
                print("  'What did you see [time] ago?' - Time-based recall")
                print("  'Show me something similar to [description]' - Semantic search")
                print("  'Take a snapshot' - Manual capture")
                print("  'Switch to [model]' - Model switching")
                print("  'Memory status' - System information")
                print("  'Clear memory' - Reset memory")
                print("  'Search for [object]' - Object search")
                print("=" * 60)

        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.camera_handler:
                self.camera_handler.release()
            self.voice_processor.cleanup()
            self.memory_manager.save_all_memory()
            cv2.destroyAllWindows()
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Cleanup error: {e}")


def main():
    """Main application function"""

    # Interactive model selection
    print("AI Studio Cam with Voice Integration")
    print("=" * 36)
    print("1. YOLO (default)")
    print("2. CNN")

    choice = input("Enter choice (1 or 2): ").strip()
    model_type = 'cnn' if choice == '2' else 'yolo'

    print(f"Selected: {model_type.upper()} with Voice Integration")
    print()

    # Check if CNN model is available
    if model_type == 'cnn':
        config = ModelConfig()
        try:
            latest_model = config.get_latest_cnn_model()
            if not latest_model:
                print("No CNN models found!")
                print("Would you like to train a CNN model now?")
                train_choice = input("Train CNN model? (y/n): ").strip().lower()

                if train_choice in ['y', 'yes']:
                    print("Starting CNN training...")
                    import subprocess

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
        except Exception as e:
            print(f"Model check error: {e}")

    # Start the application
    print(f"Starting AI Studio Cam with {model_type.upper()} model and Voice Integration...")

    use_cnn = model_type == 'cnn'

    try:
        app = AIStudioCam(use_cnn=use_cnn)
        app.initialize()
        app.run()
        return 0
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Application error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
