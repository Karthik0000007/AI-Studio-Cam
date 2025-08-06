"""
Model Configuration Module

Easy configuration for switching between YOLO and CNN models.
"""

import os
import json
import glob
from datetime import datetime

class ModelConfig:
    """Configuration manager for AI Studio Cam models"""
    
    def __init__(self, config_file=None):
        if config_file is None:
            # Default to config folder
            config_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_file = os.path.join(config_dir, 'model_config.json')
        else:
            self.config_file = config_file
        # Get the project root directory (parent of config folder)
        config_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(config_dir)
        
        self.default_config = {
            'active_model': 'yolo',
            'yolo_config': {
                'model_path': os.path.join(project_root, 'models', 'yolov8n.pt'),
                'confidence': 0.25
            },
            'cnn_config': {
                'model_path': None, 
                'confidence': 0.5,
                'auto_load_latest': True
            },
            'camera_config': {
                'camera_index': 0,
                'width': 1280,
                'height': 720
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                return self._merge_configs(self.default_config, config)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.default_config.copy()
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _merge_configs(self, default, user):
        """Recursively merge user config with defaults"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_active_model(self):
        """Get the currently active model type"""
        return self.config['active_model']
    
    def set_active_model(self, model_type):
        """Set the active model type"""
        if model_type in ['yolo', 'cnn']:
            self.config['active_model'] = model_type
            self.save_config()
        else:
            raise ValueError("Model type must be 'yolo' or 'cnn'")
    
    def get_latest_cnn_model(self):
        """Get the path to the latest trained CNN model"""
        # Get the project root directory (parent of config folder)
        config_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(config_dir)
        models_dir = os.path.join(project_root, 'models')
        
        # Look for both naming patterns: custom_cnn_*.pth and custom_cnn_model_*.pth
        model_files = glob.glob(os.path.join(models_dir, 'custom_cnn_*.pth'))
        if model_files:
            return max(model_files, key=os.path.getctime)
        return None

if __name__ == "__main__":
    config = ModelConfig()
    print(f"Active model: {config.get_active_model()}")