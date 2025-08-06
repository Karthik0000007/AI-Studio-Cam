"""
Simple test script to check if basic imports work
"""

import sys
import os

def test_imports():
    print("Testing imports...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        from core.camera_handler_cnn import CameraHandlerCNN
        print("✅ CameraHandlerCNN import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("Simple Structure Test")
    print("=" * 30)
    
    if test_imports():
        print("✅ Basic imports working!")
    else:
        print("❌ Import test failed")
    
    return 0

if __name__ == "__main__":
    exit(main())