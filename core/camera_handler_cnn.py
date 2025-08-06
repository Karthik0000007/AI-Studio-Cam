"""Simple CNN Camera Handler"""

import cv2, torch, torch.nn as nn, os, glob
from torchvision import transforms
from PIL import Image

class CustomCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2, num_filters=16):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU(True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(num_filters, num_filters*2, 3, padding=1), nn.BatchNorm2d(num_filters*2), nn.ReLU(True),
            nn.Conv2d(num_filters*2, num_filters*2, 3, padding=1), nn.BatchNorm2d(num_filters*2), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(num_filters*2, num_filters*4, 3, padding=1), nn.BatchNorm2d(num_filters*4), nn.ReLU(True),
            nn.Conv2d(num_filters*4, num_filters*4, 3, padding=1), nn.BatchNorm2d(num_filters*4), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(num_filters*4, num_filters*8, 3, padding=1), nn.BatchNorm2d(num_filters*8), nn.ReLU(True),
            nn.Conv2d(num_filters*8, num_filters*8, 3, padding=1), nn.BatchNorm2d(num_filters*8), nn.ReLU(True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(num_filters*8, num_filters*4), nn.ReLU(True),
            nn.Dropout(dropout_rate), nn.Linear(num_filters*4, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

class CNNObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        
        self._load_model()
    
    def _find_latest_model(self):
        # Get the project root directory (parent of core folder)
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
        
        # Extract hyperparameters from checkpoint
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
        
        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Model architecture: dropout_rate={dropout_rate}, num_filters={num_filters}")
    
    def predict(self, image):
        try:
            # Validate input image
            if image is None or image.size == 0:
                return None, 0.0
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image with error handling
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                confidence_score = confidence.item()
                if confidence_score >= self.confidence_threshold:
                    return self.class_names[predicted_class.item()], confidence_score
                return None, confidence_score
                
        except Exception as e:
            print(f"Error in CNN prediction: {e}")
            return None, 0.0

class CameraHandlerCNN:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.cnn_detector = None
    
    def initialize(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.cnn_detector = CNNObjectDetector()
        print("Camera and CNN model initialized")
    
    def capture_frame(self):
        success, frame = self.cap.read()
        if not success or frame is None:
            raise RuntimeError("Failed to capture frame")
        
        # Validate frame dimensions
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise RuntimeError("Invalid frame dimensions")
        
        return frame
    
    def detect_objects(self, frame):
        try:
            predicted_label, confidence = self.cnn_detector.predict(frame)
            annotated_frame = frame.copy()
            detected_objects = []
            
            if predicted_label:
                detected_objects.append(predicted_label)
                text = f"{predicted_label}: {confidence:.2f}"
                cv2.putText(annotated_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, f"No detection: {confidence:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return annotated_frame, detected_objects
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            # Return original frame with error message
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Detection Error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated_frame, []
    
    def is_opened(self):
        return self.cap and self.cap.isOpened()
    
    def release(self):
        if self.cap:
            self.cap.release()
        self.cap = None

if __name__ == "__main__":
    camera_handler = CameraHandlerCNN()
    camera_handler.initialize()
    
    while True:
        frame = camera_handler.capture_frame()
        annotated_frame, detected_objects = camera_handler.detect_objects(frame)
        cv2.imshow("CNN Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera_handler.release()
    cv2.destroyAllWindows()