import torch
import cv2
from src.models.yolo import YOLOv8

def detect_objects(image_path, model_path):
    model = YOLOv8()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    image = cv2.imread(image_path)
    # Preprocess image
    # Detect objects
    outputs = model(image)
    # Postprocess outputs
    return outputs

if __name__ == "__main__":
    detections = detect_objects("assets/images/test_image.jpg", "models/yolo_v8.pth")
    print(detections)
