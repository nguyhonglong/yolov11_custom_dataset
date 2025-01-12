import os
from ultralytics import YOLO
import cv2

def detect_objects(image_path, model_path, save_dir='results'):

    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)
    
    # Get output filename
    filename = os.path.basename(image_path)
    output_path = os.path.join(save_dir, filename)
    
    # Save results (includes bounding boxes)
    results[0].save(output_path)
    
    return output_path

if __name__ == "__main__":
    # Example usage
    image_path = "datasets/Fog/Fog/foggy-001.jpg"  # Replace with your image path
    model_path = "runs/detect/train/weights/best.pt"
    
    output_path = detect_objects(image_path, model_path)
    print(f"Detection results saved to: {output_path}")
