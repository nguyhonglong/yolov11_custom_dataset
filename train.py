from ultralytics import YOLO
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize model
model = YOLO(config['model']['path'])

# Train model
model.train(
    data=config['data']['path'],
    epochs=config['train']['epochs'],
    imgsz=config['data']['image_size'],
    batch=config['train']['batch_size'],
    device=config['train']['device'],
    workers=config['train']['workers']
)