from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="dataset_custom.yaml", epochs=100, imgsz=640,
             batch=16, device=0, workers = 1)