from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
model.train(data="data.yaml", epochs=100, imgsz=640)