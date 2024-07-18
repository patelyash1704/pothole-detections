from ultralytics import YOLO
import numpy
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

results = model.train(data='C:/Users/patel/PythonProjects/potholes/split_dataset', epochs=10, imgsz=124)