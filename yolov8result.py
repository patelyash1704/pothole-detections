from ultralytics import YOLO
import numpy as np
model=YOLO('C:/Users/patel/PythonProjects/potholes/runs/classify/train/weights/best.pt')
results=model("C:/Users/patel/PythonProjects/potholes/dataset1/Pothole/Pothole12.jpg")
names_dict = results[0].names
probs=results[0].probs.tolist()
print(probs)
print(names_dict[np.argmax(probs)])
