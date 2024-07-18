import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
filename='rf.sav'
model=pickle.load(open(filename,'rb'))
img = cv2.imread('C:/Users/patel/PythonProjects/potholes/dataset1/Pothole/Pothole21.jpg')
resized_img = cv2.resize(img, (256, 256))  # Resize the image to match the dataset's image size
flatten_img = resized_img.flatten()  # Flatten the image
input_img = flatten_img / 255.0  # Normalize the pixel values
plt.imshow(img)
plt.show()
# Make predictions
yhat = model.predict(np.expand_dims(input_img, axis=0))
print(yhat)
if yhat > 0.5:
    print(f'Predicted class is pothole')
else:
    print(f'Predicted class is normal')