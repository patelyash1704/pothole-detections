import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
model=load_model("pothole_model_4.h5")
img = cv.imread('C:/Users/patel/OneDrive/Documents/pythonaiml/potholes/Dataset/test/pothole843.jpg')

plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))

yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)
if yhat > 0.5:
    print(f'detected class is pothole')
else:
    print(f'detected class is normal')


