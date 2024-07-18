import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:/Users/patel/PythonProjects/potholes/Dataset_Info_with_paths.csv")

base_path = "C:/Users/patel/PythonProjects/potholes/Unified Dataset/Unified Dataset"


train_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # Split data into training and validation
    rescale=1./255  # Normalize pixel values to [0, 1]
)


train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_dir = train_gen.flow_from_dataframe(
    dataframe=train_data,
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=32,
    target_size=(128, 128),
    shuffle=True,
    subset='training'
)

val_dir = train_gen.flow_from_dataframe(
    dataframe=train_data,
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=32,
    target_size=(128, 128),
    shuffle=True,
    subset='validation'
)

test_dir = train_gen.flow_from_dataframe(
    dataframe=test_data,
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=32,
    target_size=(128, 128),
    shuffle=False
)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    train_dir,
    epochs=15,
    steps_per_epoch=len(train_dir),
    validation_data=val_dir,
    validation_steps=len(val_dir)
)

test_loss, test_acc = model.evaluate(test_dir, steps=len(test_dir))
print("Test Accuracy:", test_acc)

fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

img = cv2.imread('C:/Users/patel/PythonProjects/potholes/Unified Dataset/Unified Dataset/pothole_image_1804.jpg')
plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (128,128))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)
if yhat > 0.5:
    print(f'Predicted class is pothole')
else:
    print(f'Predicted class is normal')


model.save("pothole_model_5.h5")

