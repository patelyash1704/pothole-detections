from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:/Users/patel/PythonProjects/potholes/Dataset_Info_with_paths.csv")

base_path = "C:/Users/patel/PythonProjects/potholes/Unified Dataset/Unified Dataset"

# Separate ImageDataGenerator instances
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Flow of the data with separate generators
train_flow = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=16,  # Reduce batch size for efficiency
    target_size=(64, 64),  # Reduce image size for efficiency
    shuffle=True
)

val_flow = val_datagen.flow_from_dataframe(
    dataframe=train_data,  # Use train data for validation here (already split)
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=16,  # Reduce batch size for efficiency
    target_size=(64, 64),  # Reduce image size for efficiency
    shuffle=False  # No shuffling for validation
)


test_dir = train_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=32,
    target_size=(64, 64),  # Reduce image size for efficiency
    shuffle=False
)

# Model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

for layer in vgg16.layers:
    layer.trainable = False


x = GlobalAveragePooling2D()(vgg16.output)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=vgg16.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(train_flow, validation_data=val_flow, epochs=10)  # Reduce epochs to avoid overfitting

# Prediction and confusion matrix
test_loss, test_acc = model.evaluate(test_dir)
print("Test Accuracy:", test_acc)

predictions = model.predict(test_dir)
y_pred = np.round(predictions)

model.save('vgg_1.h5')
