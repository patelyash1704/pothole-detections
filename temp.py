from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
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
    batch_size=64,
    target_size=(64,64),
    shuffle=True
)

val_flow = val_datagen.flow_from_dataframe(
    dataframe=train_data,  # Use train data for validation here (already split)
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=64,
    target_size=(64,64),
    shuffle=False  # No shuffling for validation
)


test_dir = train_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode='raw',
    batch_size=64,
    target_size=(64,64),
    shuffle=False
)

# Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64,64, 3))


for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)  # Add dropout for regularization
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)  # Add dropout for regularization
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(train_flow, validation_data=val_flow, epochs=20)  # Increase epochs

# Prediction and confusion matrix
test_loss, test_accuracy = model.evaluate(test_dir)
pred = model.predict(test_dir)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

model.save('resnet50_3.h5')
