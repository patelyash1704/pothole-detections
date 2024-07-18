from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
data = tf.keras.utils.image_dataset_from_directory('C:/Users/patel/PythonProjects/potholes/dataset1', batch_size=4, image_size=(256,256))
data = data.map(lambda x,y: (x/255, y))


images = []
labels = []
for image, label in data:
    images.append(image)
    labels.append(label)

images = np.concatenate(images)
labels = np.concatenate(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, shuffle=True,random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25,shuffle=True, random_state=42)


# Load the pre-trained ResNet model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Data preprocessing and feature extraction
train_features = resnet_model.predict(train_images)
val_features = resnet_model.predict(val_images)
test_features = resnet_model.predict(test_images)

# Define and train a classifier
inputs = tf.keras.Input(shape=train_features.shape[1:])
x = GlobalAveragePooling2D()(inputs)
x= Dense(64, activation='relu')(x)
x= Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

classifier_model = Model(inputs, outputs)
classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = classifier_model.fit(train_features, train_labels, epochs=110, validation_data=(val_features, val_labels))

# Evaluate the model
test_loss, test_accuracy = classifier_model.evaluate(test_features, test_labels)
print("Test Accuracy:", test_accuracy)

# Generate classification report and confusion matrix
predictions = classifier_model.predict(test_features)
binary_predictions = np.round(predictions).astype(int)
print(classification_report(test_labels, binary_predictions))
cm = confusion_matrix(test_labels, binary_predictions)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

classifier_model.save("resnet_1.h5")
