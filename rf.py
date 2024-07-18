from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('C:/Users/patel/PythonProjects/potholes/dataset1', batch_size=4, image_size=(256, 256))
data = data.map(lambda x, y: (x / 255, y))

images = []
labels = []
for image, label in data:
    for img in image:
        images.append(np.array(img))
    labels.extend(label.numpy())

images = np.array(images)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, shuffle=True, random_state=42)

total_samples = len(images)
train_data_percentage = (len(train_images) / total_samples) * 100
test_data_percentage = (len(test_images) / total_samples) * 100
validation_data_percentage = (len(val_images) / total_samples) * 100

print("Train Data:", len(train_images))
print("Test Data :", len(test_images))
print("Validation Data :", len(val_images))

model = RandomForestClassifier(criterion='gini', max_depth=10, max_features="sqrt", min_samples_leaf=1,
                               min_samples_split=2, n_estimators=71)

history = model.fit(train_images.reshape(len(train_images), -1), train_labels)

pred = model.predict(test_images.reshape(len(test_images), -1))
print(classification_report(test_labels, pred, labels=[0, 1]))
print(confusion_matrix(test_labels, pred))

# Save the trained model
filename = 'rf_1.sav'
pickle.dump(model, open(filename, 'wb'))

# Evaluate the model on the validation set
val_pred = model.predict(val_images.reshape(len(val_images), -1))
print("Validation Report:")
print(classification_report(val_labels, val_pred, labels=[0, 1]))
print("Validation Confusion Matrix:")
print(confusion_matrix(val_labels, val_pred))

# Plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()
