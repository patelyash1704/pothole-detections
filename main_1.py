import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
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

total_samples = len(images)
train_data_percentage = (len(train_images) / total_samples) * 100
test_data_percentage = (len(test_images) / total_samples) * 100
validation_data_percentage = (len(val_images) / total_samples) * 100

print("Train Data Percentage:", len(train_images))
print("Test Data Percentage:", len(test_images))
print("Validation Data Percentage:", len(val_images))


#CNN model
model = Sequential([
    Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), 1, activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),
    Conv2D(64, (3,3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

#compilation
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

hist = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels),batch_size=4)

plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)


predictions = model.predict(test_images)
y_pred = (predictions > 0.5).astype(int).flatten()
f1 = f1_score(test_labels, y_pred)
print("F1 Score:", f1)


cm = confusion_matrix(test_labels, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt=".2f", xticklabels=['Normal', 'Pothole'], yticklabels=['Normal', 'Pothole'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Accuracy)')
plt.show()



model.save("pothole_model_4.h5")