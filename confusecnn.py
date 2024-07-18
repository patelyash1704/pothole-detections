import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the saved CNN model
cnn_model = tf.keras.models.load_model('pothole_model_4.h5')



data = tf.keras.utils.image_dataset_from_directory('C:/Users/patel/PythonProjects/potholes/dataset1', batch_size=4, image_size=(256,256))
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

pred=cnn_model.predict(test_images)
pred_labels = (pred > 0.5).astype(int)
print(classification_report(test_labels,pred_labels,labels=[0,1]))




#Confusion matrix and classification report for CNN model
cnn_conf_matrix = confusion_matrix(test_labels, pred_labels)

#
#
cm = confusion_matrix(test_labels, pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt=".2f", xticklabels=['Normal', 'Pothole'], yticklabels=['Normal', 'Pothole'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Accuracy)')
plt.show()


