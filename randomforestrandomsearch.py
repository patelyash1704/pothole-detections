from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
import pickle
import cv2
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

n_estimators=[int(x) for x in np.linspace(start=10,stop=120,num=10)]
max_features=['sqrt']
min_samples_split=[2,5]
min_samples_leaf=[1,2]
max_depth=[6,8,10]

#create a random grid
param_grid={'n_estimators': n_estimators,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'max_depth': max_depth}
print(param_grid)
model = RandomForestClassifier(criterion='gini')
rf_grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=3,verbose=2,n_jobs=4)

rf_grid.fit(train_images.reshape(len(train_images), -1), train_labels)
print(rf_grid.best_params_)

pred = model.predict(test_images.reshape(len(test_images), -1))
print(classification_report(test_labels,pred,labels=[0,1]))
# filename = 'rf.sav'
# pickle.dump(model, open(filename, 'wb'))

