from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:/Users/patel/PythonProjects/potholes/Dataset_Info_with_paths.csv")


base_path = "C:/Users/patel/PythonProjects/potholes/Unified Dataset/Unified Dataset"

#create ImageDataGenerator
train_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    rescale=1./255
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

# Load the saved model
loaded_model = load_model('resnet_1.h5')

# Make predictions on the test data
predictions = loaded_model.predict(test_dir)
predicted_classes = (predictions > 0.5).astype('int32')  # Thresholding for binary classification

# Get the true labels
true_classes = test_data['Pothole'].values

# Generate classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the heatmap for confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Pothole', 'Pothole'], yticklabels=['Non-Pothole', 'Pothole'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()