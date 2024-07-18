from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file containing information about the dataset
df = pd.read_csv(r"C:/Users/patel/PythonProjects/potholes/Dataset_Info_with_paths.csv")

# Define the base path for images
base_path = "C:/Users/patel/PythonProjects/potholes/Unified Dataset/Unified Dataset"

# Create an ImageDataGenerator
test_gen = ImageDataGenerator(rescale=1./255)  # No need for other augmentations for testing

# Create a DataFrameIterator for the testing data
test_dir = test_gen.flow_from_dataframe(
    dataframe=df,
    directory=base_path,
    x_col='FullImagePath',
    y_col='Pothole',
    class_mode=None,  # Set to None to avoid labels generation
    batch_size=32,
    target_size=(128, 128),
    shuffle=False  # Do not shuffle the test data
)

# Load the pre-trained VGG16 model
model = load_model('vgg_1.h5')

# Generate predictions for the test data
predictions = model.predict(test_dir)

# Convert probabilities to binary labels
predicted_labels = (predictions > 0.5).astype(int)

# Extract true labels from file paths or filenames
true_labels = np.array(df['Pothole'])

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Generate classification report
class_report = classification_report(true_labels, predicted_labels)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=['Non-Pothole', 'Pothole'],
            yticklabels=['Non-Pothole', 'Pothole'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()