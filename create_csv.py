import os
import cv2
import imghdr
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

# Load the CSV file
csv_file_path = r"C:/Users/patel/PythonProjects/potholes/Dataset_Info.csv"
df = pd.read_csv(csv_file_path)

# Function to construct full path for each image
def construct_full_path(Image_ID):
    base_path = r"C:/Users/patel/PythonProjects/potholes/Unified Dataset/Unified Dataset"
    return os.path.join(base_path, f"{Image_ID}.jpg")  # Assuming images have .jpg extension

# Apply the function to create a new column with full paths
df['FullImagePath'] = df['Image_ID'].apply(construct_full_path)

# Optionally, you can save the modified DataFrame back to a CSV file
output_csv_path = r"C:/Users/patel/PythonProjects/potholes/Dataset_Info_with_paths.csv"
df.to_csv(output_csv_path, index=False)

# Display the DataFrame with the new column
print(df)