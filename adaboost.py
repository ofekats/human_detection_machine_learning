import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import matplotlib.pyplot as plt

# Function to extract HOG features from images
def extract_hog_features(images, image_size):
    features = []
    for image in images:
        # Resize image
        resized_image = cv2.resize(image, image_size)
        # Extract HOG features
        hog_features = hog(resized_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features.append(hog_features)
    return np.array(features)

# Function to load images and corresponding labels
def load_data(data_dir, image_size):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(int(label))
    # Convert images to HOG feature vectors
    features = extract_hog_features(images, image_size)
    return features, np.array(labels)

# Load data
data_dir = 'human detection dataset'

# Load data with images resized to 128x128
features_128, labels_128 = load_data(data_dir, (128, 128))

# Split data into training and testing sets
X_train_128, X_test_128, y_train_128, y_test_128 = train_test_split(features_128, labels_128, test_size=0.2, random_state=42)

# Initialize AdaBoost classifier for 128x128
adaboost_128 = AdaBoostClassifier()

# Train AdaBoost classifier with HOG features of size 128x128
print("Training AdaBoost with HOG features of size 128x128...")
adaboost_128.fit(X_train_128, y_train_128)

# Predictions on training set for 128x128
y_train_pred_128 = adaboost_128.predict(X_train_128)

# Training accuracy for 128x128
training_accuracy_128 = accuracy_score(y_train_128, y_train_pred_128)
print("Training Accuracy with HOG features of size 128x128:", training_accuracy_128)

# Predictions on test set for 128x128
y_pred_128 = adaboost_128.predict(X_test_128)

# Testing accuracy for 128x128
testing_accuracy_128 = accuracy_score(y_test_128, y_pred_128)
print("Testing Accuracy with HOG features of size 128x128:", testing_accuracy_128)

# Load data with images resized to 64x64
features_64, labels_64 = load_data(data_dir, (64, 64))

# Split data into training and testing sets for 64x64
X_train_64, X_test_64, y_train_64, y_test_64 = train_test_split(features_64, labels_64, test_size=0.2, random_state=42)

# Initialize AdaBoost classifier for 64x64
adaboost_64 = AdaBoostClassifier()

# Train AdaBoost classifier with HOG features of size 64x64
print("Training AdaBoost with HOG features of size 64x64...")
adaboost_64.fit(X_train_64, y_train_64)

# Predictions on training set for 64x64
y_train_pred_64 = adaboost_64.predict(X_train_64)

# Training accuracy for 64x64
training_accuracy_64 = accuracy_score(y_train_64, y_train_pred_64)
print("Training Accuracy with HOG features of size 64x64:", training_accuracy_64)

# Predictions on test set for 64x64
y_pred_64 = adaboost_64.predict(X_test_64)

# Testing accuracy for 64x64
testing_accuracy_64 = accuracy_score(y_test_64, y_pred_64)
print("Testing Accuracy with HOG features of size 64x64:", testing_accuracy_64)

# Plot results
labels = ['128x128 Training', '128x128 Testing', '64x64 Training', '64x64 Testing']
accuracies = [training_accuracy_128, testing_accuracy_128, training_accuracy_64, testing_accuracy_64]

plt.bar(labels, accuracies, color=['blue', 'green', 'red', 'orange'])
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Score')
plt.title('Training and Testing Accuracy of AdaBoost with HOG features')
plt.show()
