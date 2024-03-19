import os
import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import matplotlib.pyplot as plt

# Function to resize images to 128x128 and flatten them
def resize_images_flatten(directory):
    flattened_images = []
    labels = []
    for label in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, label)):
            img = cv2.imread(os.path.join(directory, label, filename))
            img = cv2.resize(img, (128, 128))
            flattened_images.append(img.flatten())  # Flattening the image
            labels.append(int(label))
    return np.array(flattened_images), np.array(labels)

# Function to resize images to 128x128 and extract HOG features
def resize_images_hog(directory):
    hog_features = []
    labels = []
    for label in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, label)):
            img = cv2.imread(os.path.join(directory, label, filename))
            img = cv2.resize(img, (128, 128))
            # Convert image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # orientations = number of sells in the vector of the cumpute
            # pixels_per_cell = how many pixels in each sell we compute
            # cells_per_block = one sell for a block
            hog_feature = hog(img_gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
            hog_features.append(hog_feature)
            labels.append(int(label))
    return np.array(hog_features), np.array(labels)


# Load and preprocess data
data_dir = 'human detection dataset'

# Method 1: Flattening
X_flatten, y_flatten = resize_images_flatten(data_dir)

# Method 2: HOG
X_hog, y_hog = resize_images_hog(data_dir)

# Split data into train and test sets for Flattening
X_train_flatten, X_test_flatten, y_train_flatten, y_test_flatten = train_test_split(X_flatten, y_flatten, test_size=0.2, random_state=42)

# Split data into train and test sets for HOG
X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(X_hog, y_hog, test_size=0.2, random_state=42)

# Method 1: Train AdaBoost classifier using flattened images
adaboost_flatten = AdaBoostClassifier(n_estimators=50)
adaboost_flatten.fit(X_train_flatten, y_train_flatten)

# Method 1: Predictions on train and test sets
train_predictions_flatten = adaboost_flatten.predict(X_train_flatten)
test_predictions_flatten = adaboost_flatten.predict(X_test_flatten)

# Method 1: Calculate accuracy
train_accuracy_flatten = accuracy_score(y_train_flatten, train_predictions_flatten)
test_accuracy_flatten = accuracy_score(y_test_flatten, test_predictions_flatten)

# Method 2: Train AdaBoost classifier using HOG features
adaboost_hog = AdaBoostClassifier(n_estimators=50)
adaboost_hog.fit(X_train_hog, y_train_hog)

# Method 2: Predictions on train and test sets
train_predictions_hog = adaboost_hog.predict(X_train_hog)
test_predictions_hog = adaboost_hog.predict(X_test_hog)

# Method 2: Calculate accuracy
train_accuracy_hog = accuracy_score(y_train_hog, train_predictions_hog)
test_accuracy_hog = accuracy_score(y_test_hog, test_predictions_hog)

# Print the accuracies
print("Method 1 (Flattening):")
print("Train Accuracy:", train_accuracy_flatten)
print("Test Accuracy:", test_accuracy_flatten)
print("\nMethod 2 (HOG):")
print("Train Accuracy:", train_accuracy_hog)
print("Test Accuracy:", test_accuracy_hog)

# Plotting the results
methods = ['Method 1 (Flattening)', 'Method 2 (HOG)']
train_accuracies = [train_accuracy_flatten, train_accuracy_hog]
test_accuracies = [test_accuracy_flatten, test_accuracy_hog]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(methods, train_accuracies, color='skyblue')
plt.title('Train Accuracies')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.bar(methods, test_accuracies, color='salmon')
plt.title('Test Accuracies')

plt.show()
