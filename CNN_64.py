import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

# Directory containing images without people
no_people_dir = 'human detection dataset/0'

# Directory containing images with people
with_people_dir = 'human detection dataset/1'

# Function to load and resize images
def load_resize_images(directory, target_size=(64, 64)):
    images = []
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Rescale pixel values to [0, 1]
        images.append(img_array)
    return np.array(images)

# Load and resize images without people
no_people_images = load_resize_images(no_people_dir)

# Load and resize images with people
with_people_images = load_resize_images(with_people_dir)

# Concatenate the images
X = np.concatenate([no_people_images, with_people_images])

# Create labels for the images
y = np.concatenate([np.zeros(len(no_people_images)), np.ones(len(with_people_images))])

# Shuffle the data
shuffle_indices = np.random.permutation(len(X))
X_shuffled = X[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Model Selection
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
history = model.fit(
    X_shuffled, y_shuffled,
    batch_size=20,
    epochs=30
)

# Plotting the training loss
plt.plot(history.history['loss'], label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()