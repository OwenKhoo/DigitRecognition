import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import models, layers, utils

# Load the MNIST dataset
# (X_train, y_train) are the training images and labels
# (X_test, y_test) are the testing images and labels
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display the shape of the training and testing datasets
print("Shape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)

# Plot the first 4 training images with their labels
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title("Label: {}".format(y_train[i]))
    plt.axis("off")
plt.tight_layout()
plt.show()

# Normalize the pixel values of the training and testing datasets to the range [-0.5, 0.5]
# Original pixel values are in the range [0, 255]
X_train = (X_train / 255) - 0.5
X_test = (X_test / 255) - 0.5

# Add a channel dimension to the data to match the input shape required by Conv2D
# Shape changes from (28, 28) to (28, 28, 1)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Print the new shape of the training and testing datasets after adding the channel dimension
print("Shape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)

# Build a Convolutional Neural Network (CNN) model
model = models.Sequential([
    # First convolutional layer with 32 filters and ReLU activation
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    # Second convolutional layer with 32 filters
    layers.Conv2D(32, 3, activation='relu'),
    # Max pooling layer to reduce spatial dimensions
    layers.MaxPooling2D(pool_size=2),
    # Dropout layer to prevent overfitting
    layers.Dropout(0.25),
    
    # Third convolutional layer with 64 filters
    layers.Conv2D(64, 3, activation='relu'),
    # Fourth convolutional layer with 64 filters
    layers.Conv2D(64, 3, activation='relu'),
    # Max pooling layer to further reduce spatial dimensions
    layers.MaxPooling2D(pool_size=2),
    # Dropout layer to prevent overfitting
    layers.Dropout(0.25),
    
    # Flatten the feature maps into a 1D vector
    layers.Flatten(),
    # Fully connected layer with 128 neurons and ReLU activation
    layers.Dense(128, activation='relu'),
    # Dropout layer to prevent overfitting
    layers.Dropout(0.5),
    # Output layer with 10 neurons (one for each digit) and softmax activation
    layers.Dense(10, activation='softmax')
])

# Compile the model using the Adam optimizer, categorical crossentropy loss, and accuracy metric
model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model on the training data
# Convert y_train and y_test to one-hot encoded format using utils.to_categorical()
model.fit(X_train, utils.to_categorical(y_train), epochs=10, validation_data=(X_test, utils.to_categorical(y_test)))

# Randomly select 10 test samples for predictions
index = np.random.randint(0, 10001, size=10)
test_data = np.array([X_test[i] for i in index])
test_labels = np.array([y_test[i] for i in index])

# Predict the classes of the selected test images
predictions = model.predict(test_data)

# Print the predicted and actual labels of the selected test samples
print("Predicted:", np.argmax(predictions, axis=1))
print("Actual:", test_labels)

# Save the trained model in TensorFlow format
model.save("model", save_format="tf")