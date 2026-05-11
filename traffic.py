import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Load image data from directory
    images, labels = load_data(sys.argv[1])

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels)

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Train model
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model if filename provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` contains one directory named after each category,
    numbered 0 through NUM_CATEGORIES - 1.

    Return tuple `(images, labels)`.
    """

    images = []
    labels = []

    # Loop through each category folder
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        # Skip if folder doesn't exist
        if not os.path.isdir(category_path):
            continue

        # Loop through all files in the category
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)

            # Read image
            image = cv2.imread(file_path)

            # Skip unreadable images
            if image is None:
                continue

            # Resize to 30x30
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Add to dataset
            images.append(image)
            labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model.
    """

    model = tf.keras.models.Sequential([

        # Convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3),
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(
            64, (3, 3),
            activation="relu"
        ),

        # Second pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten
        tf.keras.layers.Flatten(),

        # Dense hidden layer
        tf.keras.layers.Dense(128, activation="relu"),

        # Dropout to reduce overfitting
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
