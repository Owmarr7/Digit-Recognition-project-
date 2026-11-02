
import tensorflow as tf
import importlib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

try:
    tfds = importlib.import_module("tensorflow_datasets")
except ModuleNotFoundError:
    tfds = None

# 1. Load MNIST
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize MNIST
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# 2. Load USPS
if tfds is not None:
    usps_train, usps_test = tfds.load(
        "usps",
        split=["train", "test"],
        as_supervised=True
    )

def prepare_usps(dataset):
    images = []
    labels = []

    for image, label in tfds.as_numpy(dataset):
        image = image.astype("float32")

        # USPS is normally black digits on white background.
        # Invert to match MNIST: white digits on black background.
        image = 1.0 - image

        # Resize from 16x16 to 28x28
        image = tf.image.resize(
            image, (28, 28)
        ).numpy()

        images.append(image)
        labels.append(label)

    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    return images, labels

print("MNIST training:", x_train.shape)
print("MNIST testing:", x_test.shape)
if tfds is not None:
    x_usps_train, y_usps_train = prepare_usps(usps_train)
    x_usps_test, y_usps_test = prepare_usps(usps_test)
    print("USPS testing:", x_usps_test.shape)
else:
    print("tensorflow-datasets is unavailable; skipping USPS steps.")

# 3. Build CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# 4. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train ONLY on MNIST
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 6. Evaluate on MNIST
mnist_loss, mnist_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print(f"MNIST accuracy: {mnist_accuracy * 100:.2f}%")

# 7. Evaluate on USPS without retraining when tensorflow-datasets is available.
if tfds is not None:
    usps_loss, usps_accuracy = model.evaluate(
        x_usps_test, y_usps_test, verbose=0
    )

    print(f"USPS accuracy: {usps_accuracy * 100:.2f}%")

    # 8. Display USPS predictions
    predictions = model.predict(x_usps_test[:10], verbose=0)

    for i in range(min(10, len(x_usps_test))):
        predicted = np.argmax(predictions[i])

        plt.imshow(
            x_usps_test[i].reshape(28, 28),
            cmap="gray"
        )

        plt.title(
            f"Predicted: {predicted} | Actual: {y_usps_test[i]}"
        )

        plt.axis("off")
        plt.show()

# 9. Save model
model.save("mnist_digit_model.keras")

print("Model saved successfully!")
