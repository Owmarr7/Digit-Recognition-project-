
import os
import ssl
import bz2
import certifi
import urllib.request

import numpy as np
import tensorflow as tf


# ==========================================
# 1. Configuration
# ==========================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

USPS_TRAIN_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/"
    "datasets/multiclass/usps.bz2"
)

USPS_TEST_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/"
    "datasets/multiclass/usps.t.bz2"
)

USPS_TRAIN_FILE = os.path.join(DATA_DIR, "usps.bz2")
USPS_TEST_FILE = os.path.join(DATA_DIR, "usps.t.bz2")

EPOCHS = 5
BATCH_SIZE = 64


# ==========================================
# 2. Download USPS safely
# ==========================================

def download_file(url, destination):

    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"File already exists: {destination}")
        return

    print(f"Downloading {url}")

    context = ssl.create_default_context(
        cafile=certifi.where()
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib.request.urlopen(
            request,
            context=context,
            timeout=60
        ) as response:

            with open(destination, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    file.write(chunk)

        print("Download completed!")

    except Exception:
        if os.path.exists(destination):
            os.remove(destination)
        raise


# ==========================================
# 3. Load USPS dataset
# ==========================================

def load_usps(filepath):

    images = []
    labels = []

    print(f"Loading {filepath}")

    with bz2.open(filepath, "rt") as file:

        for line in file:

            parts = line.strip().split()

            if not parts:
                continue

            # CORRECT LABEL MAPPING:
            # USPS labels: 1-10
            # Actual digits: 0-9

            label = int(float(parts[0])) - 1

            if not 0 <= label <= 9:
                raise ValueError(f"Invalid USPS label: {label}")

            # Create 256 pixel values
            pixels = np.zeros(256, dtype=np.float32)

            for item in parts[1:]:

                index, value = item.split(":")

                index = int(index) - 1
                value = float(value)

                if not 0 <= index < 256:
                    raise ValueError("Invalid pixel index")

                pixels[index] = value

            # Reshape into 16x16 image
            image = pixels.reshape(16, 16)

            # Convert pixel range [-1, 1] to [0, 1]
            image = (image + 1.0) / 2.0

            image = np.clip(image, 0.0, 1.0)

            # Resize to MNIST dimensions: 28x28
            image = tf.image.resize(
                image[..., np.newaxis],
                (28, 28)
            ).numpy()

            images.append(image)
            labels.append(label)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )


# ==========================================
# 4. Load MNIST
# ==========================================

print("\nLoading MNIST...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize pixels
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("MNIST training:", x_train.shape)
print("MNIST testing:", x_test.shape)


# ==========================================
# 5. Download USPS
# ==========================================

print("\nDownloading USPS...")

download_file(USPS_TRAIN_URL, USPS_TRAIN_FILE)
download_file(USPS_TEST_URL, USPS_TEST_FILE)


# ==========================================
# 6. Load USPS
# ==========================================

print("\nLoading USPS...")

x_usps_train, y_usps_train = load_usps(USPS_TRAIN_FILE)

x_usps_test, y_usps_test = load_usps(USPS_TEST_FILE)

print("USPS training:", x_usps_train.shape)
print("USPS testing:", x_usps_test.shape)

print("USPS labels:", np.unique(y_usps_test))


# ==========================================
# 7. Build CNN
# ==========================================

print("\nBuilding CNN model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        10, activation="softmax"
    )
])


# ==========================================
# 8. Compile model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 9. Train on MNIST ONLY
# ==========================================

print("\nTraining on MNIST...")

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)


# ==========================================
# 10. Evaluate MNIST
# ==========================================

print("\nEvaluating MNIST...")

mnist_loss, mnist_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)


# ==========================================
# 11. Evaluate USPS
# ==========================================

print("\nEvaluating USPS...")

usps_loss, usps_accuracy = model.evaluate(
    x_usps_test,
    y_usps_test,
    verbose=1
)


# ==========================================
# 12. Display predictions
# ==========================================

print("\nChecking first 20 USPS predictions...")

predictions = model.predict(
    x_usps_test[:20],
    verbose=0
)

predicted_labels = np.argmax(predictions, axis=1)

print("Predicted:", predicted_labels)
print("Actual:   ", y_usps_test[:20])


# ==========================================
# 13. Final comparison
# ==========================================

print("\n" + "=" * 45)
print("              FINAL RESULTS")
print("=" * 45)

print(f"MNIST Accuracy: {mnist_accuracy * 100:.2f}%")
print(f"USPS Accuracy:  {usps_accuracy * 100:.2f}%")

difference = mnist_accuracy - usps_accuracy

print(f"Accuracy Difference: {difference * 100:.2f} percentage points")

print("=" * 45)


# ==========================================
# 14. Save model
# ==========================================

model.save("digit_recognition_model.keras")

print("\nModel saved successfully!")
