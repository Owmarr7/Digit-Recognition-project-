
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from streamlit_drawable_canvas import st_canvas

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("AI Handwritten Digit Recognition")
st.write(
    "Recognize handwritten digits using your trained "
    "Convolutional Neural Network."
)

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_digit_model.keras")

model = load_model()

# =====================================
# 3. IMAGE PREPROCESSING
# =====================================

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")
    img = np.array(image)

    # Detect background brightness using border pixels
    border = np.concatenate([
        img[0, :],
        img[-1, :],
        img[:, 0],
        img[:, -1]
    ])

    # Make background black and digit white
    if np.median(border) > 127:
        img = 255 - img

    # Detect digit boundaries
    binary = img > 30

    if not binary.any():
        return None

    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]

    # Crop around the digit
    cropped = img[
        rows[0]:rows[-1] + 1,
        cols[0]:cols[-1] + 1
    ]

    # Resize while preserving proportions
    h, w = cropped.shape
    scale = 20.0 / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = Image.fromarray(cropped).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    resized = np.array(resized).astype("float32") / 255.0

    # Center digit on a 28x28 image
    final_image = np.zeros((28, 28), dtype="float32")

    x = (28 - new_w) // 2
    y = (28 - new_h) // 2

    final_image[y:y + new_h, x:x + new_w] = resized

    return final_image.reshape(1, 28, 28, 1)


# =====================================
# 4. PREDICTION FUNCTION
# =====================================

def predict_digit(image):

    prediction = model.predict(image, verbose=0)[0]

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    return digit, confidence, prediction


# =====================================
# 5. DRAWING TAB
# =====================================

tab1, tab2 = st.tabs([
    "Draw a Digit",
    "Upload an Image"
])

with tab1:

    st.subheader("Draw your own digit")

    st.write("Use your mouse to draw a number from 0 to 9.")

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        return_image_data=True
    )

    if st.button("Predict Drawing", type="primary"):

        if canvas.image_data is None:
            st.warning("Please draw a digit first.")

        else:
            image = Image.fromarray(
                canvas.image_data.astype("uint8")
            ).convert("L")

            processed = preprocess_image(image)

            if processed is None:
                st.warning("No digit detected. Try drawing again.")

            else:
                digit, confidence, probabilities = predict_digit(
                    processed
                )

                st.success(f"Recognized Digit: {digit}")

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Prediction Probabilities")

                for i, probability in enumerate(probabilities):
                    st.write(
                        f"Digit {i}: {probability * 100:.2f}%"
                    )

                    st.progress(float(probability))


# =====================================
# 6. UPLOAD IMAGE TAB
# =====================================

with tab2:

    st.subheader("Upload a handwritten digit")

    st.write(
        "Upload an image containing one handwritten digit. "
        "You can use your own handwriting or a USPS sample."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=200
        )

        if st.button("Predict Uploaded Image", type="primary"):

            processed = preprocess_image(image)

            if processed is None:
                st.warning("No digit detected in this image.")

            else:
                digit, confidence, probabilities = predict_digit(
                    processed
                )

                st.success(f"Recognized Digit: {digit}")

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Prediction Probabilities")

                for i, probability in enumerate(probabilities):
                    st.write(
                        f"Digit {i}: {probability * 100:.2f}%"
                    )

                    st.progress(float(probability))


# =====================================
# 7. FOOTER
# =====================================

st.divider()

st.caption(
    "CNN | TensorFlow | Keras | MNIST | USPS"
)
