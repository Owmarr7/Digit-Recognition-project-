
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -------------------------
# 1. Page configuration
# -------------------------

st.set_page_config(
    page_title="AI Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("AI Handwritten Digit Recognition")
st.write("Draw a digit from 0 to 9 and let the CNN predict it.")

# -------------------------
# 2. Load trained model
# -------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_digit_model.keras")

model = load_model()

# -------------------------
# 3. Drawing canvas
# -------------------------

st.subheader("Draw your digit below")

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    return_image_data=True
)

# -------------------------
# 4. Prediction
# -------------------------

if st.button("Predict Digit", type="primary"):

    if canvas.image_data is None:
        st.warning("Please draw a digit first.")

    else:
        # Convert canvas to image
        image = Image.fromarray(
            canvas.image_data.astype(np.uint8)
        ).convert("L")

        # Resize to MNIST image size
        image = image.resize((28, 28))

        # Normalize pixels
        image_array = np.array(image).astype("float32") / 255.0

        # Prepare input for CNN
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict digit
        predictions = model.predict(image_array, verbose=0)

        digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        # Display result
        st.divider()
        st.subheader("Prediction Result")

        st.success(f"Predicted Digit: {digit}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        # Display probabilities
        st.subheader("All Digit Probabilities")

        for i, probability in enumerate(predictions[0]):
            st.write(f"Digit {i}: {probability * 100:.2f}%")
            st.progress(float(probability))

# -------------------------
# 5. Footer
# -------------------------

st.divider()
st.caption("Built with Python, TensorFlow, Keras and Streamlit")
