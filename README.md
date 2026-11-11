# Handwritten Digit Recognition (CNN, MNIST + USPS)

A convolutional neural network that recognizes handwritten digits (0-9), built with TensorFlow/Keras. The model is trained on MNIST, tested on a second dataset (USPS) to check how well it generalizes, and wrapped in a small web app where you can draw a digit and get a prediction.

## What it does

- Loads and normalizes MNIST (28x28 grayscale digits).
- Trains a CNN: 2 x (Conv2D + MaxPooling), then Dense(128) with Dropout(0.3), then a 10-class softmax. Optimizer: Adam, 5 epochs, 10% validation split.
- Evaluates on the MNIST test set.
- Tests the same model, without retraining, on USPS. USPS is downloaded automatically from the LIBSVM dataset collection; its pixel values are rescaled to [0, 1] and the 16x16 images are resized to 28x28 to match MNIST.
- Streamlit app (`src/test.py`): draw or upload a digit. The app detects the background, crops the digit, resizes it while keeping proportions, and centers it before prediction.

## Project structure

```
src/main.py     # data loading, model, training, evaluation, saving
src/test.py     # Streamlit drawing / upload app
digit_recognition_model.keras   # trained model (saved by main.py)
Requirements.txt
```

## Technologies

Python, TensorFlow/Keras, NumPy, Matplotlib, Pillow, Streamlit, certifi

## How to run

```bash
pip install -r Requirements.txt

# train and evaluate (prints MNIST and USPS accuracy, saves the model)
python src/main.py

# launch the drawing app (run from the repo root so the model file is found)
streamlit run src/test.py
```

## Results

- **MNIST test accuracy: 99.26%** (5 epochs)
- **USPS accuracy: 81.42%** (model trained on MNIST only, no retraining). The 17.84-point drop shows the effect of dataset shift: USPS images are lower resolution (16x16) and come from a different source.

## Notes

This was my first machine learning project. Possible next steps: data augmentation, a deeper model, and comparing against a PyTorch version.
