import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("models/fish_classifier.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Take image path from command line
img_path = sys.argv[1]

# Preprocess image
img = image.load_img(img_path, target_size=(224, 224))  # change to your model input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = labels[np.argmax(predictions)]

print(f"Predicted Fish: {predicted_class}")
