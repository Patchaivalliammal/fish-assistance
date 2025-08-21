import io
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Paths
MODEL_PATH = 'models/fish_classifier.h5'
LABELS_PATH = 'labels.txt'
KB_PATH = 'kb/fish_knowledge.json'
IMG_SIZE = (224, 224)


# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


# Load labels
@st.cache_resource
def load_labels():
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels
    except FileNotFoundError:
        return []


# Load knowledge base
@st.cache_resource
def load_kb():
    try:
        with open(KB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# Initialize
model = load_model()
labels = load_labels()
kb = load_kb()

# Streamlit UI
st.set_page_config(page_title='Smart Fish Assistant', page_icon='🐟')
st.title('🐟 Smart Fish Assistant')
st.caption('Identify fish ➜ Clean it right ➜ Cook something delicious')

# File uploader
uploaded = st.file_uploader('Upload a fish image', type=['jpg', 'jpeg', 'png'])

# Settings
col1, col2 = st.columns(2)
with col1:
    conf_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.4, 0.01)
with col2:
    topk = st.selectbox('Show top-K predictions', options=[1, 2, 3], index=2)


# Prediction
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_container_width=True)

    if model is None or not labels:
        st.error("Model or labels not available.")
    else:
        # Preprocess
        img_resized = img.resize(IMG_SIZE)
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        preds = model.predict(arr)[0]
        top_indices = preds.argsort()[-topk:][::-1]

        st.subheader("Predictions:")
        for idx in top_indices:
            label = labels[idx] if idx < len(labels) else "Unknown"
            conf = preds[idx]
            if conf >= conf_threshold:
                st.write(f"✅ {label} ({conf:.2f})")

                # Show knowledge base info
                if label in kb:
                    st.info(f"**Cleaning tip:** {kb[label].get('cleaning', 'No info')}")
                    st.success(f"**Recipe idea:** {kb[label].get('recipe', 'No info')}")

img = Image.open(uploaded).convert('RGB')
st.image(img, caption='Uploaded image', use_container_width=True)



if model is None or not labels:
