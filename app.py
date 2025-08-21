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


# Load Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


# Load Labels
@st.cache_resource
def load_labels():
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels
    except FileNotFoundError:
        return []


# Load Knowledge Base
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

st.set_page_config(page_title='Smart Fish Assistant', page_icon='🐟')
st.title('🐟 Smart Fish Assistant')
st.caption('Identify fish ➜ Clean it right ➜ Cook something delicious')


# File uploader
uploaded = st.file_uploader('Upload a fish image', type=['jpg', 'jpeg', 'png'])

col1, col2 = st.columns(2)
with col1:
    conf_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.4, 0.01)
with col2:
    topk = st.selectbox('Show top-K predictions', options=[1, 2, 3], index=2)


if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_container_width=True)

    if model is not None and labels:
        # Preprocess
        img_resized = img.resize(IMG_SIZE)
        x = np.array(img_resized) / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict
        preds = model.predict(x)[0]
        top_indices = preds.argsort()[-topk:][::-1]

        st.subheader("Predictions")
        for idx in top_indices:
            conf = preds[idx]
            if conf < conf_threshold:
                continue
            fish_name = labels[idx]
            st.write(f"**{fish_name}** ({conf:.2f} confidence)")

            # Fetch KB info safely
            fish_info = kb.get(fish_name, {})
            cleaning = fish_info.get("cleaning", "No cleaning info available for this fish.")
            recipe = fish_info.get("recipe", "No recipe info available for this fish.")

            with st.expander(f"ℹ️ Info about {fish_name}"):
                st.write(f"🧽 **Cleaning tip:** {cleaning}")
                st.write(f"🍲 **Recipe idea:** {recipe}")

    else:
        st.warning("⚠️ Model or labels not loaded correctly. Please check your files.")




if model is None or not labels:

