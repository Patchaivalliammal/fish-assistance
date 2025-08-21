import io
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


MODEL_PATH = 'models/fish_classifier.h5'
LABELS_PATH = 'labels.txt'
KB_PATH = 'kb/fish_knowledge.json'
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
try:
model = tf.keras.models.load_model(MODEL_PATH)
return model
except Exception as e:
st.error(f"Could not load model: {e}")
return None


@st.cache_resource
def load_labels():
try:
with open(LABELS_PATH, 'r') as f:
labels = [line.strip() for line in f if line.strip()]
return labels
except FileNotFoundError:
return []


@st.cache_resource
def load_kb():
try:
with open(KB_PATH, 'r', encoding='utf-8') as f:
return json.load(f)
except FileNotFoundError:
return {}


model = load_model()
labels = load_labels()
kb = load_kb()


st.set_page_config(page_title='Smart Fish Assistant', page_icon='🐟')
st.title('🐟 Smart Fish Assistant')
st.caption('Identify fish ➜ Clean it right ➜ Cook something delicious')


uploaded = st.file_uploader('Upload a fish image', type=['jpg', 'jpeg', 'png'])


col1, col2 = st.columns(2)
with col1:
conf_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.4, 0.01)
with col2:
topk = st.selectbox('Show top‑K predictions', options=[1, 2, 3], index=2)


if uploaded is not None:
img = Image.open(uploaded).convert('RGB')
st.image(img, caption='Uploaded image', use_container_width=True)


if model is None or not labels: