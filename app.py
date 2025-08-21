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

# ---------------------------
# Load Model, Labels, KB
# ---------------------------
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

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title='Smart Fish Assistant', page_icon='🐟')
st.title('🐟 Smart Fish Assistant')
st.caption('Identify fish ➜ Clean it right ➜ Cook something delicious')

uploaded = st.file_uploader('Upload a fish image', type=['jpg', 'jpeg', 'png'])

col1, col2 = st.columns(2)
with col1:
    conf_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.4, 0.01)
with col2:
    topk = st.selectbox('Show top-K predictions', options=[1, 2, 3], index=2)

# ---------------------------
# Prediction
# ---------------------------
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_container_width=True)

    if model is None or not labels:
        st.error("Model or labels not available!")
    else:
        # Preprocess image
        img_resized = img.resize(IMG_SIZE)
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        top_indices = preds.argsort()[-topk:][::-1]

        st.subheader("🔎 Predictions")
        for i in top_indices:
            conf = preds[i]
            if conf >= conf_threshold:
                fish_name = labels[i]
                st.write(f"**{fish_name}** ({conf:.2f})")

                if fish_name in kb:
                    st.info(f"🧹 Cleaning: {kb[fish_name].get('cleaning', 'No info')}")
                    st.success(f"🍲 Recipe: {kb[fish_name].get('recipe', 'No info')}")
                else:
                    st.warning("⚠️ No knowledge found for this fish.")

# ---------------------------
# Manual Knowledge Base Input
# ---------------------------
st.subheader("📝 Add Fish Knowledge")

with st.form("add_fish_info"):
    new_fish = st.text_input("Fish name")
    new_cleaning = st.text_area("Cleaning tip")
    new_recipe = st.text_area("Recipe idea")
    submitted = st.form_submit_button("Save")

    if submitted:
        if new_fish.strip():
            kb[new_fish] = {
                "cleaning": new_cleaning if new_cleaning.strip() else "No cleaning info available.",
                "recipe": new_recipe if new_recipe.strip() else "No recipe info available."
            }
            # Save to JSON file
            with open(KB_PATH, 'w', encoding='utf-8') as f:
                json.dump(kb, f, indent=4, ensure_ascii=False)

            st.success(f"✅ Info for '{new_fish}' saved successfully!")
        else:
            st.error("⚠️ Please enter a fish name before saving.")


